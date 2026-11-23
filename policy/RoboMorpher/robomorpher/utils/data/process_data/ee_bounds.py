"""Data-derived EE workspace bounds shared by action and text-state guards."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np


def resolve_ee_slot_bounds(*, stats_path=None, stats=None, key, num_arms=2,
                           value_low=None, value_high=None):
    """Return xyz raw min/max across present arms, plus fixed rot6d bounds.

    Min/max/count must be computed after removing absent-arm padding. q01/q99
    are never used here; they remain the independent action normalization.
    """
    if (value_low is None) != (value_high is None):
        raise ValueError("EE value_low/value_high must be supplied together")
    if value_low is not None:
        low = np.asarray(value_low, dtype=np.float32)
        high = np.asarray(value_high, dtype=np.float32)
        source = {"kind": "explicit_config", "column": key}
    else:
        if stats is None:
            if stats_path is None:
                raise ValueError("EE bounds require masked raw min/max stats or explicit value_low/value_high")
            raw_bytes = Path(stats_path).read_bytes()
            raw = json.loads(raw_bytes)
            source_sha = hashlib.sha256(raw_bytes).hexdigest()
        else:
            raw = stats
            source_sha = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if not isinstance(raw, Mapping) or not isinstance(raw.get(key), Mapping):
            raise ValueError(f"Masked raw statistics missing {key!r}")
        record = raw[key]
        width = int(num_arms) * 10
        minima = np.asarray(record.get("min"), dtype=np.float64)
        maxima = np.asarray(record.get("max"), dtype=np.float64)
        count = np.asarray(record.get("count"), dtype=np.float64)
        if minima.shape != (width,) or maxima.shape != (width,) or count.shape != (width,):
            raise ValueError(f"Masked EE {key} min/max/count must have shape [{width}]")
        if (not np.isfinite(minima).all() or not np.isfinite(maxima).all()
                or not np.isfinite(count).all() or np.any(count < 0)
                or np.any(count != np.floor(count))):
            raise ValueError("Masked EE raw bounds/count must be finite with nonnegative integer counts")
        minima, maxima, count = (a.reshape(int(num_arms), 10) for a in (minima, maxima, count))
        if np.any(count != count[:, :1]):
            raise ValueError("Masked EE arm slots must have matching valid counts")
        present = count[:, 0] > 0
        if not np.any(present):
            raise ValueError("Masked EE bounds have no valid arm observations")
        if np.any(maxima[present] < minima[present]):
            raise ValueError("Masked EE raw max is below min")
        xyz_low = minima[present, :3].min(axis=0)
        xyz_high = maxima[present, :3].max(axis=0)
        low = np.asarray([*xyz_low, *([-1.] * 6)], dtype=np.float32)
        high = np.asarray([*xyz_high, *([1.] * 6)], dtype=np.float32)
        source = {"kind": "masked_raw_minmax", "column": key,
                  "stats_path": str(Path(stats_path).resolve()) if stats_path else None,
                  "stats_sha256": source_sha, "valid_counts_by_arm": count[:, 0].astype(np.int64).tolist(),
                  "axes": "shared xyz bounds across present arms; no absent-arm padding",
                  "rot6d": "physical [-1,1]"}
    if (low.shape != (9,) or high.shape != (9,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(high <= low)):
        raise ValueError("EE slot bounds must have nine finite values with high > low")
    if not np.array_equal(low[3:], np.full(6, -1.)) or not np.array_equal(high[3:], np.ones(6)):
        raise ValueError("EE rot6d physical bounds must remain [-1,1]")
    source.update(value_low=low.tolist(), value_high=high.tolist())
    semantic = {k: v for k, v in source.items() if k != "stats_path"}
    source["fingerprint"] = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return low.tolist(), high.tolist(), source


def load_ee_state_quantiles(*, stats_path=None, stats=None):
    """Load per-arm observed-state q01/q99. Never read action quantiles."""
    if isinstance(stats, Mapping) and stats.get("normalization") == "state_q01_q99":
        spec = json.loads(json.dumps(stats, allow_nan=False))
        claimed = spec.pop("fingerprint", None)
        semantic = {k: v for k, v in spec.items() if k != "stats_path"}
        actual = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if claimed != actual:
            raise ValueError("Pinned state normalization fingerprint is invalid")
        spec["fingerprint"] = claimed
        q01, q99 = np.asarray(spec.get("q01"), dtype=np.float32), np.asarray(spec.get("q99"), dtype=np.float32)
        counts = np.asarray(spec.get("counts"), dtype=np.float64)
    else:
        if stats is None:
            if stats_path is None:
                raise ValueError("state_q01_q99 requires observed-state masked statistics")
            content = Path(stats_path).read_bytes()
            raw = json.loads(content)
            source_sha = hashlib.sha256(content).hexdigest()
        else:
            raw = stats
            source_sha = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if not isinstance(raw, Mapping) or not isinstance(raw.get("observation.state"), Mapping):
            raise ValueError("state_q01_q99 requires observation.state statistics; action stats are not a fallback")
        record = raw["observation.state"]
        low, high = np.asarray(record.get("q01"), dtype=np.float32), np.asarray(record.get("q99"), dtype=np.float32)
        count20 = np.asarray(record.get("count"), dtype=np.float64)
        if low.shape != (20,) or high.shape != (20,) or count20.shape != (20,):
            raise ValueError("Observed EE state q01/q99/count must have 20 per-arm slots")
        low, high, count20 = (v.reshape(2, 10) for v in (low, high, count20))
        if np.any(count20 != count20[:, :1]):
            raise ValueError("Masked observed-state arm counts must match across all ten slots")
        q01, q99, counts = low[:, :9], high[:, :9], count20[:, :9]
        spec = {"normalization": "state_q01_q99", "source_column": "observation.state",
                "stats_path": str(Path(stats_path).resolve()) if stats_path else None,
                "stats_sha256": source_sha, "quantile_method": record.get("quantile_method", raw.get("quantile_method", "linear")),
                "layout": "arm_major_xyz3_rot6d6", "q01": q01.tolist(), "q99": q99.tolist(),
                "counts": counts.astype(np.int64).tolist(), "constant_slot_policy": "center_bin_for_finite_valid_values",
                "physical_checks": "finite xyz; rot6d[-1,1] tolerance1e-6; normalized gripper[0,1]",
                "tail_policy": "clip only quantile token coordinates to [0,1]",
                "gripper_policy": "independent uniform opening bins; no quantile normalization"}
    if (q01.shape != (2, 9) or q99.shape != (2, 9) or counts.shape != (2, 9)
            or not np.isfinite(q01).all() or not np.isfinite(q99).all()
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or np.any(q99 < q01)):
        raise ValueError("State quantiles/counts must be finite per-arm [2,9], q99>=q01, integer count>=0")
    if not np.any(counts > 0):
        raise ValueError("State normalization has no valid observations")
    if spec.get("constant_slot_policy") != "center_bin_for_finite_valid_values":
        raise ValueError("Unsupported constant state-quantile policy")
    spec["constant_slots"] = (q99 == q01).tolist()
    semantic = {k: v for k, v in spec.items() if k not in {"stats_path", "fingerprint"}}
    fingerprint = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if "fingerprint" in spec and spec["fingerprint"] != fingerprint:
        raise ValueError("Pinned state normalization semantics are inconsistent")
    spec["fingerprint"] = fingerprint
    return q01.tolist(), q99.tolist(), spec
