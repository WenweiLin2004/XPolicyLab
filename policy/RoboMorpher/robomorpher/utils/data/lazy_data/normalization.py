"""Versioned normalization fitting, cache identity, and stat loading for EAT.

New fitting reads only valid training joints, with shared or per-slot channel
statistics. Tensor-only [J,D] dictionaries remain readable; new [D]
statistics require explicit schema metadata. Metadata stays in Python objects,
never in torch.as_tensor. Denominators retain the historical range + 1e-6 and
sqrt(population variance) + 1e-6 formulas; epsilon is not a clamp floor.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
import torch


NORM_SCHEMA_VERSION = 2
NUMERIC_STAT_KEYS = tuple(f"{prefix}_{suffix}" for prefix in ("x", "y")
                          for suffix in ("mean", "std", "min", "max", "q01", "q99"))
NORMALIZATION_METHODS = ("minmax", "zscore", "q99")
NORMALIZATION_SCOPES = ("shared_feature", "per_slot")
ANGLE_WRAP_MODES = ("none",)
CONSTANT_POLICY = {
    "minmax_denominator": "range_plus_1e-6",
    "zscore_denominator": "sqrt_population_variance_plus_1e-6",
    "constant_minmax_value": -1.0,
    "padding_after_normalization": "masked_fill_zero",
}


def validate_angle_wrap(angle_wrap):
    """Validate the serialized joint-angle convention; never guess unknown modes."""
    if angle_wrap not in ANGLE_WRAP_MODES:
        raise ValueError(f"Unknown joint angle_wrap: {angle_wrap!r}; expected {ANGLE_WRAP_MODES}")
    return angle_wrap


def norm_angle_wrap(stats):
    """Return the only supported convention: raw physical joint radians."""
    metadata = stats.get("metadata")
    # Metadata-free statistics are accepted as raw because the old wrapped
    # preprocessing is intentionally no longer available at runtime.
    return validate_angle_wrap("none" if metadata is None else metadata.get("angle_wrap"))


def coerce_norm_stats(raw, normalization_method=None):
    """Copy only known numerical statistics to CPU FP32; preserve metadata.

    The returned mapping is suitable for a checkpoint. Unknown keys raise so
    future fields cannot silently be lost or interpreted as tensor statistics.
    ``method`` is accepted as a runtime alias of normalization_method.
    """
    if not isinstance(raw, dict):
        raise TypeError("Normalization statistics must be a dictionary")
    allowed = set(NUMERIC_STAT_KEYS) | {"metadata", "normalization_method", "method"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown normalization fields: {sorted(unknown)}")
    result = {}
    for key in NUMERIC_STAT_KEYS:
        if key in raw:
            value = raw[key]
            result[key] = (value.detach().cpu().float().clone() if torch.is_tensor(value)
                           else torch.as_tensor(value, dtype=torch.float32).clone())
    for prefix in ("x", "y"):
        for suffix in ("mean", "std", "min", "max"):
            if f"{prefix}_{suffix}" not in result:
                raise ValueError(f"Missing normalization statistic: {prefix}_{suffix}")
    method = raw.get("normalization_method", raw.get("method"))
    if normalization_method is not None:
        if method is not None and str(method) != str(normalization_method):
            raise ValueError("Requested normalization method differs from frozen statistics")
        method = normalization_method
    if method is not None:
        method = str(method)
        if method not in NORMALIZATION_METHODS:
            raise ValueError(f"Unknown normalization method: {method!r}")
        result["normalization_method"] = method
    if "metadata" in raw:
        if not isinstance(raw["metadata"], dict):
            raise TypeError("Normalization metadata must be a Python dictionary")
        result["metadata"] = copy.deepcopy(raw["metadata"])
    return result


def validate_norm_stats(stats, *, max_num_joints, urdf_rotation_6d,
                        normalization_method, expected_metadata=None, angle_wrap=None,
                        input_representation="joint"):
    input_representation = str(input_representation).lower()
    """Validate shape/representation/method. Metadata-free [J,D] stats stay per-slot.

    Only a cache computed for this dataset receives expected_metadata. Frozen
    training stats remain reusable when evaluating an independent dataset.
    """
    frozen_angle_wrap = norm_angle_wrap(stats)
    if angle_wrap is not None and validate_angle_wrap(angle_wrap) != frozen_angle_wrap:
        raise ValueError("Requested angle_wrap differs from frozen normalization statistics")
    metadata = stats.get("metadata")
    if metadata is not None:
        # Tokenizer checkpoints produced before schema-v2 added the explicit
        # input_representation field.  Their representation/output pair
        # unambiguously identifies the joint contract.
        metadata = dict(metadata)
        if "input_representation" not in metadata:
            rep = metadata.get("representation")
            out_rep = metadata.get("output_representation")
            if rep == "origin_xyz_rpy_q" and out_rep == "active_child_xyz_rot6d_q":
                metadata["input_representation"] = "joint"
    if metadata is None:
        scope = "per_slot"
        if any(value.ndim != 2 for key, value in stats.items() if key in NUMERIC_STAT_KEYS):
            raise ValueError("Metadata-free normalization must use [J,D] statistics")
        if expected_metadata is not None:
            raise ValueError("A metadata-free stats cache cannot satisfy a new schema-v2 request")
    else:
        if metadata.get("schema_version") != NORM_SCHEMA_VERSION:
            raise ValueError("Unsupported normalization schema_version")
        scope = metadata.get("scope")
        if scope not in NORMALIZATION_SCOPES:
            raise ValueError("Invalid normalization scope")
        representation = ("urdf_ee" if input_representation == "urdf_ee" else
                         "ee_only" if input_representation == "ee_only" else
                         "origin_xyz_rot6d_q" if urdf_rotation_6d else "origin_xyz_rpy_q")
        expected = {
            "representation": representation,
            "output_representation": ("active_child_xyz_rot6d" if input_representation in {"urdf_ee", "ee_only"} else "active_child_xyz_rot6d_q"),
        "input_representation": input_representation,
            "normalization_method": normalization_method,
            "max_num_joints": int(max_num_joints),
            "angle_wrap": frozen_angle_wrap,
            "valid_rule": "joint_valid_mask_true_training_rows_only",
            "constant_policy": CONSTANT_POLICY,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Normalization metadata mismatch: {key}")
        split = metadata.get("train_split")
        if not isinstance(split, dict) or split.get("algorithm") != "global_modulo_holdout_v1":
            raise ValueError("Missing/unknown normalization train split identity")
        if not isinstance(split.get("index_digest"), str) or not split.get("index_digest"):
            raise ValueError("Normalization train split is missing its dataset index digest")
        if expected_metadata is not None:
            for key, value in expected_metadata.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Normalization cache identity mismatch: {key}")
    if stats.get("normalization_method", normalization_method) != normalization_method:
        raise ValueError("Frozen normalization method does not match the requested method")
    dx = {"urdf_ee": 15, "ee_only": 9}.get(input_representation, 10 if urdf_rotation_6d else 7)
    dy = 9 if input_representation in {"urdf_ee", "ee_only"} else 10
    for prefix, width in (("x", dx), ("y", dy)):
        expected_shape = (width,) if scope == "shared_feature" else (int(max_num_joints), width)
        suffixes = ("mean", "std", "min", "max") + (("q01", "q99") if normalization_method == "q99" else ())
        for suffix in suffixes:
            key = f"{prefix}_{suffix}"
            value = stats.get(key)
            if value is None or tuple(value.shape) != expected_shape:
                raise ValueError(f"norm_stats[{key!r}] must have shape {expected_shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite normalization statistic: {key}")
        if not bool((stats[f"{prefix}_std"] > 0).all()):
            raise ValueError(f"Normalization {prefix}_std must be positive")
        if bool((stats[f"{prefix}_max"] < stats[f"{prefix}_min"]).any()):
            raise ValueError(f"Normalization {prefix}_min exceeds max")
        if normalization_method == "q99" and bool((stats[f"{prefix}_q99"] < stats[f"{prefix}_q01"]).any()):
            raise ValueError(f"Normalization {prefix}_q01 exceeds q99")
    return scope


def load_norm_stats_cache(path, *, normalization_method=None):
    """Read metadata-free or v2 NPZ without pickle or blind metadata-to-tensor casts."""
    with np.load(path, allow_pickle=False) as loaded:
        raw = {}
        for key in loaded.files:
            if key in NUMERIC_STAT_KEYS:
                raw[key] = loaded[key].copy()
            elif key == "metadata_json":
                raw["metadata"] = json.loads(str(loaded[key].item()))
            elif key == "normalization_method":
                raw[key] = str(loaded[key].item())
            else:
                raise ValueError(f"Unknown normalization cache field: {key}")
    return coerce_norm_stats(raw, normalization_method=normalization_method)


def norm_stats_cache_payload(stats):
    """Return only numeric arrays and JSON/text arrays for np.savez."""
    normalized = coerce_norm_stats(stats)
    payload = {key: value.numpy() for key, value in normalized.items() if key in NUMERIC_STAT_KEYS}
    if "metadata" in normalized:
        payload["metadata_json"] = np.asarray(json.dumps(normalized["metadata"], sort_keys=True, separators=(",", ":")))
    if "normalization_method" in normalized:
        payload["normalization_method"] = np.asarray(normalized["normalization_method"])
    return payload


def split_period(test_split):
    """The dataset's existing deterministic global-row holdout rule."""
    value = float(test_split)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("test_split must be finite and in [0,1)")
    return max(2, round(1.0 / value)) if value > 0 else 0


def _read_index(data_dir):
    with open(os.path.join(data_dir, "index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Dataset index must contain nonempty ordered shards")
    total = sum(int(shard["rows"]) for shard in shards)
    if any(int(shard["rows"]) < 0 for shard in shards) or total <= 0:
        raise ValueError("Dataset shard row counts must be nonnegative with a positive total")
    if int(index.get("total_rows", total)) != total:
        raise ValueError("Dataset index total_rows does not match its ordered shards")
    return index, total


def validate_training_angle_wrap(angle_wrap):
    """Reject wrapped angles in every new tokenizer training run."""
    if angle_wrap != "none":
        raise ValueError(
            "Tokenizer training requires joint_angle_wrap=none; "
            "principal-angle preprocessing is removed; only raw physical radians are supported."
        )
    return "none"


def stats_request_metadata(data_dir, *, normalization_method="minmax", stats_scope="shared_feature",
                           urdf_rotation_6d=False, max_num_joints=None, test_split=0.01,
                           input_representation="joint",
                           quantile=None, angle_wrap="none"):
    angle_wrap = validate_angle_wrap(angle_wrap)
    if max_num_joints is None or int(max_num_joints) < 1:
        raise ValueError("max_num_joints is required and must be positive")
    if stats_scope not in NORMALIZATION_SCOPES:
        raise ValueError("normalization_scope must be shared_feature or per_slot")
    if normalization_method not in NORMALIZATION_METHODS:
        raise ValueError("Unknown normalization method")
    expected_quantile = 0.99 if normalization_method == "q99" else None
    if quantile is not None and quantile != expected_quantile:
        raise ValueError("quantile must match normalization_method (only q99 uses 0.99)")
    index, total = _read_index(data_dir)
    period = split_period(test_split)
    test_rows = (total + period - 1) // period if period else 0
    index_digest = hashlib.sha256(json.dumps(index, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": NORM_SCHEMA_VERSION,
        "scope": stats_scope,
        "normalization_method": normalization_method,
        "representation": ("urdf_ee" if input_representation == "urdf_ee" else
                          "ee_only" if input_representation == "ee_only" else
                          "origin_xyz_rot6d_q" if urdf_rotation_6d else "origin_xyz_rpy_q"),
        "output_representation": ("active_child_xyz_rot6d" if input_representation in {"urdf_ee", "ee_only"} else "active_child_xyz_rot6d_q"),
        "input_representation": input_representation,
        "max_num_joints": int(max_num_joints),
        "angle_wrap": angle_wrap,
        "valid_rule": "joint_valid_mask_true_training_rows_only",
        "constant_policy": copy.deepcopy(CONSTANT_POLICY),
        "quantile": expected_quantile,
        "quantile_estimator": "histogram_bin_centers_4096" if expected_quantile is not None else None,
        "train_split": {
            "algorithm": "global_modulo_holdout_v1", "period": period,
            "total_rows": total, "train_rows": total - test_rows,
            "dataset_root": os.path.realpath(data_dir), "index_digest": index_digest,
        },
    }


def stats_cache_filename(metadata):
    # Observed token count is recorded after fitting, not part of the request.
    # The saved metadata and the pre-fit request must resolve to the same path.
    request = {key: value for key, value in metadata.items() if key != "valid_train_tokens"}
    identity = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    rotation = "rot6d" if metadata["representation"] == "origin_xyz_rot6d_q" else "rpy"
    validate_angle_wrap(metadata["angle_wrap"])
    angle_tag = "raw"
    return (f"norm_stats_v2_{metadata['scope']}_{rotation}_J{metadata['max_num_joints']}_"
            f"{metadata['normalization_method']}_{angle_tag}_train_{identity}.npz")


def _selected_blocks(args):
    """Read bounded blocks; discard held-out rows before any feature transform."""
    import zarr
    from .zarr_dataset import _truncate_joint_arrays, wrap_joint_angles
    from .masking import binary_joint_mask
    from robomorpher.utils.geometry.rot6d import urdf_rpy_to_rot6d_np

    path, offset, expected_rows, metadata = args
    group = zarr.open_group(path, mode="r")["data"]
    xa, ya, ma = group["input"], group["output"], group["joint_valid_mask"]
    if (len(xa.shape) != 3 or len(ya.shape) != 3 or len(ma.shape) != 2
            or tuple(xa.shape[:2]) != tuple(ma.shape) or tuple(ya.shape[:2]) != tuple(ma.shape)
            or xa.shape[-1] != 7 or ya.shape[-1] != 10 or xa.shape[0] != expected_rows):
        raise ValueError(f"Dataset arrays disagree with the indexed [N,J,7]/[N,J,10]/[N,J] contract: {path}")
    period = metadata["train_split"]["period"]
    for start in range(0, expected_rows, 4096):
        end = min(start + 4096, expected_rows)
        selected = np.ones(end - start, dtype=bool) if period == 0 else (np.arange(offset + start, offset + end) % period != 0)
        if not selected.any():
            continue
        x = np.asarray(xa[start:end], dtype=np.float64)[selected]
        y = np.asarray(ya[start:end], dtype=np.float64)[selected]
        mask = binary_joint_mask(
            np.asarray(ma[start:end])[selected], where=f"stats/{os.path.basename(path)}"
        )
        x, y, mask = _truncate_joint_arrays(x, y, mask, metadata["max_num_joints"], where=f"stats/{os.path.basename(path)}")
        # np.where removes NaN/Inf padding before rotation/angle conversions;
        # multiplying by zero would keep those nonfinite values alive.
        if metadata.get("input_representation") == "urdf_ee":
            x = np.concatenate([x[..., :6], y[..., :9]], axis=-1)
            y = y[..., :9]
        elif metadata.get("input_representation") == "ee_only":
            x = y[..., :9]
            y = y[..., :9]
        x, y = np.where(mask[..., None], x, 0.0), np.where(mask[..., None], y, 0.0)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite valid training features in {path}")
        if metadata["representation"] == "origin_xyz_rot6d_q":
            x = urdf_rpy_to_rot6d_np(x)
        x, y = wrap_joint_angles(x, y, angle_wrap=metadata["angle_wrap"])
        yield x, y, mask


def _empty_partial(metadata):
    shared = metadata["scope"] == "shared_feature"
    dx = {"urdf_ee": 15, "ee_only": 9}.get(metadata.get("input_representation"), 10 if metadata["representation"] == "origin_xyz_rot6d_q" else 7)
    dy = 9 if metadata.get("input_representation") in {"urdf_ee", "ee_only"} else 10
    shape_prefix = () if shared else (metadata["max_num_joints"],)
    result = {"count": np.zeros((), dtype=np.float64) if shared
              else np.zeros((metadata["max_num_joints"], 1), dtype=np.float64)}
    for prefix, width in (("x", dx), ("y", dy)):
        shape = shape_prefix + (width,)
        result.update({f"{prefix}_sum": np.zeros(shape), f"{prefix}_sumsq": np.zeros(shape),
                       f"{prefix}_min": np.full(shape, np.inf), f"{prefix}_max": np.full(shape, -np.inf)})
    return result


def _shard_stats(args):
    """Raw moments/ranges over exactly the selected training valid joints."""
    metadata = args[3]
    result = _empty_partial(metadata)
    shared = metadata["scope"] == "shared_feature"
    for x, y, mask in _selected_blocks(args):
        result["count"] += mask.sum() if shared else mask.sum(axis=0)[:, None]
        for prefix, values in (("x", x), ("y", y)):
            if shared:
                values = values[mask]
                if not len(values):
                    continue
                minimum, maximum = values.min(axis=0), values.max(axis=0)
            else:
                minimum = np.where(mask[..., None], values, np.inf).min(axis=0)
                maximum = np.where(mask[..., None], values, -np.inf).max(axis=0)
                values = np.where(mask[..., None], values, 0.0)
            result[f"{prefix}_sum"] += values.sum(axis=0)
            result[f"{prefix}_sumsq"] += np.square(values).sum(axis=0)
            np.minimum(result[f"{prefix}_min"], minimum, out=result[f"{prefix}_min"])
            np.maximum(result[f"{prefix}_max"], maximum, out=result[f"{prefix}_max"])
    return result


_HIST_CONTEXT = {}


def _hist_initializer(x_edges, y_edges):
    _HIST_CONTEXT["x"] = x_edges
    _HIST_CONTEXT["y"] = y_edges


def _shard_hist(args):
    edges = _HIST_CONTEXT
    result = {prefix: np.zeros(array.shape[:-1] + (array.shape[-1] - 1,), dtype=np.int64)
              for prefix, array in edges.items()}
    shared = args[3]["scope"] == "shared_feature"
    for x, y, mask in _selected_blocks(args):
        for prefix, values in (("x", x), ("y", y)):
            if shared:
                selected = values[mask]
                for d in range(values.shape[-1]):
                    result[prefix][d] += np.histogram(selected[:, d], bins=edges[prefix][d])[0]
            else:
                for j in range(values.shape[1]):
                    selected = values[mask[:, j], j]
                    for d in range(values.shape[-1]):
                        result[prefix][j, d] += np.histogram(selected[:, d], bins=edges[prefix][j, d])[0]
    return result


def _quantile_from_hist(hist, edges, q, fallback):
    cdf = np.cumsum(hist, axis=-1)
    total = cdf[..., -1]
    idx = np.clip((cdf < (q * total)[..., None]).sum(axis=-1), 0, hist.shape[-1] - 1)
    centers = 0.5 * (edges[..., :-1] + edges[..., 1:])
    values = np.take_along_axis(centers, idx[..., None], axis=-1)[..., 0]
    return np.where(total > 0, values, fallback)


def compute_norm_stats(data_dir, workers=16, urdf_rotation_6d=False, max_num_joints=None,
                       quantile=None, *, test_split=0.01, normalization_method=None,
                       stats_scope="shared_feature", angle_wrap="none", input_representation="joint"):
    """Version-2 valid-train-only statistics, independently for x/y channels.

    per_slot is an optional v2 control, also excluding test and padding. It does
    not recreate historical full-dataset/padding-contaminated caches.
    """
    method = normalization_method or ("q99" if quantile is not None else "minmax")
    validate_training_angle_wrap(angle_wrap)
    metadata = stats_request_metadata(data_dir, normalization_method=method, stats_scope=stats_scope,
                                      urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                                      test_split=test_split, quantile=quantile, angle_wrap=angle_wrap, input_representation=input_representation)
    original_request = copy.deepcopy(metadata)
    index, _ = _read_index(data_dir)
    args, offset = [], 0
    for shard in index["shards"]:
        args.append((os.path.join(data_dir, shard["path"]), offset, int(shard["rows"]), metadata))
        offset += int(shard["rows"])
    print(f"[stats] schema=2 scope={stats_scope}, train_rows={metadata['train_split']['train_rows']}, "
          f"shards={len(args)}, method={method}, angle_wrap={angle_wrap}; valid joints only")
    aggregate = _empty_partial(metadata)

    def merge(partial):
        for key in ("count", "x_sum", "x_sumsq", "y_sum", "y_sumsq"):
            aggregate[key] += partial[key]
        for prefix in ("x", "y"):
            np.minimum(aggregate[f"{prefix}_min"], partial[f"{prefix}_min"], out=aggregate[f"{prefix}_min"])
            np.maximum(aggregate[f"{prefix}_max"], partial[f"{prefix}_max"], out=aggregate[f"{prefix}_max"])

    if workers <= 1:
        for arg in args:
            merge(_shard_stats(arg))
    else:
        # Ordered merging gives identical arithmetic order across worker counts.
        with Pool(int(workers)) as pool:
            for partial in pool.imap(_shard_stats, args):
                merge(partial)
    valid_tokens = int(aggregate["count"].sum())
    if not valid_tokens:
        raise ValueError("Cannot compute normalization: no valid joints in the training split")
    metadata["valid_train_tokens"] = valid_tokens
    count = np.clip(aggregate["count"], 1.0, None)
    stats = {"metadata": metadata, "normalization_method": method}
    for prefix in ("x", "y"):
        mean = aggregate[f"{prefix}_sum"] / count
        var = np.clip(aggregate[f"{prefix}_sumsq"] / count - mean * mean, 0.0, None)
        stats[f"{prefix}_mean"] = mean.astype(np.float32)
        stats[f"{prefix}_std"] = np.sqrt(var).astype(np.float32) + 1e-6
        for suffix in ("min", "max"):
            value = aggregate[f"{prefix}_{suffix}"].astype(np.float32)
            stats[f"{prefix}_{suffix}"] = np.where(np.isfinite(value), value, 0.0)
    if method == "q99":
        edges = {}
        for prefix in ("x", "y"):
            # Histogram samples are FP64 (including wrapped angles). Building
            # edges from already-rounded FP32 output statistics can exclude the
            # true min/max and collapse a small-sample quantile onto its middle
            # observation. Use the exact fitting extrema until the final save.
            low = aggregate[f"{prefix}_min"]
            high = aggregate[f"{prefix}_max"]
            low = np.where(np.isfinite(low), low, 0.0)
            high = np.where(np.isfinite(high), high, 0.0)
            high = np.where(high > low, high, low + 1.0)
            edges[prefix] = np.linspace(low, high, 4097, axis=-1)
        hist = {prefix: np.zeros(array.shape[:-1] + (4096,), dtype=np.int64) for prefix, array in edges.items()}
        if workers <= 1:
            _hist_initializer(edges["x"], edges["y"])
            for arg in args:
                partial = _shard_hist(arg)
                for prefix in hist:
                    hist[prefix] += partial[prefix]
        else:
            with Pool(int(workers), initializer=_hist_initializer, initargs=(edges["x"], edges["y"])) as pool:
                for partial in pool.imap(_shard_hist, args):
                    for prefix in hist:
                        hist[prefix] += partial[prefix]
        for prefix in ("x", "y"):
            observed = hist[prefix].sum(axis=-1)
            expected_counts = np.broadcast_to(aggregate["count"], observed.shape)
            if not np.array_equal(observed, expected_counts):
                raise RuntimeError(f"Quantile histogram lost valid training samples for {prefix}")
            low, high = stats[f"{prefix}_min"], stats[f"{prefix}_max"]
            qlow = np.clip(_quantile_from_hist(hist[prefix], edges[prefix], 0.01, low), low, high)
            qhigh = np.clip(_quantile_from_hist(hist[prefix], edges[prefix], 0.99, high), low, high)
            # Retain historical q99's range guard and subsequent +1e-6 formula.
            stats[f"{prefix}_q01"] = qlow.astype(np.float32)
            stats[f"{prefix}_q99"] = np.maximum(qhigh, qlow + 1e-6).astype(np.float32)
    final_request = stats_request_metadata(data_dir, normalization_method=method, stats_scope=stats_scope,
                                          urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                                          test_split=test_split, quantile=quantile, angle_wrap=angle_wrap, input_representation=input_representation)
    if final_request != original_request:
        raise RuntimeError("Dataset index changed while fitting normalization; refusing statistics from mixed snapshots")
    return stats


def synchronize_norm_stage(stage, error=None, identity=None):
    """Every rank enters, including failed readers, before the next setup step."""
    record = {"error": error, "identity": identity}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        records = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(records, record)
    else:
        records = [record]
    failed = [(rank, item["error"]) for rank, item in enumerate(records) if item["error"] is not None]
    if failed:
        rank, message = failed[0]
        raise RuntimeError(f"Normalization {stage} failed on rank {rank}: {message}")
    if any(item["identity"] != records[0]["identity"] for item in records[1:]):
        raise RuntimeError(f"Normalization {stage}: cache request identities differ across ranks")


def load_or_compute_stats(data_dir, accelerator, stats_filename=None, workers=16,
                          urdf_rotation_6d=False, max_num_joints=None, quantile=None, *,
                          test_split=0.01, normalization_method=None, stats_scope="shared_feature",
                          angle_wrap="none", input_representation="joint"):
    """Publish a new-schema cache on rank zero; validate identity on every load."""
    method = normalization_method or ("q99" if quantile is not None else "minmax")
    validate_training_angle_wrap(angle_wrap)
    expected, path, identity, error = None, None, None, None
    try:
        expected = stats_request_metadata(data_dir, normalization_method=method, stats_scope=stats_scope,
                                         urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                                         test_split=test_split, quantile=quantile, angle_wrap=angle_wrap, input_representation=input_representation)
        filename = stats_filename or stats_cache_filename(expected)
        if os.path.basename(filename) != filename or not filename.startswith("norm_stats_v2_") or not filename.endswith(".npz"):
            raise ValueError("New normalization caches must use a local norm_stats_v2_*.npz filename; old caches are read-only")
        path = os.path.join(data_dir, filename)
        identity = {"metadata": expected, "cache_path": os.path.realpath(path)}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    synchronize_norm_stage("cache preflight", error, identity)

    def validated_load():
        value = load_norm_stats_cache(path, normalization_method=method)
        validate_norm_stats(value, max_num_joints=max_num_joints, urdf_rotation_6d=urdf_rotation_6d,
                            normalization_method=method, expected_metadata=expected, input_representation=input_representation)
        return value

    error = None
    if accelerator.is_main_process:
        try:
            if os.path.exists(path):
                validated_load()  # Wrong schema/split must fail, never overwrite.
                print(f"[stats] validated existing cache: {path}")
            else:
                value = compute_norm_stats(data_dir, workers=workers, urdf_rotation_6d=urdf_rotation_6d,
                                           max_num_joints=max_num_joints, quantile=quantile,
                                           test_split=test_split, normalization_method=method, stats_scope=stats_scope,
                                           angle_wrap=angle_wrap, input_representation=input_representation)
                validate_norm_stats(coerce_norm_stats(value), max_num_joints=max_num_joints,
                                    urdf_rotation_6d=urdf_rotation_6d, normalization_method=method,
                                    expected_metadata=expected, angle_wrap=angle_wrap,
                                    input_representation=input_representation)
                if stats_request_metadata(data_dir, normalization_method=method, stats_scope=stats_scope,
                                          urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                                          test_split=test_split, quantile=quantile, angle_wrap=angle_wrap, input_representation=input_representation) != expected:
                    raise RuntimeError("Dataset index changed before publishing normalization cache")
                temporary = f"{path}.tmp.{os.getpid()}.npz"
                try:
                    np.savez(temporary, **norm_stats_cache_payload(value))
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                print(f"[stats] saved new cache: {path}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    # Propagate main-process failure before peers enter a success-only barrier.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error is not None:
        raise RuntimeError(f"Normalization cache preparation failed: {error}")
    accelerator.wait_for_everyone()
    loaded, error = None, None
    try:
        loaded = validated_load()
        if stats_request_metadata(data_dir, normalization_method=method, stats_scope=stats_scope,
                                  urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                                  test_split=test_split, quantile=quantile, angle_wrap=angle_wrap,
                                  input_representation=input_representation) != expected:
            raise RuntimeError("Dataset index changed before completing normalization cache load")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    synchronize_norm_stage("cache load", error, identity)
    return loaded


def dataset_norm_checkpoint(dataset):
    """Complete frozen stats, including metadata, for trainer checkpoint saves."""
    return coerce_norm_stats(dataset.norm_stats, normalization_method=dataset.normalization_method)
