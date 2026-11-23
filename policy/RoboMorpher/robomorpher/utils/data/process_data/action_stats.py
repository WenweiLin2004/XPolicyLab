"""Masked action quantiles for slot_q99_q01.

Joint stats keep the historical rule: a joint slot uses joint_valid_mask, and
the gripper is valid when that arm has any valid joint. Padded joints therefore
stay out of q01/q99. A never-observed joint uses neutral bounds [-1, 1]
and a never-observed gripper uses [0, 1], each with count zero. EE stats use the same per-slot quantiles on the 20-d
xyz+rot6d+gripper vector, with each arm's ten slots gated by
observation.arm_valid_mask. An arm that never appears gets the fixed EE bounds
instead of the padded zeros.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from robomorpher.utils.data.process_data.ee_vector import ARM_WIDTH

_KEYS = ("q01", "q99", "min", "max", "mean", "std", "count")
JOINT_MASK_RULE = (
    "arm joints from joint_valid_mask; gripper valid iff arm has any valid joint; "
    "always-masked slots use joint[-1,1] or gripper[0,1] with count=0"
)
EE_MASK_RULE = (
    "per-arm xyz+rot6d+gripper follows observation.arm_valid_mask; "
    "an arm absent from the dataset uses xyz[-2,2], rot6d[-1,1], gripper[0,1]"
)


def _as_2d(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [frames, dim]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _binary_mask(values, name, width):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [frames, {width}]")
    number = np.asarray(array, dtype=np.float32)
    if not np.isfinite(number).all() or not np.all((number == 0) | (number == 1)):
        raise ValueError(f"{name} must be finite and binary")
    return number.astype(bool)


def joint_action_mask(joint_valid_mask):
    """Expand [frames, 2J] joint validity to [frames, 2(J+1)] action validity."""
    joints = np.asarray(joint_valid_mask)
    if joints.ndim != 2 or joints.shape[1] < 2 or joints.shape[1] % 2:
        raise ValueError("joint_valid_mask must have shape [frames, 2J]")
    per_arm = joints.shape[1] // 2
    if joints.dtype != bool:
        joints = _binary_mask(joints, "joint_valid_mask", 2 * per_arm)
    valid = np.empty((joints.shape[0], 2 * (per_arm + 1)), dtype=bool)
    for side in (0, 1):
        block = joints[:, side * per_arm:(side + 1) * per_arm]
        start = side * (per_arm + 1)
        valid[:, start:start + per_arm] = block
        valid[:, start + per_arm] = block.any(axis=1)
    return valid


def ee_action_mask(arm_valid_mask):
    """Repeat each arm flag across xyz+rot6d+gripper."""
    arms = _binary_mask(arm_valid_mask, "arm_valid_mask", 2)
    return np.repeat(arms, ARM_WIDTH, axis=1)


def _ee_absent_bounds(slot):
    local = slot % ARM_WIDTH
    if local < 3:
        return -2.0, 2.0
    if local < 9:
        return -1.0, 1.0
    return 0.0, 1.0


def summarize_masked_action(action, *, action_type, joint_valid_mask=None, arm_valid_mask=None):
    """Return q01/q99/min/max/mean/std/count for each action dimension."""
    action_type = str(action_type).lower()
    if action_type not in {"joint", "ee"}:
        raise ValueError("action_type must be joint or ee")
    action = _as_2d(action, "action")
    if action_type == "joint":
        if joint_valid_mask is None:
            raise ValueError("joint stats require joint_valid_mask")
        valid = joint_action_mask(joint_valid_mask)
        if valid.shape[0] != action.shape[0]:
            raise ValueError("joint_valid_mask frame count does not match action")
    else:
        if arm_valid_mask is None:
            raise ValueError("ee stats require arm_valid_mask")
        valid = ee_action_mask(arm_valid_mask)
        if action.shape[1] != valid.shape[1] or valid.shape[0] != action.shape[0]:
            raise ValueError("ee action must be [frames, 20] and arm_valid_mask [frames, 2]")
    if valid.shape != action.shape:
        raise ValueError(
            f"action width {action.shape[1]} does not match the {action_type} validity mask {valid.shape[1]}"
        )
    if not np.any(valid):
        raise ValueError("Dataset has no valid action observations")
    stats = {key: [] for key in _KEYS}
    for slot in range(action.shape[1]):
        values = action[:, slot][valid[:, slot]]
        if values.size == 0:
            if action_type == "ee":
                low, high = _ee_absent_bounds(slot)
            else:
                arm_width = action.shape[1] // 2
                is_gripper = slot % arm_width == arm_width - 1
                low, high = (0.0, 1.0) if is_gripper else (-1.0, 1.0)
            numbers = (low, high, low, high, (low + high) / 2, 0.0, 0)
        else:
            numbers = (
                float(np.quantile(values, 0.01)),
                float(np.quantile(values, 0.99)),
                float(values.min()),
                float(values.max()),
                float(values.mean()),
                float(values.std()),
                int(values.size),
            )
        for key, number in zip(_KEYS, numbers):
            stats[key].append(number)
    return stats


def _read_action_columns(root, action_type):
    import pyarrow.parquet as pq

    paths = sorted((Path(root) / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files in {root}")
    mask_column = "observation.joint_valid_mask" if action_type == "joint" else "observation.arm_valid_mask"
    actions = []
    masks = []
    for path in paths:
        table = pq.ParquetFile(path).read(columns=["action", mask_column])
        frame = table.to_pydict()
        actions.append(np.stack([np.asarray(row, dtype=np.float32) for row in frame["action"]]))
        masks.append(np.stack([np.asarray(row, dtype=np.float32) for row in frame[mask_column]]))
    return np.concatenate(actions), np.concatenate(masks)


def resolve_action_type(root, action_type=None):
    if action_type is not None:
        action_type = str(action_type).lower()
        if action_type not in {"joint", "ee"}:
            raise ValueError("action_type must be joint or ee")
        return action_type
    metadata_path = Path(root) / "meta" / "dataset_metadata.json"
    if metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8")).get("action_type")
        if recorded is not None:
            return resolve_action_type(root, recorded)
    import pyarrow.parquet as pq

    paths = sorted((Path(root) / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files in {root}")
    names = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    if "observation.joint_valid_mask" in names:
        return "joint"
    if "observation.arm_valid_mask" in names:
        return "ee"
    raise ValueError(f"{root} has neither joint_valid_mask nor arm_valid_mask")


def masked_action_report(root, *, action_type=None):
    root = Path(root)
    action_type = resolve_action_type(root, action_type)
    action, mask = _read_action_columns(root, action_type)
    stats = summarize_masked_action(
        action,
        action_type=action_type,
        joint_valid_mask=mask if action_type == "joint" else None,
        arm_valid_mask=mask if action_type == "ee" else None,
    )
    return {
        "action": stats,
        "source_frames": int(action.shape[0]),
        "action_type": action_type,
        "mask_rule": JOINT_MASK_RULE if action_type == "joint" else EE_MASK_RULE,
        "created_utc": time.strftime("%F %T", time.gmtime()),
    }


def write_masked_action_stats(root, *, action_type=None):
    root = Path(root)
    report = masked_action_report(root, action_type=action_type)
    path = root / "meta" / "stats_masked.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
