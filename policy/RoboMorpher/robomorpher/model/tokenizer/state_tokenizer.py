"""Deterministic bins for observed joint/gripper state, independent of action codes.

This is not a learned action tokenizer and has no trainable state. Angle IDs
represent raw physical radians in the configured range; no principal-angle
wrapping is performed. Gripper IDs represent normalized opening (0 closed, 1
open). ``format_instructions`` writes these IDs into ordinary instruction
text. It does not register special vocabulary tokens or mutate input examples.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import operator

import numpy as np
import torch
from torch import nn

from robomorpher.model.tokenizer.gripper_tokenizer import GripperTokenizer
from robomorpher.utils.model.masking import binary_bool_mask

_FIXED_STATE_FIELDS = {"angle_low", "angle_high", "normalization", "value_low", "value_high"}


def reject_fixed_state_bounds(state_cfg):
    present = _FIXED_STATE_FIELDS & set(state_cfg)
    if present:
        raise ValueError(
            "Instruction-text state bins use dataset q01/q99; delete "
            + ", ".join(sorted(present)))


def instruction_state_quantiles(*, action_type, num_arms, max_num_joints, stats_path=None, state_stats=None,
                                stats_column="action"):
    """Per-slot q01/q99 for instruction text. Gripper stays on uniform [0, 1] bins."""
    if state_stats is not None:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(state_stats):
            state_stats = OmegaConf.to_container(state_stats, resolve=True)
        from robomorpher.utils.data.process_data.ee_bounds import load_ee_state_quantiles
        if state_stats.get("layout") == "arm_major_joint_angles":
            return _coerce_joint_quantile_spec(state_stats)
        _, _, spec = load_ee_state_quantiles(stats=state_stats)
        return spec
    if not stats_path:
        raise ValueError("Instruction-text state q01/q99 require data.root or pinned state_stats")
    if action_type == "ee" and stats_column == "action":
        from robomorpher.utils.data.process_data.ee_bounds import load_ee_state_quantiles
        _, _, spec = load_ee_state_quantiles(stats_path=stats_path)
        return spec
    import json
    from pathlib import Path
    raw = json.loads(Path(stats_path).read_text())
    action = raw.get(stats_column) if stats_column != "action" else raw.get("action", raw)
    if not isinstance(action, dict) or "q01" not in action:
        raise ValueError(f"Instruction-text joint state stats are missing {stats_column}")
    width = int(num_arms) * (int(max_num_joints) + 1)
    q01 = np.asarray(action.get("q01"), dtype=np.float32)
    q99 = np.asarray(action.get("q99"), dtype=np.float32)
    count = np.asarray(action.get("count"), dtype=np.float64)
    if q01.shape != (width,) or q99.shape != (width,) or count.shape != (width,):
        raise ValueError(f"Joint state q01/q99/count must have shape [{width}]")
    q01 = q01.reshape(int(num_arms), int(max_num_joints) + 1)[:, :int(max_num_joints)]
    q99 = q99.reshape(int(num_arms), int(max_num_joints) + 1)[:, :int(max_num_joints)]
    counts = count.reshape(int(num_arms), int(max_num_joints) + 1)[:, :int(max_num_joints)]
    return _joint_quantile_spec(q01, q99, counts, stats_path=str(Path(stats_path).resolve()),
                                source_column=stats_column)


def bounds_to_state_quantiles(low, high, num_arms, max_num_joints):
    """Record a previously fixed interval as per-slot q01/q99."""
    q01 = np.full((int(num_arms), int(max_num_joints)), float(low), dtype=np.float32)
    q99 = np.full((int(num_arms), int(max_num_joints)), float(high), dtype=np.float32)
    counts = np.ones_like(q01)
    return _joint_quantile_spec(q01, q99, counts, stats_path=None)


def _joint_quantile_spec(q01, q99, counts, *, stats_path, source_column="action"):
    import hashlib
    import json
    spec = {
        "normalization": "state_q01_q99",
        "source_column": source_column,
        "stats_path": stats_path,
        "layout": "arm_major_joint_angles",
        "q01": np.asarray(q01, dtype=np.float32).tolist(),
        "q99": np.asarray(q99, dtype=np.float32).tolist(),
        "counts": np.asarray(counts, dtype=np.int64).tolist(),
        "constant_slot_policy": "center_bin_for_finite_valid_values",
        "gripper_policy": "independent uniform opening bins; no quantile normalization",
    }
    spec["constant_slots"] = (np.asarray(q99) == np.asarray(q01)).tolist()
    semantic = {key: value for key, value in spec.items() if key not in {"stats_path", "fingerprint"}}
    spec["fingerprint"] = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return spec


def _coerce_joint_quantile_spec(stats):
    import hashlib
    import json
    spec = json.loads(json.dumps(stats, allow_nan=False))
    claimed = spec.pop("fingerprint", None)
    semantic = {key: value for key, value in spec.items() if key != "stats_path"}
    actual = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("Pinned joint state q01/q99 fingerprint is invalid")
    spec["fingerprint"] = claimed
    return spec


class JointStateTokenizer(nn.Module):
    """FP32 angle/gripper state binning with explicit heterogeneous-joint masks.

    ``encode`` takes angle [B,A,J] in radians, gripper [B,A] in [0,1], and
    binary joint_valid_mask [B,A,J]. Padding may be nonfinite and is sanitized
    before quantization. Invalid joints and entirely absent arms produce num_bins
    (default 256) as an independent PAD ID. Valid IDs remain 0..num_bins-1. No
    validity is inferred from numerical state values. Angles are never folded into
    a principal interval.
    """

    def __init__(self, num_bins=256, text_format="grouped_numeric", pad_id=None,
                 angle_wrap="none", angle_low=None, angle_high=None,
                 value_low=None, value_high=None, normalization=None,
                 state_stats_path=None, state_stats=None):
        super().__init__()
        num_bins = self._integer(num_bins, "num_bins", minimum=2)
        angle_wrap = str(angle_wrap)
        if angle_wrap != "none":
            raise ValueError("Only raw angle_wrap=none is supported; principal-angle wrapping was removed")
        default_low, default_high = -2.0 * math.pi, 2.0 * math.pi
        angle_low = default_low if angle_low is None else float(angle_low)
        angle_high = default_high if angle_high is None else float(angle_high)
        if not math.isfinite(angle_low) or not math.isfinite(angle_high) or not angle_low < angle_high:
            raise ValueError("angle_low/high must be finite with angle_low < angle_high")
        self.angle_wrap = angle_wrap
        self.angle_low = angle_low
        self.angle_high = angle_high
        self.normalization = "fixed_bounds" if normalization is None else str(normalization)
        if self.normalization not in {"fixed_bounds", "state_q01_q99"}:
            raise ValueError("state normalization must be fixed_bounds or state_q01_q99")
        self.state_stats_path = str(state_stats_path) if state_stats_path else None
        if self.normalization == "state_q01_q99" or state_stats is not None:
            if value_low is not None or value_high is not None:
                raise ValueError("state q01/q99 cannot be combined with value_low/value_high")
            spec = instruction_state_quantiles(
                action_type="ee", num_arms=1, max_num_joints=9,
                stats_path=state_stats_path, state_stats=state_stats)
            self.normalization = "state_q01_q99"
            self._state_normalization = spec
            self.register_buffer("_state_q01", torch.as_tensor(spec["q01"], dtype=torch.float32), persistent=False)
            self.register_buffer("_state_q99", torch.as_tensor(spec["q99"], dtype=torch.float32), persistent=False)
        elif state_stats_path is not None or state_stats is not None:
            raise ValueError("state stats require explicit normalization=state_q01_q99")
        if (value_low is None) != (value_high is None):
            raise ValueError("value_low and value_high must be set together")
        if value_low is None:
            self.slot_low = None
            self.slot_high = None
        else:
            low = torch.as_tensor(value_low, dtype=torch.float32)
            high = torch.as_tensor(value_high, dtype=torch.float32)
            if low.ndim != 1 or tuple(low.shape) != tuple(high.shape) or low.numel() < 1:
                raise ValueError("value_low/value_high must be equal 1-D bounds")
            if not bool(torch.isfinite(low).all() and torch.isfinite(high).all()) or bool((high <= low).any()):
                raise ValueError("value_low/value_high must be finite with high > low")
            self.register_buffer("slot_low", low, persistent=False)
            self.register_buffer("slot_high", high, persistent=False)
        # Reuse the independently tested opening policy and strict bin count
        # validation. Its FP32 guard also protects module-wide precision casts.
        self.gripper_tokenizer = GripperTokenizer(mode="uniform_bins", num_bins=num_bins)
        self.num_bins = self.gripper_tokenizer.num_bins
        if text_format != "grouped_numeric":
            raise ValueError("state text_format must be grouped_numeric")
        if pad_id is None:
            pad_id = self.num_bins
        pad_id = self._integer(pad_id, "pad_id", minimum=0)
        if pad_id != self.num_bins:
            raise ValueError("state pad_id must equal num_bins, outside the valid bin range")
        self.pad_id = pad_id
        self.text_format = text_format

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.normalization == "state_q01_q99":
            # Keep normalization exact across module-wide bf16/half casts.
            # Rebuild from pinned FP32 values, not already-rounded buffers.
            device = self._state_q01.device
            self._state_q01 = torch.tensor(self._state_normalization["q01"], dtype=torch.float32, device=device)
            self._state_q99 = torch.tensor(self._state_normalization["q99"], dtype=torch.float32, device=device)
        return result

    @staticmethod
    def _integer(value, name, minimum=1):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError(f"{name} must be an integer >= {minimum}") from error
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    @property
    def device(self):
        return self.gripper_tokenizer.device

    @property
    def vocab_size(self):
        return self.num_bins + 1

    def to_config(self):
        result = {"num_bins": self.num_bins, "text_format": self.text_format,
                  "pad_id": self.pad_id, "angle_wrap": self.angle_wrap,
                  "angle_low": self.angle_low, "angle_high": self.angle_high}
        if self.slot_low is not None:
            result["value_low"] = self.slot_low.detach().cpu().tolist()
            result["value_high"] = self.slot_high.detach().cpu().tolist()
        if self.normalization == "state_q01_q99":
            result["normalization"] = "state_q01_q99"
            result["state_stats"] = self._state_normalization
        return result

    def metadata(self):
        result = {
            "schema_version": 2,
            "policy": ("ee_xyz_rot6d_uniform_bins" if self.slot_low is not None
                       else "joint_state_uniform_bins"),
            "num_bins": self.num_bins,
            "angle_units": "radians",
            "angle_wrap": self.angle_wrap,
            "angle_bounds": [self.angle_low, self.angle_high],
            "gripper_bounds": [0.0, 1.0],
            "gripper_policy": "normalized_opening_closed0_open1",
            "quantization": "floor_then_clip_inclusive_upper_endpoint",
            "layout": "separate_angle_BAJ_gripper_BA",
            "padding_id": self.pad_id,
            "vocab_size_including_pad": self.vocab_size,
            "conditioning": "instruction_text",
            "text_format_version": 2,
            "text_format": self.text_format,
            "text_padding": str(self.pad_id),
            "text_markers": ["[STATE_ARM]", "[STATE_GRIPPER]", "[ACTION]"],
            "text_arm_order": "arm_major_joint_slots_then_arm_major_grippers",
            "text_scalar_count": "num_arms * (max_num_joints + 1)",
            "text_ids_are_vlm_tokens": False,
            "text_joint_order": "fixed_joint_slots",
            "special_tokens_added": False,
        }

        if self.normalization == "state_q01_q99":
            pose = int(self._state_q01.shape[-1]) == 9
            result["policy"] = ("ee_xyz_rot6d_state_q01_q99_bins" if pose else "joint_state_q01_q99_bins")
            result["normalization"] = self._state_normalization
            result["quantization"] = "clip_q01_q99_to_0_1_then_floor_bins_inclusive_upper"
        return result

    def to_dict(self):
        return self.metadata()

    @torch.no_grad()
    def encode(self, angle, gripper, joint_valid_mask):
        angle = torch.as_tensor(angle, device=self.device)
        gripper = torch.as_tensor(gripper, device=self.device)
        mask = binary_bool_mask(joint_valid_mask, device=self.device, name="joint_valid_mask")
        if angle.ndim != 3 or angle.shape[1] < 1 or angle.shape[2] < 1:
            raise ValueError("Observed angle must have [B,A,J] shape with positive A/J")
        if mask.shape != angle.shape or gripper.shape != angle.shape[:2]:
            raise ValueError("Expected joint_valid_mask [B,A,J] and gripper [B,A] matching angle")
        if angle.is_complex() or angle.is_quantized:
            raise ValueError("Observed angle must contain real numeric radians")
        if not bool(torch.isfinite(angle[mask]).all()):
            raise ValueError("Valid observed angles must be finite")
        safe_angle = angle.float().masked_fill(~mask, 0)
        if not bool(torch.isfinite(safe_angle).all()):
            raise ValueError("Valid observed angles must remain finite in FP32")
        physical = safe_angle
        # Joint state uses one radian interval. EE state uses xyz in [-2, 2]
        # and rot6d in [-1, 1], the same quantities StarVLA puts in the state
        # vector rather than joint angles.
        if self.normalization == "state_q01_q99":
            if tuple(physical.shape[1:]) != tuple(self._state_q01.shape):
                raise ValueError("state q01/q99 shape does not match the observed state")
            if physical.shape[-1] == 9 and bool(((physical[..., 3:] < -1.0 - 1e-6) |
                     (physical[..., 3:] > 1.0 + 1e-6))[mask[..., 3:]].any()):
                raise ValueError("Observed EE rot6d is outside physical [-1,1]")
            count = torch.as_tensor(self._state_normalization["counts"], device=physical.device)
            if bool((mask & (count[None] == 0)).any()):
                raise ValueError("Observed EE arm has no valid normalization observations")
            low = self._state_q01.to(device=physical.device, dtype=physical.dtype)[None]
            high = self._state_q99.to(device=physical.device, dtype=physical.dtype)[None]
            constant = high == low
            width = torch.where(constant, torch.ones_like(high), high - low)
            # Only the derived token coordinates clip quantile tails. Raw state
            # remains untouched; gripper uses its independent opening bins.
            unit = ((physical - low) / width).clamp(0.0, 1.0)
            unit = torch.where(constant, torch.full_like(unit, 0.5), unit)
            angle_ids = (unit * self.num_bins).floor().long()
        elif self.slot_low is not None:
            if physical.shape[-1] != int(self.slot_low.numel()):
                raise ValueError("EE state width does not match value_low/value_high")
            low = self.slot_low.to(device=physical.device, dtype=physical.dtype).view(1, 1, -1)
            high = self.slot_high.to(device=physical.device, dtype=physical.dtype).view(1, 1, -1)
            if bool(((physical < low - 1e-3) | (physical > high + 1e-3))[mask].any()):
                raise ValueError("Observed EE state is outside xyz[-2, 2] or rot6d[-1, 1]")
            angle_ids = (((physical - low) / (high - low)) * self.num_bins).floor().long()
        else:
            if bool(((physical < self.angle_low - 1e-3) |
                     (physical > self.angle_high + 1e-3))[mask].any()):
                raise ValueError(
                    f"Observed angles outside configured state range "
                    f"[{self.angle_low}, {self.angle_high}]")
            angle_ids = (((physical - self.angle_low) /
                          (self.angle_high - self.angle_low)) * self.num_bins).floor().long()
        angle_ids = angle_ids.clamp(0, self.num_bins - 1).masked_fill(~mask, self.pad_id)
        arm_mask = mask.any(-1)
        # GripperTokenizer retains -1 for action-code padding; observed state
        # deliberately has a distinct, nonnegative PAD outside its valid bins.
        gripper_ids = self.gripper_tokenizer.encode(gripper, arm_mask).masked_fill(~arm_mask, self.pad_id)
        return {"angle_ids": angle_ids, "gripper_ids": gripper_ids,
                "joint_valid_mask": mask}


    @torch.no_grad()
    def encode_examples(self, examples, num_arms, max_num_joints):
        """Batch only observed state fields; never access future action targets.

        NumPy values are stacked without first casting masks to bool, so invalid
        mask values cannot silently unmask padding. Quantization itself is FP32;
        raw gripper bounds are checked before casting to avoid hiding invalid
        float64 values through rounding. The caller's examples are unchanged.
        """
        num_arms = self._integer(num_arms, "num_arms")
        max_num_joints = self._integer(max_num_joints, "max_num_joints")
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
            raise ValueError("examples must be a sequence of observed-state mappings")
        if any(not isinstance(example, Mapping) for example in examples):
            raise ValueError("Every example must be an observed-state mapping")
        shape = (len(examples), num_arms, max_num_joints)
        if not examples:
            return self.encode(torch.empty(shape, device=self.device),
                               torch.empty(shape[:2], device=self.device),
                               torch.empty(shape, dtype=torch.bool, device=self.device))
        try:
            angle = np.stack([example["state_angle"] for example in examples])
            gripper = np.stack([example["state_gripper"] for example in examples])
            mask = np.stack([example["joint_valid_mask"] for example in examples])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Examples require stackable state_angle, state_gripper and joint_valid_mask") from error
        if angle.shape != shape or mask.shape != shape or gripper.shape != shape[:2]:
            raise ValueError(f"Observed state shape must match angle/mask {shape} and gripper {shape[:2]}")
        return self.encode(angle, gripper, mask)

    @torch.no_grad()
    def format_instructions(self, instructions, encoded):
        """Return ordinary text prompts with categorical observed state appended.

        Example: ``Pick [STATE_ARM] 128 256 [STATE_GRIPPER] 255 [ACTION]``.
        Joint IDs are flattened arm-major, followed by a separate arm-major
        gripper group. Fixed slots retain missing joints/arms as num_bins PAD.
        These are ordinary decimal strings and ordinary text markers: scalar
        count is not the Qwen BPE-token count, and no vocabulary is registered.
        """
        if isinstance(instructions, (str, bytes)) or not isinstance(instructions, Sequence):
            raise ValueError("instructions must be a sequence of strings")
        if any(not isinstance(instruction, str) for instruction in instructions):
            raise ValueError("Every instruction must be a string")
        required = ("angle_ids", "gripper_ids", "joint_valid_mask")
        if not isinstance(encoded, Mapping) or any(key not in encoded for key in required):
            raise ValueError("encoded requires angle_ids, gripper_ids and joint_valid_mask")
        mask = binary_bool_mask(encoded["joint_valid_mask"], device=self.device, name="joint_valid_mask")
        if mask.ndim != 3 or mask.shape[1] < 1 or mask.shape[2] < 1:
            raise ValueError("Encoded joint_valid_mask must have [B,A,J] shape")
        if len(instructions) != mask.shape[0]:
            raise ValueError("Instruction count must match the encoded state batch")

        def validate_ids(values, valid, name):
            values = torch.as_tensor(values, device=self.device)
            if values.shape != valid.shape:
                raise ValueError(f"{name} shape must match its structural mask")
            integer_types = (torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                             torch.int8, torch.int16, torch.int32, torch.int64)
            if values.dtype not in integer_types:
                raise ValueError(f"{name} must have integer dtype")
            values = values.long()
            if bool(((values[valid] < 0) | (values[valid] >= self.num_bins)).any()):
                raise ValueError(f"Valid {name} outside state vocabulary")
            if bool((values[~valid] != self.pad_id).any()):
                raise ValueError(f"Padded {name} must equal {self.pad_id}")
            return values.cpu().tolist()

        angles = validate_ids(encoded["angle_ids"], mask, "angle_ids")
        grippers = validate_ids(encoded["gripper_ids"], mask.any(-1), "gripper_ids")
        results = []
        for instruction, sample_angles, sample_grippers in zip(instructions, angles, grippers):
            joints_text = " ".join(str(token) for joint_ids in sample_angles for token in joint_ids)
            gripper_text = " ".join(str(token) for token in sample_grippers)
            prefix = instruction + " " if instruction else ""
            results.append(prefix + "[STATE_ARM] " + joints_text +
                           " [STATE_GRIPPER] " + gripper_text + " [ACTION]")
        return results
