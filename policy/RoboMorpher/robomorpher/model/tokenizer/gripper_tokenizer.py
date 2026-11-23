"""Parameter-free gripper action tokenization, separate from arm tokenizers.

Inputs and decoded values use physical normalized opening: 0 is closed and 1
is open. ``uniform_bins`` preserves the historical floor/clip and bin-center
policy exactly. ``binary`` is a separate endpoint policy: opening >= threshold
maps to token 1, and tokens 0/1 decode to exactly 0/1 (not 0.25/0.75).

Masks must match the full input shape. Invalid input openings may contain any
value, including NaN/Inf; encode sanitizes them before arithmetic and emits -1.
Decode requires -1 in masked positions and returns zero there. Without a mask,
every position is valid, so -1 is not a legal input ID. No missing mask is
inferred from values. Metadata is JSON serializable for outer codec identity
hashes; state dictionaries intentionally contain no learned state.
"""
from __future__ import annotations

import math
import operator

import torch
from torch import nn

from robomorpher.utils.model.masking import binary_bool_mask


class GripperTokenizer(nn.Module):
    """Independent gripper policy with FP32 arithmetic and no parameters.

    Args:
        mode: ``uniform_bins`` (default) or ``binary``.
        num_bins: Defaults to 256 for uniform bins, 2 for binary. Binary only
            accepts 2; choosing uniform bins with 2 retains midpoint decoding.
        threshold: Binary-only threshold, finite and strictly inside (0, 1).
            Defaults to 0.5. An explicit threshold with uniform bins is an error.

    Device transfers are supported. Precision casts of this module are rejected:
    the surrounding VLM may use autocast, but discrete boundaries stay FP32.
    ``to_config()`` returns constructor arguments; ``metadata()`` / ``to_dict()``
    returns the complete policy contract suitable for a codec fingerprint.
    """

    def __init__(self, mode="uniform_bins", num_bins=None, threshold=None):
        super().__init__()
        if mode not in ("uniform_bins", "binary"):
            raise ValueError("Gripper mode must be 'uniform_bins' or 'binary'")
        if num_bins is None:
            num_bins = 2 if mode == "binary" else 256
        if isinstance(num_bins, bool):
            raise ValueError("Gripper num_bins must be an integer >= 2")
        try:
            num_bins = operator.index(num_bins)
        except TypeError as error:
            raise ValueError("Gripper num_bins must be an integer >= 2") from error
        if num_bins < 2 or num_bins > torch.iinfo(torch.int64).max:
            raise ValueError("Gripper num_bins must be an integer >= 2 representable as int64")
        if mode == "binary":
            if num_bins != 2:
                raise ValueError("Binary gripper requires num_bins=2")
            if threshold is None:
                threshold = 0.5
            if isinstance(threshold, bool):
                raise ValueError("Binary threshold must be finite and strictly between 0 and 1")
            try:
                threshold = float(threshold)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Binary threshold must be finite and strictly between 0 and 1") from error
            if not math.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError("Binary threshold must be finite and strictly between 0 and 1")
        elif threshold is not None:
            raise ValueError("threshold is only supported by binary gripper mode")
        self.mode = mode
        self.num_bins = int(num_bins)
        self.threshold = threshold
        self.register_buffer("_device_anchor", torch.zeros(0), persistent=False)

    @property
    def device(self):
        return self._device_anchor.device

    @property
    def num_tokens(self):
        return self.num_bins

    @property
    def vocab_size(self):
        return self.num_bins

    def _apply(self, fn, recurse=True):
        if fn(torch.zeros(1, device=self.device)).dtype != torch.float32:
            raise ValueError("Keep GripperTokenizer FP32; use autocast around the VLM/head")
        return super()._apply(fn, recurse)

    def to_config(self):
        config = {"mode": self.mode, "num_bins": self.num_bins}
        if self.mode == "binary":
            config["threshold"] = self.threshold
        return config

    def metadata(self):
        return {
            "schema_version": 1,
            "policy": "normalized_opening_closed0_open1",
            "mode": self.mode,
            "count": self.num_bins,
            "threshold": self.threshold,
            "representative": "binary_endpoints" if self.mode == "binary" else "bin_center",
            "quantization": "greater_equal_threshold" if self.mode == "binary" else "floor_then_clip_inclusive_upper_endpoint",
            "bounds": [0.0, 1.0],
            "padding_id": -1,
            "padding_decoded_value": 0.0,
        }

    def to_dict(self):
        return self.metadata()

    def _valid(self, values, valid_mask):
        if valid_mask is None:
            return torch.ones_like(values, dtype=torch.bool)
        valid = binary_bool_mask(valid_mask, device=self.device, name="gripper valid_mask")
        if valid.shape != values.shape:
            raise ValueError("Gripper valid_mask must match the exact input shape")
        return valid

    @torch.no_grad()
    def encode(self, opening, valid_mask=None):
        """Map valid normalized openings to local IDs; padded outputs are -1."""
        opening = torch.as_tensor(opening, device=self.device)
        if opening.is_complex() or opening.is_quantized:
            raise ValueError("Gripper opening must be real numeric values in [0,1]")
        valid = self._valid(opening, valid_mask)
        selected = opening[valid]
        # Validate before the FP32 cast: out-of-range doubles must not silently
        # round into range. Sanitization precedes all quantization arithmetic.
        if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
            raise ValueError("Valid gripper opening must be finite and in [0,1]")
        safe = opening.masked_fill(~valid, 0).float()
        if self.mode == "binary":
            ids = (safe >= self.threshold).long()
        else:
            ids = (safe * self.num_bins).floor().long().clamp(0, self.num_bins - 1)
        return ids.masked_fill(~valid, -1)

    @torch.no_grad()
    def decode(self, ids, valid_mask=None):
        """Decode local integer IDs to FP32 opening; padded outputs are zero."""
        ids = torch.as_tensor(ids, device=self.device)
        integer_types = (torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                         torch.int8, torch.int16, torch.int32, torch.int64)
        if ids.dtype not in integer_types:
            raise ValueError("Gripper IDs must have integer dtype (not bool or float)")
        valid = self._valid(ids, valid_mask)
        unsigned = ids.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
        if unsigned and bool((~valid).any()):
            raise ValueError("Padding gripper IDs must equal -1; use a signed integer dtype")
        ids = ids.long()
        if bool(((ids[valid] < 0) | (ids[valid] >= self.num_bins)).any()):
            raise ValueError("Valid gripper ID is outside the mode vocabulary")
        if bool((ids[~valid] != -1).any()):
            raise ValueError("Padding gripper IDs must equal -1")
        safe = ids.masked_fill(~valid, 0).float()
        decoded = safe if self.mode == "binary" else (safe + 0.5) / self.num_bins
        return decoded.masked_fill(~valid, 0)
