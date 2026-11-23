"""Tensor mask validation shared by tokenizer and discrete VLA boundaries."""
from __future__ import annotations

import torch


def binary_bool_mask(values, *, device=None, name="mask"):
    """Accept booleans or finite exact 0/1 values without changing mask shape.

    Boolean inputs take the fast path, with no value check or device-to-host
    synchronization. Numeric masks are checked before conversion: bool(NaN),
    bool(-1) and bool(0.5) are all True and must never mark padding as valid.
    Shape checks and broadcasting remain the responsibility of the caller.
    """
    try:
        mask = torch.as_tensor(values, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be boolean or numeric finite exact 0/1 values") from error
    if mask.dtype == torch.bool:
        return mask
    integer_types = (torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                     torch.int8, torch.int16, torch.int32, torch.int64)
    if not mask.is_floating_point() and mask.dtype not in integer_types:
        raise ValueError(f"{name} must be boolean or numeric finite exact 0/1 values")
    valid = torch.isfinite(mask) & ((mask == 0) | (mask == 1))
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain only finite exact 0/1 values")
    return mask.to(dtype=torch.bool)
