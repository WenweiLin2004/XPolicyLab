"""Validate stored joint validity without inferring it from feature values."""
from __future__ import annotations
import numpy as np


def binary_joint_mask(values, *, where="joint_valid_mask"):
    """Return a boolean mask only for finite, exact 0/1 stored values.

    Boolean and numeric 0/1 masks are equivalent. Empty rows and holes retain
    their existing meaning here; model-specific structural rules belong to the
    model. Never infer validity from q=0 or use truthiness on corrupt floats.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{where}: joint_valid_mask must be boolean or numeric 0/1")
    valid = np.isfinite(array) & ((array == 0) | (array == 1))
    if not bool(np.all(valid)):
        raise ValueError(f"{where}: joint_valid_mask must contain only finite exact 0/1 values")
    return array.astype(np.bool_, copy=False)
