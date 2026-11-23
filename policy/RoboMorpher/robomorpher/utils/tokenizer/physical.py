"""Physical evaluation helpers; dataset pose9+angle layout is independent of decoder heads."""
import numpy as np
import torch
from robomorpher.utils.geometry.diff_fk import rotation_6d_to_matrix


def denorm(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
           vmin: torch.Tensor, vmax: torch.Tensor, method: str) -> torch.Tensor:
    if method == "zscore":
        return t * std + mean
    # minmax / quantile (q99...) range mapping
    return (t + 1.0) / 2.0 * (vmax - vmin + 1e-6) + vmin


def denorm_features(values, stats, prefix, method, feature_slice=slice(None)):
    """Select channel statistics for shared [D] and historical [J,D] payloads."""
    mean = stats[f"{prefix}_mean"][..., feature_slice]
    std = stats[f"{prefix}_std"][..., feature_slice]
    if method == "zscore":
        low = high = None
    elif method in ("minmax", "q99"):
        low_key, high_key = ("min", "max") if method == "minmax" else ("q01", "q99")
        low = stats[f"{prefix}_{low_key}"][..., feature_slice]
        high = stats[f"{prefix}_{high_key}"][..., feature_slice]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return denorm(values, mean, std, low, high, method)


def rot_geodesic_deg(pred_6d: torch.Tensor, gt_6d: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error in degrees between two rot6d batches (...)."""
    r_pred = rotation_6d_to_matrix(pred_6d)
    r_gt = rotation_6d_to_matrix(gt_6d)
    cos = ((r_pred.transpose(-1, -2) @ r_gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.rad2deg(torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))

def physical_reconstruction(input_data, output_data, ee_pred, joint_pred, mask, stats, method):
    """Return physical input, target, optional direct pose, predicted/target q.

    The stored target is always pose9+angle, even for an angle-only decoder.
    Padding is cleared before and after inverse normalization.
    """
    valid = mask.to(device=input_data.device, dtype=torch.bool)
    if output_data.shape[:-1] != valid.shape or output_data.shape[-1] != 10:
        raise ValueError("Physical evaluation requires dataset targets [B,J,10] (pose9+angle)")
    if joint_pred.shape != (*valid.shape, 1):
        raise ValueError("Physical evaluation requires one predicted angle per joint")
    if ee_pred.shape[:-1] != valid.shape or ee_pred.shape[-1] not in (0, 9):
        raise ValueError("Decoder EE must have zero or nine features per joint")
    def inverse(value, prefix, feature_slice=slice(None)):
        clean = value.masked_fill(~valid[..., None], 0.0)
        return denorm_features(clean, stats, prefix, method, feature_slice).masked_fill(~valid[..., None], 0.0)
    xp = inverse(input_data, "x")
    yp = inverse(output_data, "y")
    ep = inverse(ee_pred, "y", slice(0, 9)) if ee_pred.shape[-1] else None
    qp = inverse(joint_pred, "y", slice(9, 10))
    return xp, yp, ep, qp, yp[..., 9:10]


def error_summary(values):
    """One statistic over actual selected errors, never a mean of slot percentiles."""
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not a.size:
        return {"n": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite reconstruction errors in valid samples")
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def valid_error_summary(values, mask):
    values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("Error values and validity mask must have matching shapes")
    return error_summary(values[mask])
