"""Evaluation-only FK diagnostics; never part of the training objective."""

import torch
from robomorpher.utils.geometry.diff_fk import forward_kinematics_6d


def _masked_squared_error(pred, target, joint_valid_mask):
    """Clear padding before arithmetic, retaining at least FP32 loss precision."""
    valid = joint_valid_mask[..., None].to(device=pred.device, dtype=torch.bool)
    dtype = torch.promote_types(pred.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    # Multiplying an already-computed NaN/Inf error by zero does not mask it,
    # and also leaves nonfinite gradients. Mask both operands before subtraction.
    clean_pred = pred.to(dtype=dtype).masked_fill(~valid, 0.0)
    clean_target = target.to(dtype=dtype).masked_fill(~valid, 0.0)
    return (clean_pred - clean_target).square(), valid


def masked_joint_mse(pred, target, joint_valid_mask):
    """MSE over valid joint elements only (T=1 shapes: pred (B, J, D), mask (B, J))."""
    squared_error, valid = _masked_squared_error(pred, target, joint_valid_mask)
    valid_elements = valid.sum(dtype=squared_error.dtype) * pred.shape[-1]
    return squared_error.sum() / valid_elements.clamp_min(1.0)


def per_joint_mse(pred, target, joint_valid_mask):
    """Per-joint masked MSE -> (num_joints,)."""
    squared_error, valid = _masked_squared_error(pred, target, joint_valid_mask)
    denom = (valid.sum(dim=(0, 2), dtype=squared_error.dtype) * pred.shape[-1]).clamp_min(1.0)
    return squared_error.sum(dim=(0, 2)) / denom


def denorm_to_physical(input_data, output_data, joint_pred, joint_valid_mask, norm_stats, ee_dim, joint_dim):
    """De-normalize model I/O to physical units (masked). T=1 shapes."""
    # Decoder EE suppression does not remove the dataset pose9 target/statistics.
    ee_dim = 9 if ee_dim == 0 else ee_dim
    joint_mask = joint_valid_mask[..., None].to(device=input_data.device, dtype=torch.bool)
    # Sanitizing before de-normalization also makes padding a finite identity
    # transform in FK; a NaN middle slot must not poison later valid slots.
    input_data = input_data.masked_fill(~joint_mask, 0.0)
    output_data = output_data.masked_fill(~joint_mask, 0.0)
    joint_pred = joint_pred.masked_fill(~joint_mask, 0.0)

    method = str(norm_stats["method"])
    if method == "minmax":
        x_lo, x_hi = norm_stats["x_min"], norm_stats["x_max"]       # full range
        y_lo, y_hi = norm_stats["y_min"], norm_stats["y_max"]
    elif method != "zscore":
        x_lo, x_hi = norm_stats["x_q01"], norm_stats["x_q99"]       # quantile range
        y_lo, y_hi = norm_stats["y_q01"], norm_stats["y_q99"]

    if method != "zscore":  # minmax / quantile: [-1, 1] -> [lo, hi]
        input_phys = (
            (input_data + 1.0) / 2.0 * (x_hi - x_lo + 1e-6) + x_lo
        ).masked_fill(~joint_mask, 0.0)
        output_range = y_hi - y_lo + 1e-6
        ee_gt_phys = (
            (output_data[..., :ee_dim] + 1.0) / 2.0 * output_range[..., :ee_dim]
            + y_lo[..., :ee_dim]
        ).masked_fill(~joint_mask, 0.0)
        joint_pred_phys = (
            (joint_pred + 1.0) / 2.0 * output_range[..., ee_dim : ee_dim + joint_dim]
            + y_lo[..., ee_dim : ee_dim + joint_dim]
        ).masked_fill(~joint_mask, 0.0)
    else:  # zscore
        input_phys = (input_data * norm_stats["x_std"] + norm_stats["x_mean"]).masked_fill(~joint_mask, 0.0)
        ee_gt_phys = (
            output_data[..., :ee_dim] * norm_stats["y_std"][..., :ee_dim]
            + norm_stats["y_mean"][..., :ee_dim]
        ).masked_fill(~joint_mask, 0.0)
        joint_pred_phys = (
            joint_pred * norm_stats["y_std"][..., ee_dim : ee_dim + joint_dim]
            + norm_stats["y_mean"][..., ee_dim : ee_dim + joint_dim]
        ).masked_fill(~joint_mask, 0.0)
    return input_phys, ee_gt_phys, joint_pred_phys


def fk_pose_from_joints(input_phys, joint_pred_phys, joint_valid_mask):
    """Differentiable FK on (physical) joint angles -> per-joint poses (B, J, 9)."""
    urdf_features = input_phys[..., :-1]  # (B, J, 6|rpy or 9|rot6d), angle channel dropped
    if urdf_features.shape[-1] == 9:
        # masked padding joints denormalize to an all-zero rot6d, which is not a
        # valid rotation -> replace with identity so they compose as no-ops in
        # the FK chain (same as rpy=0 in the 6D case).
        # NOTE: `urdf[~mask, 3:9] = v` is INVALID (the slice is checked against
        # the mask dims); flatten to a 1D mask instead.
        urdf_features = urdf_features.clone()
        flat = urdf_features.reshape(-1, urdf_features.shape[-1])
        flat[(~joint_valid_mask).reshape(-1), 3:9] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            device=urdf_features.device, dtype=urdf_features.dtype,
        )
    return forward_kinematics_6d(urdf_features, joint_pred_phys.squeeze(-1))


def normalize_ee_phys(ee_phys, norm_stats, ee_dim):
    """Physical ee pose (B, J, ee_dim) -> normalized space (inverse of denorm)."""
    method = str(norm_stats["method"])
    if method == "minmax":
        y_lo, y_hi = norm_stats["y_min"], norm_stats["y_max"]       # full range
    elif method != "zscore":
        y_lo, y_hi = norm_stats["y_q01"], norm_stats["y_q99"]       # quantile range
    if method != "zscore":  # minmax / quantile: [lo, hi] -> [-1, 1]
        normalized = (
            (ee_phys - y_lo[..., :ee_dim])
            / (y_hi[..., :ee_dim] - y_lo[..., :ee_dim] + 1e-6)
            * 2.0
            - 1.0
        )
        # Dataset minmax extrapolates; only quantile normalization clips.
        # Clipping minmax here would silently hide out-of-range FK errors.
        return normalized if method == "minmax" else normalized.clamp(-1.0, 1.0)
    return (ee_phys - norm_stats["y_mean"][..., :ee_dim]) / norm_stats["y_std"][..., :ee_dim]


@torch.no_grad()
def compute_fk_ee_loss(
    input_data, output_data, joint_pred, joint_valid_mask, norm_stats, ee_dim, joint_dim,
):
    """Compare FK(known URDF, predicted joints) with target EE during eval.

    Returns normalized MSE and per-joint MSE, with no autograd graph even if
    called from a context where gradients are enabled. No training weight or
    auxiliary decoder is involved.
    """
    ee_dim = 9 if ee_dim == 0 else ee_dim
    if ee_dim != 9 or joint_dim != 1:
        raise ValueError("FK evaluation requires pose9 and one angle per joint")
    input_phys, _, joint_pred_phys = denorm_to_physical(
        input_data, output_data, joint_pred, joint_valid_mask, norm_stats, ee_dim, joint_dim
    )
    fk_pose = fk_pose_from_joints(input_phys, joint_pred_phys, joint_valid_mask)
    fk_ee_norm = normalize_ee_phys(fk_pose, norm_stats, ee_dim)
    ee_target = output_data[..., :ee_dim]
    return (masked_joint_mse(fk_ee_norm, ee_target, joint_valid_mask),
            per_joint_mse(fk_ee_norm, ee_target, joint_valid_mask))
