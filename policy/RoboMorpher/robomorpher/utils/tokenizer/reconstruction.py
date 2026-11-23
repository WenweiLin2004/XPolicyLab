"""Direct reconstruction loss configuration and physical rotation supervision.

No FK, model, optimizer, or quantizer updates happen in this module.
Rotation convention and acos clamp match X-Tokenizer / RDT2: first two
columns, Gram-Schmidt, acos(clamp((trace(Rp.T @ Rt)-1)/2))/pi.
"""
import math
import torch
from torch.nn import functional as F
from robomorpher.model.tokenizer.utils.config import normalize_model_config


# Kept as an alias for existing configurations and importers.
SPLIT_EE_LOSS = "xyz_mse_rot_geodesic"
SPLIT_EE_LOSS_TYPES = frozenset({SPLIT_EE_LOSS, "xyz_rot_geodesic"})


def is_split_ee_loss(mode):
    """Both the name and the regression-neutral name use split losses."""
    return str(mode).strip().lower() in SPLIT_EE_LOSS_TYPES


def normalize_regression_loss_type(loss_type):
    """L2 means coordinate-wise MSE, not Euclidean distance or half-MSE."""
    name = str(loss_type).strip().lower()
    aliases = {"l1": "l1", "mae": "l1", "l2": "l2", "mse": "l2"}
    if name not in aliases:
        raise ValueError(f"Unsupported regression loss type: {loss_type!r}; expected l1/l2 (mae/mse)")
    return aliases[name]


def resolve_reconstruction_loss_config(model_config):
    """Old configs keep EE9/joint MSE; split losses choose XYZ and joint independently."""
    vm = normalize_model_config(model_config)
    losses = model_config.get("losses", {})
    ee = losses.get("ee", {})
    joint = losses.get("joint", {})
    mode = str(ee.get("type", "mse")).strip().lower()
    if mode != "mse" and not is_split_ee_loss(mode):
        raise ValueError(f"Unsupported losses.ee.type: {mode!r}")
    xyz_loss_type = normalize_regression_loss_type(ee.get("xyz_loss_type", "l2"))
    joint_loss_type = normalize_regression_loss_type(joint.get("type", "l2"))
    result = {
        "ee_type": mode,
        "xyz_loss_type": xyz_loss_type,
        "joint_loss_type": joint_loss_type,
        "ee_weight": float(vm["ee_loss_weight"]),
        "joint_weight": float(vm["joint_loss_weight"]),
        "xyz_weight": float(ee.get("xyz_weight", 1.0)),
        "rot_weight": float(ee.get("rot_weight", 1.0)),
    }
    for key, value in result.items():
        if key.endswith("weight") and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    if is_split_ee_loss(mode):
        if vm.get("output_with_ee", True) and vm["decoder_ee_dim"] != 9:
            raise ValueError("Split EE loss requires xyz + rot6D (ee_dim=9)")
        if result["ee_weight"] != 1.0:
            raise ValueError("Split EE loss uses xyz_weight / rot_weight; omit ee.weight or set it to 1")
    else:
        if xyz_loss_type != "l2":
            raise ValueError("xyz_loss_type=l1 requires losses.ee.type=xyz_rot_geodesic; EE9 MSE cannot select XYZ L1")
        if "xyz_weight" in ee or "rot_weight" in ee:
            raise ValueError("xyz_weight / rot_weight require a split losses.ee.type (xyz_rot_geodesic)")
    return result


def masked_regression_loss(pred, target, joint_valid_mask, loss_type="l2"):
    """Return scalar and per-slot normalized coordinate MAE/MSE over valid joints.

    Mask inputs before arithmetic so padded NaN/Inf contributes neither a value
    nor a gradient. Counts include features: XYZ is averaged over 3 coordinates,
    not the vector norm. Empty batches/slots return differentiable zero. Compute
    in at least FP32 under AMP, retaining FP64 for numerical checks when supplied.
    """
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape[-1] == 0:
        raise ValueError("Expected matching prediction/target tensors with shape [B, J, D], D > 0")
    kind = normalize_regression_loss_type(loss_type)
    valid = joint_valid_mask.to(device=pred.device, dtype=torch.bool)
    if valid.ndim == 1 and valid.shape[0] == pred.shape[1]:
        valid = valid.unsqueeze(0).expand(pred.shape[0], -1)
    if valid.shape != pred.shape[:-1]:
        raise ValueError("joint_valid_mask must have shape [B, J] or [J]")
    dtype = torch.float64 if pred.dtype == torch.float64 or target.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=pred.device.type, enabled=False):
        clean_pred = pred.to(dtype=dtype).masked_fill(~valid[..., None], 0.0)
        clean_target = target.to(device=pred.device, dtype=dtype).masked_fill(~valid[..., None], 0.0)
        difference = clean_pred - clean_target
        error = difference.abs() if kind == "l1" else difference.square()
        sums = error.sum(dim=(0, 2))
        counts = valid.sum(dim=0, dtype=dtype) * pred.shape[-1]
        return sums.sum() / counts.sum().clamp_min(1.0), sums / counts.clamp_min(1.0)


def output_norm_stats(dataset, device):
    """Use the dataset's frozen output statistics, including on checkpoint resume."""
    method = str(dataset.normalization_method)
    keys = (["y_mean", "y_std"] if method == "zscore" else
            ["y_min", "y_max"] if method == "minmax" else ["y_q01", "y_q99"])
    return {"method": method, **{k: getattr(dataset, k).to(device) for k in keys}}


def _physical_rot6d(ee, valid, norm_stats):
    if norm_stats is None:
        raise ValueError("Physical rotation loss requires frozen output normalization statistics")
    values = ee[..., 3:9].float().masked_fill(~valid[..., None], 0.0)
    def stat(key, padding_value):
        raw = torch.as_tensor(norm_stats[key], device=values.device, dtype=values.dtype)[..., 3:9]
        return torch.broadcast_to(raw, values.shape).masked_fill(~valid[..., None], padding_value)
    method = str(norm_stats["method"])
    if method == "zscore":
        physical = values * stat("y_std", 1.0) + stat("y_mean", 0.0)
    else:
        lo_key, hi_key = (("y_min", "y_max") if method == "minmax" else ("y_q01", "y_q99"))
        lo, hi = stat(lo_key, 0.0), stat(hi_key, 0.0)
        physical = (values + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
    # A padding zero-vector is not a rotation. Replace it BEFORE Gram-Schmidt.
    identity = values.new_tensor([1., 0., 0., 0., 1., 0.])
    return torch.where(valid[..., None], physical, identity)


def _rot6d_to_matrix(rot):
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def masked_rotation_geodesic(ee_pred, ee_target, joint_valid_mask, norm_stats):
    """Return mean theta/pi and per-slot means, each over valid joints only.

    Full FP32 (including matrix multiply) under AMP. The 1e-7 acos clamp is
    the reference implementation's numerical guard, with a small nonzero
    floor near identity. It does not remove the raw-6D degeneracy of zero or
    collinear vectors. No additional rotation anchor loss is introduced.
    """
    if ee_pred.shape != ee_target.shape or ee_pred.shape[-1] != 9:
        raise ValueError("Expected matching EE tensors with shape [B, J, 9]")
    valid = joint_valid_mask.to(device=ee_pred.device, dtype=torch.bool)
    if valid.shape != ee_pred.shape[:-1]:
        raise ValueError("joint_valid_mask must have shape [B, J]")
    with torch.autocast(device_type=ee_pred.device.type, enabled=False):
        pred = _rot6d_to_matrix(_physical_rot6d(ee_pred, valid, norm_stats))
        target = _rot6d_to_matrix(_physical_rot6d(ee_target, valid, norm_stats))
        relative = pred.transpose(-1, -2) @ target
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) / torch.pi
        theta = theta.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=0, dtype=theta.dtype)
        return (theta.sum() / counts.sum().clamp_min(1.0),
                theta.sum(dim=0) / counts.clamp_min(1.0))
