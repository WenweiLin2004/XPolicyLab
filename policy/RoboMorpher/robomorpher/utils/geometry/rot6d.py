"""URDF input representation conversion: xyz+rpy+angle (7D) -> xyz+rot6d+angle (10D).

Rotation convention matches diff_fk.euler_xyz_to_matrix (extrinsic xyz Euler,
scipy 'xyz'); rot6d = first two columns of the rotation matrix, same as
diff_fk.matrix_to_rotation_6d.

Padding joints: rpy=0 converts to an IDENTITY rot6d [1,0,0,0,1,0] (non-zero),
so callers that need padding to stay zero must pass joint_valid_mask — invalid
joints are zeroed AFTER conversion.
"""

import numpy as np
import torch


def urdf_rpy_to_rot6d_np(features: np.ndarray, joint_valid_mask: np.ndarray | None = None) -> np.ndarray:
    """(..., 7) [xyz(3), rpy(3), angle(1)] -> (..., 10) [xyz(3), rot6d(6), angle(1)]."""
    assert features.shape[-1] == 7, f"expected 7D [xyz, rpy, angle], got {features.shape[-1]}D"
    roll, pitch, yaw = features[..., 3], features[..., 4], features[..., 5]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    col0 = np.stack([cy * cp, sy * cp, -sp], axis=-1)
    col1 = np.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr], axis=-1)
    out = np.concatenate([features[..., :3], col0, col1, features[..., 6:7]], axis=-1)
    if joint_valid_mask is not None:
        out = out * joint_valid_mask[..., None].astype(out.dtype)
    return out


def urdf_rpy_to_rot6d(features: torch.Tensor, joint_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """(..., 7) [xyz(3), rpy(3), angle(1)] -> (..., 10) [xyz(3), rot6d(6), angle(1)]."""
    assert features.shape[-1] == 7, f"expected 7D [xyz, rpy, angle], got {features.shape[-1]}D"
    roll, pitch, yaw = features[..., 3], features[..., 4], features[..., 5]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    col0 = torch.stack([cy * cp, sy * cp, -sp], dim=-1)
    col1 = torch.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr], dim=-1)
    out = torch.cat([features[..., :3], col0, col1, features[..., 6:7]], dim=-1)
    if joint_valid_mask is not None:
        out = out * joint_valid_mask[..., None].to(dtype=out.dtype)
    return out
