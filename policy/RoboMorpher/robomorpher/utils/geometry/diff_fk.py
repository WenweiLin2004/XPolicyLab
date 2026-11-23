"""
Differentiable forward kinematics from URDF joint features + joint angles.

Assumes standardized URDF (revolute joints, local axis [0, 0, 1]) in serial chain order,
matching EmbodiActionTokenizer data: each joint feature is origin xyz (3) + origin rpy (3).
The origin rpy already includes any axis-alignment rotation from standardize_joint_axes;
joint motion is always applied as RotZ(angle) in the local frame after origin.

Joint-0 origin should include the parent-link base transform when the arm is not mounted
at the URDF root (see urdf_features.extract_fk_urdf_features).

No URDF files required; fully PyTorch-differentiable for backpropagation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def euler_xyz_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Extrinsic xyz Euler angles (scipy 'xyz') to rotation matrix."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def rotation_z_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Rotation about Z axis. angle shape (...), returns (..., 3, 3)."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def make_transform(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Build homogeneous transform (..., 4, 4) from (..., 3) and (..., 3, 3)."""
    *batch_shape, _ = translation.shape
    transform = torch.zeros(*batch_shape, 4, 4, dtype=translation.dtype, device=translation.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def transform_from_xyz_rpy(xyz: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
    return make_transform(xyz, euler_xyz_to_matrix(rpy))


def transform_from_rot_z(angle: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(*angle.shape, 3, dtype=angle.dtype, device=angle.device)
    return make_transform(zero, rotation_z_matrix(angle))


def compose_transforms(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    return parent @ child


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Match EmbodiTokenizer rotation_matrix_to_6d: first two normalized columns."""
    a = F.normalize(rotation[..., :, 0], dim=-1)
    b = rotation[..., :, 1]
    b = b - (a * b).sum(dim=-1, keepdim=True) * a
    b = F.normalize(b, dim=-1)
    return torch.cat([a, b], dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Rotation 6D (first two columns) -> orthonormal rotation matrix."""
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def transform_to_pose6d(transform: torch.Tensor) -> torch.Tensor:
    """(4, 4) transform -> [xyz(3), rot6d(6)]."""
    position = transform[..., :3, 3]
    rot6d = matrix_to_rotation_6d(transform[..., :3, :3])
    return torch.cat([position, rot6d], dim=-1)


def embed_arm_base_in_urdf_features(
    urdf_features: torch.Tensor,
    base_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pre-compose the first-joint parent-link transform into joint-0 origin.

    Use when urdf_features still contain raw joint-local origins (data).
    FK-ready features from ``extract_fk_urdf_features`` do not need this call.
    """
    if base_transform is None:
        return urdf_features

    features = urdf_features.clone()
    base = base_transform.to(dtype=features.dtype, device=features.device)
    if base.dim() == 2:
        for _ in range(features.dim() - 3):
            base = base.unsqueeze(0)
    elif base.dim() == 3 and base.shape[0] == 1:
        for _ in range(features.dim() - 4):
            base = base.unsqueeze(0)
    if base.shape[:-2] != features.shape[:-2]:
        base = base.expand(*features.shape[:-2], 4, 4)

    if features.shape[-1] == 6:
        origin_rotation = euler_xyz_to_matrix(features[..., 0, 3:6])
    elif features.shape[-1] == 9:
        origin_rotation = rotation_6d_to_matrix(features[..., 0, 3:9])
    else:
        raise ValueError(
            "URDF features must be xyz+rpy (6D) or xyz+rot6d (9D), "
            f"got {features.shape[-1]}D"
        )
    origin = make_transform(features[..., 0, :3], origin_rotation)
    composed = compose_transforms(base, origin)
    features[..., 0, :3] = composed[..., :3, 3]
    if features.shape[-1] == 6:
        features[..., 0, 3:6] = _matrix_to_xyz_rpy(composed[..., :3, :3])
    else:
        features[..., 0, 3:9] = matrix_to_rotation_6d(composed[..., :3, :3])
    return features


def _matrix_to_xyz_rpy(rotation: torch.Tensor) -> torch.Tensor:
    """Differentiable enough for constant URDF features; batch (... , 3, 3)."""
    flat = rotation.reshape(-1, 3, 3)
    rpys = []
    for mat in flat:
        r00, r01, r02 = mat[0]
        r10, r11, r12 = mat[1]
        r20, r21, r22 = mat[2]
        pitch = torch.asin((-r20).clamp(-1.0, 1.0))
        roll = torch.atan2(r21, r22)
        yaw = torch.atan2(r10, r00)
        rpys.append(torch.stack([roll, pitch, yaw]))
    return torch.stack(rpys, dim=0).reshape(*rotation.shape[:-2], 3).to(dtype=rotation.dtype, device=rotation.device)


def forward_kinematics_6d(
    urdf_features: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_valid_mask: torch.Tensor | None = None,
    first_joint_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Serial-chain FK for standardized URDF features.

    Args:
        urdf_features: (..., num_joints, 6 or 9), either origin xyz+rpy or
            xyz+rotation6d per joint.
            Joint-0 should already include the parent-link base transform when the arm
            is not mounted at the URDF root (see urdf_features.extract_fk_urdf_features).
        joint_angles: (..., num_joints) or (..., num_joints, 1).
        joint_valid_mask: optional (..., num_joints) bool/float mask.
        first_joint_base: optional (..., 4, 4) parent-link transform for joint 0.
            Applied on-the-fly for features that omit the embedded base.

    Returns:
        Per-joint global pose (..., num_joints, 9) = xyz(3) + rot6d(6).
    """
    urdf_features = embed_arm_base_in_urdf_features(urdf_features, first_joint_base)
    if joint_angles.shape[-1] == 1:
        joint_angles = joint_angles.squeeze(-1)

    num_joints = urdf_features.shape[-2]
    batch_shape = urdf_features.shape[:-2]
    dtype = urdf_features.dtype
    device = urdf_features.device

    parent = torch.eye(4, dtype=dtype, device=device).expand(*batch_shape, 4, 4).contiguous()
    poses = []

    for joint_idx in range(num_joints):
        xyz = urdf_features[..., joint_idx, :3]
        angle = joint_angles[..., joint_idx]

        if urdf_features.shape[-1] == 6:
            origin_rotation = euler_xyz_to_matrix(
                urdf_features[..., joint_idx, 3:6]
            )
        elif urdf_features.shape[-1] == 9:
            origin_rotation = rotation_6d_to_matrix(
                urdf_features[..., joint_idx, 3:9]
            )
        else:
            raise ValueError(
                "URDF features must be xyz+rpy (6D) or xyz+rot6d (9D), "
                f"got {urdf_features.shape[-1]}D"
            )
        origin = make_transform(xyz, origin_rotation)
        motion = transform_from_rot_z(angle)
        joint_transform = compose_transforms(origin, motion)
        global_transform = compose_transforms(parent, joint_transform)
        poses.append(transform_to_pose6d(global_transform))
        parent = global_transform

    pose = torch.stack(poses, dim=-2)
    if joint_valid_mask is not None:
        mask = joint_valid_mask.to(dtype=pose.dtype, device=pose.device)
        while mask.dim() < pose.dim():
            mask = mask.unsqueeze(-1)
        pose = pose * mask
    return pose


class DifferentiableFK(nn.Module):
    """nn.Module wrapper for differentiable FK (supports backpropagation)."""

    def forward(
        self,
        urdf_features: torch.Tensor,
        joint_angles: torch.Tensor,
        joint_valid_mask: torch.Tensor | None = None,
        first_joint_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return forward_kinematics_6d(
            urdf_features,
            joint_angles,
            joint_valid_mask,
            first_joint_base=first_joint_base,
        )
