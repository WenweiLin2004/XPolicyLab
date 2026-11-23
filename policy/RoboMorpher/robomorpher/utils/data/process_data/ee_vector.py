"""Pack native HDF5 end-effector poses into the QwenPI EE vector.

HDF5 ``{side}_ee_poses`` is xyz plus a wxyz quaternion. The stored policy
vector is xyz plus rot6d (the first two rotation columns) plus gripper.
State uses the same vector as the action.
"""
from __future__ import annotations

import numpy as np

POSE_DIM = 9
ARM_WIDTH = POSE_DIM + 1


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion norm is not finite")
    w, x, y, z = quat / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose7_to_xyz_rot6d(pose: np.ndarray) -> np.ndarray:
    """Convert (T, 7) xyz+wxyz, or one such row, to xyz+rot6d."""
    pose = np.asarray(pose, dtype=np.float64)
    single = pose.ndim == 1
    if single:
        pose = pose[None]
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"ee pose must have shape (T, 7), got {pose.shape}")
    out = np.empty((pose.shape[0], POSE_DIM), dtype=np.float32)
    out[:, :3] = pose[:, :3]
    for index, quat in enumerate(pose[:, 3:]):
        matrix = quat_wxyz_to_matrix(quat)
        out[index, 3:6] = matrix[:, 0]
        out[index, 6:9] = matrix[:, 1]
    return out[0] if single else out


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Invert the stored first two rotation columns with Gram-Schmidt."""
    columns = np.asarray(rot6d, dtype=np.float64).reshape(6)
    first = columns[:3]
    second = columns[3:]
    first_norm = np.linalg.norm(first)
    if not np.isfinite(first_norm) or first_norm < 1e-8:
        raise ValueError("rot6d first column is degenerate")
    b1 = first / first_norm
    b2 = second - np.dot(b1, second) * b1
    second_norm = np.linalg.norm(b2)
    if not np.isfinite(second_norm) or second_norm < 1e-8:
        raise ValueError("rot6d second column is degenerate")
    b2 = b2 / second_norm
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        ])
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        ])
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array([
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ])
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def xyz_rot6d_to_pose7(pose: np.ndarray) -> np.ndarray:
    """Convert xyz+rot6d back to the simulator xyz+wxyz pose."""
    pose = np.asarray(pose, dtype=np.float64).reshape(POSE_DIM)
    quat = matrix_to_quat_wxyz(rot6d_to_matrix(pose[3:]))
    return np.concatenate([pose[:3], quat]).astype(np.float32)


def pack_arm(pose9: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(pose9.shape[0], 1)
    if pose9.ndim != 2 or pose9.shape[1] != POSE_DIM or gripper.shape[0] != pose9.shape[0]:
        raise ValueError("arm pose must be (T, 9) and gripper (T, 1)")
    return np.concatenate([pose9, gripper], axis=1)


def pack_bimanual(left_pose, left_grip, right_pose, right_grip) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, 20) and a length-2 arm mask. A missing arm is zeros."""
    present = [item is not None for item in (left_pose, right_pose)]
    if not present[0]:
        raise ValueError("the left arm EE pose is required")
    length = int(np.asarray(left_pose).shape[0])
    packed = np.zeros((length, 2 * ARM_WIDTH), dtype=np.float32)
    valid = np.zeros(2, dtype=np.float32)
    for side, (pose, grip) in enumerate(((left_pose, left_grip), (right_pose, right_grip))):
        if pose is None:
            continue
        arm = pack_arm(pose7_to_xyz_rot6d(pose), grip)
        if arm.shape[0] != length:
            raise ValueError("left and right EE sequences have different lengths")
        packed[:, side * ARM_WIDTH:(side + 1) * ARM_WIDTH] = arm
        valid[side] = 1
    return packed, valid
