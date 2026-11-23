"""Build tokenizer features for serial chains of revolute joints.

Fold fixed joints and express each active joint axis along its local Z axis.
Each frame stores the joint origin as XYZ and roll, pitch, yaw, with the joint
angle appended separately. Pose targets use forward kinematics in the original
URDF frames. Other equivalent choices of joint frame are possible.
"""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Mapping, Sequence

import numpy as np

try:
    import transforms3d as tf3d
except ImportError as exc:
    raise ImportError("urdf_features requires transforms3d: pip install transforms3d") from exc


MOVABLE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}
TOKENIZER_JOINT_TYPES = {"revolute", "continuous"}
SCHEMA_VERSION = "eat.local_z.single_frame.v1"


def _parse_vector(text: str | None, dim: int = 3) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.zeros(dim, dtype=np.float64)
    values = [float(x) for x in str(text).strip().split()]
    if len(values) != dim:
        raise ValueError(f"Expected {dim} values, got {len(values)} from {text!r}")
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite URDF vector: {text!r}")
    return result


def _read_urdf_root(urdf: str | Path | ET.Element) -> ET.Element:
    if isinstance(urdf, ET.Element):
        return copy.deepcopy(urdf)
    if isinstance(urdf, Path):
        urdf = str(urdf)
    if isinstance(urdf, str):
        if "<robot" in urdf:
            return ET.fromstring(urdf)
        with open(urdf, "r", encoding="utf-8") as file:
            return ET.parse(file).getroot()
    raise TypeError("urdf must be a file path, XML string, or ElementTree root")


def _index_tree(root: ET.Element):
    """Validate a resolved URDF tree; XML document order has no semantics."""
    if root.tag != "robot" or any("xacro" in str(e.tag) for e in root.iter()):
        raise ValueError("Expected a resolved URDF <robot>; expand Xacro first")
    links = {}
    joints = {}
    incoming = {}
    children = defaultdict(list)
    for link in root.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError(f"Missing or duplicate link name: {name!r}")
        links[name] = link
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or name in joints:
            raise ValueError(f"Missing or duplicate joint name: {name!r}")
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {name!r} requires parent and child links")
        p, c = parent.get("link"), child.get("link")
        if p not in links or c not in links or c in incoming:
            raise ValueError(f"Invalid or multiply-parented edge {name!r}: {p}->{c}")
        joints[name] = joint
        incoming[c] = joint
        children[p].append((name, c))
    roots = set(links) - set(incoming)
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root, found {sorted(roots)}")
    root_link = next(iter(roots))
    seen = set()
    queue = deque([root_link])
    while queue:
        link = queue.popleft()
        if link in seen:
            raise ValueError("URDF graph contains a cycle")
        seen.add(link)
        queue.extend(child for _, child in children[link])
    if seen != set(links):
        raise ValueError("URDF contains a disconnected component or cycle")
    return joints, links, incoming, children, root_link


def forward_kinematics_tree(
    urdf: str | Path | ET.Element,
    joint_positions: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Independent float64 graph FK using original local axes (root pose=I).

    Unspecified independent joints use zero; mimic joints are resolved from their
    source.  This utility supports prismatic branches, but the tokenizer's selected
    chain deliberately rejects prismatic/mimic joints until its codec supports them.
    """
    root = _read_urdf_root(urdf)
    joints, _, _, children, root_link = _index_tree(root)
    supplied = {} if joint_positions is None else dict(joint_positions)
    if set(supplied) - set(joints):
        raise ValueError(f"Unknown q joint names: {sorted(set(supplied) - set(joints))}")
    values = {}

    def joint_q(name, visiting=None):
        if name in values:
            return values[name]
        visiting = set() if visiting is None else set(visiting)
        if name in visiting or name not in joints:
            raise ValueError(f"Invalid mimic dependency at {name!r}")
        visiting.add(name)
        mimic = joints[name].find("mimic")
        if mimic is not None:
            if name in supplied:
                raise ValueError(f"Do not supply an independent q for mimic joint {name!r}")
            value = float(mimic.get("multiplier", "1")) * joint_q(mimic.get("joint"), visiting)
            value += float(mimic.get("offset", "0"))
        else:
            value = float(supplied.get(name, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite joint position for {name!r}")
        values[name] = value
        return value

    poses = {root_link: np.eye(4, dtype=np.float64)}
    queue = deque([root_link])
    while queue:
        parent = queue.popleft()
        for name, child in children[parent]:
            joint = joints[name]
            motion = np.eye(4, dtype=np.float64)
            kind = joint.get("type")
            if kind in MOVABLE_JOINT_TYPES:
                element = joint.find("axis")
                axis = _parse_vector(element.get("xyz", "1 0 0") if element is not None else "1 0 0")
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError(f"Zero-length axis for {name!r}")
                axis /= norm
                if kind == "prismatic":
                    motion[:3, 3] = axis * joint_q(name)
                else:
                    motion[:3, :3] = tf3d.axangles.axangle2mat(axis, joint_q(name))
            elif kind != "fixed":
                raise ValueError(f"Unsupported joint type {kind!r} at {name!r}")
            poses[child] = poses[parent] @ _origin_to_transform(joint) @ motion
            queue.append(child)
    return poses


def _origin_to_transform(element: ET.Element) -> np.ndarray:
    """Return the fixed ``origin`` transform attached to a URDF element."""
    origin = element.find("origin")
    xyz = _parse_vector(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
    rpy = _parse_vector(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = tf3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], "sxyz")
    transform[:3, 3] = xyz
    return transform


def _set_origin_transform(element: ET.Element, transform: np.ndarray) -> None:
    """Write a homogeneous transform to an element's URDF ``origin``."""
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    rpy = tf3d.euler.mat2euler(np.asarray(transform[:3, :3]), "sxyz")
    origin.set("xyz", " ".join(f"{value:.12g}" for value in transform[:3, 3]))
    origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))


def _rotation_z_to_axis(axis: np.ndarray) -> np.ndarray:
    """Return the deterministic minimal rotation ``C`` satisfying ``C @ z = axis``."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Movable URDF joint has a non-finite axis: {axis.tolist()}")
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise ValueError("Movable URDF joint has a zero-length axis")
    axis = axis / axis_norm

    cosine = float(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
    cross = np.cross(z_axis, axis)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-14 and cosine > 0:
        rotation = np.eye(3, dtype=np.float64)
    elif sine < 1e-14 and cosine < 0:
        # The antiparallel case has infinitely many solutions.  Fix +X as the
        # rotation axis so identical URDFs always produce identical features.
        rotation = tf3d.euler.euler2mat(np.pi, 0.0, 0.0, "sxyz")
    else:
        rotation_axis = cross / sine
        angle = np.arctan2(sine, cosine)
        rotation = tf3d.axangles.axangle2mat(rotation_axis, angle)

    if not np.allclose(rotation @ z_axis, axis, atol=1e-10, rtol=0.0):
        raise ValueError(f"Failed to align canonical Z axis to {axis.tolist()}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Axis canonicalization produced an invalid rotation matrix")
    return rotation


def standardize_joint_axes(urdf: str | ET.Element) -> ET.Element:
    """
    Rewrite every revolute/continuous/prismatic axis to ``[0, 0, 1]`` without
    changing physical kinematics.

    For an original joint ``O_i @ Motion(axis_i, q_i)``, choose a pure child-link
    frame gauge ``C_i`` with ``C_i @ z = axis_i`` and write

        O_i_canonical = inv(C_parent) @ O_i @ C_i.

    The resulting link pose obeys

        T_i_canonical(q) = T_i_original(q) @ C_i,

    so every link/joint position is exactly preserved.  The parent gauge is
    essential: changing only the current joint origin corrupts all descendants.
    Link-local visual, collision, and inertial origins are counter-rotated so a
    serialized canonical URDF also keeps the original geometry in place.
    """
    root = _read_urdf_root(urdf)

    joint_map, link_map, _, link_children, root_link = _index_tree(root)
    root_links = {root_link}

    identity = np.eye(4, dtype=np.float64)
    link_gauges: dict[str, np.ndarray] = {}

    for root_link in sorted(root_links):
        link_gauges[root_link] = identity.copy()
        queue = deque([root_link])

        while queue:
            link_name = queue.popleft()
            parent_gauge = link_gauges[link_name]
            parent_gauge_inv = parent_gauge.T.copy()
            parent_gauge_inv[3, :] = np.array([0.0, 0.0, 0.0, 1.0])

            for joint_name, child_link_name in link_children.get(link_name, []):
                joint = joint_map[joint_name]
                joint_type = joint.get("type", "")
                if joint_type not in MOVABLE_JOINT_TYPES | {"fixed"}:
                    raise ValueError(f"Unsupported joint type {joint_type!r} at {joint_name!r}")
                current_transform = _origin_to_transform(joint)

                child_gauge = identity.copy()
                if joint_type in MOVABLE_JOINT_TYPES:
                    axis_elem = joint.find("axis")
                    if axis_elem is None:
                        # URDF specifies +X as the default revolute/prismatic axis.
                        axis_elem = ET.SubElement(joint, "axis", xyz="1 0 0")
                    current_axis = _parse_vector(axis_elem.get("xyz", "1 0 0"))
                    child_gauge[:3, :3] = _rotation_z_to_axis(current_axis)
                    axis_elem.set("xyz", "0 0 1")

                canonical_transform = parent_gauge_inv @ current_transform @ child_gauge
                _set_origin_transform(joint, canonical_transform)

                link_gauges[child_link_name] = child_gauge
                queue.append(child_link_name)

    # A canonical link frame is T_original @ C.  Therefore a fixed element pose
    # relative to that link must become inv(C) @ element_pose to leave its world
    # pose unchanged.  Keeping inertial origins aligned also leaves their inertia
    # tensors expressed in the same physical inertial frame.
    for link_name, link in link_map.items():
        gauge = link_gauges.get(link_name, identity)
        gauge_inv = gauge.T.copy()
        gauge_inv[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        if np.allclose(gauge, identity, atol=1e-12, rtol=0.0):
            continue
        for element_type in ("visual", "collision", "inertial"):
            for element in link.findall(element_type):
                element_transform = _origin_to_transform(element)
                _set_origin_transform(element, gauge_inv @ element_transform)

    return root


def xyz_rpy_to_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Joint origin xyz + extrinsic xyz rpy -> 4x4 homogeneous transform."""
    rot = tf3d.euler.euler2mat(float(rpy[0]), float(rpy[1]), float(rpy[2]), "sxyz")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def matrix_to_xyz_rpy(transform: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> joint origin xyz + extrinsic xyz rpy."""
    xyz = np.asarray(transform[:3, 3], dtype=np.float64)
    rpy = tf3d.euler.mat2euler(transform[:3, :3], "sxyz")
    return np.concatenate([xyz, rpy], axis=0)


def compose_base_into_first_joint(
    features: np.ndarray,
    base_transform: np.ndarray,
) -> np.ndarray:
    """
    Pre-compose the first joint's parent-link transform into joint-0 origin.

    Requires standardized URDF where every revolute joint uses axis [0, 0, 1] and
    any former axis tilt is already absorbed into joint origin rpy. Under that
    convention diff_fk applies joint motion as RotZ(angle) after each origin.

    After this, serial-chain FK from identity base matches full URDF-tree FK:
        T_i = T_{i-1} @ origin_i @ RotZ(angle_i),  T_{-1} = I
    is equivalent to:
        T_i = T_base @ origin_i @ RotZ(angle_i) @ ...   (original joint-0 parent frame)

    For single-arm robots whose first parent link is the identity frame, this is a no-op.
    """
    features = np.asarray(features, dtype=np.float32).copy()
    if features.shape[0] == 0:
        return features

    base = np.asarray(base_transform, dtype=np.float64)
    if np.array_equal(base, np.eye(4)):
        return features

    origin = xyz_rpy_to_matrix(features[0, :3], features[0, 3:6])
    features[0] = matrix_to_xyz_rpy(base @ origin).astype(np.float32)
    return features


def compute_first_joint_parent_transform(
    urdf: str | ET.Element,
    joint_names: Sequence[str],
) -> np.ndarray:
    """
    World transform of the first arm joint's parent link at zero joint angles.

    The supplied URDF's frame convention is preserved; original axes are valid.

    For robots mounted directly at the URDF root this is identity; for aloha-like
    embodiments with a mobile-base chain before the arm, this captures that prefix.
    """
    if not joint_names:
        return np.eye(4, dtype=np.float64)

    root = _read_urdf_root(urdf)
    joints, _, _, _, _ = _index_tree(root)
    if joint_names[0] not in joints:
        raise KeyError(f"Joint {joint_names[0]!r} not found")
    parent_link = joints[joint_names[0]].find("parent").get("link")
    return forward_kinematics_tree(root)[parent_link]


def _selected_chain(root: ET.Element, joint_names: Sequence[str] | None):
    if joint_names is None:
        raise ValueError("Pass explicit joint_names in controller/serial-chain order; XML order is not joint order")
    names = list(joint_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("joint_names must be a nonempty list without duplicates")
    joints, _, incoming, _, _ = _index_tree(root)
    for name in names:
        if name not in joints:
            raise KeyError(f"Joint {name!r} not found")
        joint = joints[name]
        if joint.get("type") not in TOKENIZER_JOINT_TYPES:
            raise ValueError(f"EAT RotZ codec supports revolute/continuous only; {name!r} is {joint.get('type')!r}")
        if joint.find("mimic") is not None:
            raise ValueError(f"Selected mimic joint {name!r} is not an independent action dimension")
    base_link = joints[names[0]].find("parent").get("link")
    previous = base_link
    origins = []
    fixed_paths = []
    for name in names:
        joint = joints[name]
        transform = _origin_to_transform(joint)
        cursor = joint.find("parent").get("link")
        path = []
        while cursor != previous:
            if cursor not in incoming:
                raise ValueError(f"Selected joints are not in one parent-to-child chain at {name!r}")
            upstream = incoming[cursor]
            if upstream.get("type") != "fixed":
                raise ValueError(f"Joint order omits/crosses movable joint {upstream.get('name')!r} before {name!r}")
            transform = _origin_to_transform(upstream) @ transform
            path.insert(0, upstream.get("name"))
            cursor = upstream.find("parent").get("link")
        origins.append(transform)
        fixed_paths.append(path)
        previous = joint.find("child").get("link")
    return names, joints, base_link, origins, fixed_paths


def extract_fk_urdf_features(
    urdf: str | ET.Element,
    joint_names: Sequence[str] | None = None,
    exclude_gripper: bool = True,
    gripper_keywords: Sequence[str] = (),
    embed_arm_base: bool = False,
    return_torch: bool = False,
    tip_link: str | None = None,
):
    """
    Extract URDF joint features ready for diff_fk / serial-chain FK.

    Explicit joint order is required. Only independent revolute/continuous
    actions are accepted. Fixed paths are folded; skipped movable joints fail.
    With the ORIGINAL first-joint parent as base, O'_i=inv(C_previous) O_i C_i,
    C_i Z=axis_i. q, its sign/zero, and physical joint positions are preserved.
    Child-frame orientations differ by the returned joint_frame_corrections.
    An explicit fixed-descendant tip gets a compensating terminal transform.

    No q wrapping, normalization, guessed TCP or controller remapping occurs.
    embed_arm_base=True adds the root-to-base HOME transform, assuming unselected
    upstream joints remain at their declared zero. exclude_gripper/keywords are
    retained for source compatibility; explicit joint_names always take priority.
    """
    source = _read_urdf_root(urdf)
    names, joints, base_link, origins, fixed_paths = _selected_chain(source, joint_names)
    if embed_arm_base:
        # Independent unselected prefix joints stay at q=0. A prefix mimic may
        # instead move as a function of a selected q, making a static embedded
        # base inconsistent with source-FK labels. Keep that case explicit.
        _, _, incoming, _, _ = _index_tree(source)
        cursor = base_link
        while cursor in incoming:
            upstream = incoming[cursor]
            if upstream.find("mimic") is not None:
                raise ValueError("embed_arm_base does not support upstream mimic joints; use arm-relative features")
            cursor = upstream.find("parent").get("link")
    gauges, canonical_origins = [], []
    previous_gauge = np.eye(4, dtype=np.float64)
    for name, origin in zip(names, origins):
        axis = joints[name].find("axis")
        direction = _parse_vector(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0")
        gauge = np.eye(4, dtype=np.float64)
        gauge[:3, :3] = _rotation_z_to_axis(direction)
        canonical_origins.append(previous_gauge.T @ origin @ gauge)
        gauges.append(gauge)
        previous_gauge = gauge
    features = np.stack([matrix_to_xyz_rpy(t) for t in canonical_origins]).astype(np.float32)
    base_transform = forward_kinematics_tree(source)[base_link]
    if embed_arm_base:
        features = compose_base_into_first_joint(features, base_transform)
    if not np.all(np.isfinite(features)):
        raise ValueError("URDF features overflow float32 storage")
    result = {
        "features": features,
        "joint_names": names,
        "joint_types": [joints[name].get("type") for name in names],
        "child_links": [joints[name].find("child").get("link") for name in names],
        "base_link": base_link,
        "first_joint_parent_transform": base_transform,
        "joint_frame_corrections": np.stack(gauges),
        "fixed_paths": fixed_paths,
        "embed_arm_base": bool(embed_arm_base),
        "schema_version": SCHEMA_VERSION,
        "source_root": source,
    }
    if tip_link is not None:
        _, links, incoming, _, _ = _index_tree(source)
        if tip_link not in links:
            raise KeyError(f"Unknown tip link {tip_link!r}")
        terminal = np.eye(4, dtype=np.float64)
        cursor = tip_link
        last_child = result["child_links"][-1]
        while cursor != last_child:
            if cursor not in incoming or incoming[cursor].get("type") != "fixed":
                raise ValueError("tip_link must be the last child or its fixed descendant")
            edge = incoming[cursor]
            terminal = _origin_to_transform(edge) @ terminal
            cursor = edge.find("parent").get("link")
        result["tip_link"] = tip_link
        result["tip_transform"] = gauges[-1].T @ terminal

    if return_torch:
        import torch

        result["features"] = torch.from_numpy(result["features"])
        result["first_joint_parent_transform"] = torch.from_numpy(result["first_joint_parent_transform"])

    return result


def extract_urdf_joint_features(
    urdf: str | ET.Element,
    joint_names: Sequence[str] | None = None,
    exclude_gripper: bool = True,
    gripper_keywords: Sequence[str] = (),
    return_torch: bool = False,
):
    """Alias for validated, fixed-path-aware FK features. No XML-order fallback."""
    return extract_fk_urdf_features(
        urdf, joint_names, exclude_gripper, gripper_keywords, return_torch=return_torch
    )


def prepare_tokenizer_chain(urdf: str | Path | ET.Element, joint_names: Sequence[str], **kwargs) -> dict:
    """Prepare per-robot metadata once, before iterating independent action frames."""
    if kwargs.get("return_torch"):
        raise ValueError("Offline frame construction requires NumPy features")
    return extract_fk_urdf_features(urdf, joint_names=joint_names, **kwargs)


def serial_fk(features: np.ndarray, joint_angles: np.ndarray) -> np.ndarray:
    """Float64 local-Z FK oracle for feature validation, not a training loss."""
    features = np.asarray(features, dtype=np.float64)
    q = np.asarray(joint_angles, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 6 or q.shape != (len(features),) or not len(features):
        raise ValueError("Expected nonempty features (J,6) and q (J,)")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(q)):
        raise ValueError("FK inputs must be finite")
    accumulated = np.eye(4, dtype=np.float64)
    poses = []
    for origin, angle in zip(features, q):
        motion = np.eye(4)
        motion[:3, :3] = tf3d.axangles.axangle2mat([0.0, 0.0, 1.0], angle)
        accumulated = accumulated @ xyz_rpy_to_matrix(origin[:3], origin[3:]) @ motion
        poses.append(accumulated.copy())
    return np.stack(poses)


def build_tokenizer_frame(chain: dict, joint_angles: np.ndarray, max_joints: int = 8) -> dict:
    """One frame: raw input (Jmax,7), source-FK target (Jmax,10), boolean mask.

    Labels come from original-URDF graph FK plus stored frame corrections, NOT
    serial_fk(features,q). Rotation6d is column 1 then column 2, matching EAT.
    """
    names = chain["joint_names"]
    q = np.asarray(joint_angles, dtype=np.float64)
    if q.shape != (len(names),) or not np.all(np.isfinite(q)):
        raise ValueError(f"Expected {len(names)} finite q values in declared joint_names order")
    if not isinstance(max_joints, int) or isinstance(max_joints, bool) or max_joints < len(names):
        raise ValueError("max_joints would truncate real joints")
    if np.any(np.abs(q) > np.finfo(np.float32).max):
        raise ValueError("Joint positions overflow float32 storage")
    source_poses = forward_kinematics_tree(chain["source_root"], dict(zip(names, q)))
    base_inverse = np.eye(4) if chain["embed_arm_base"] else np.linalg.inv(source_poses[chain["base_link"]])
    inp = np.zeros((max_joints, 7), dtype=np.float32)
    output = np.zeros((max_joints, 10), dtype=np.float32)
    mask = np.zeros(max_joints, dtype=bool)
    inp[:len(names), :6] = chain["features"]
    inp[:len(names), 6] = q
    for i, child in enumerate(chain["child_links"]):
        pose = base_inverse @ source_poses[child] @ chain["joint_frame_corrections"][i]
        output[i, :3] = pose[:3, 3]
        output[i, 3:6] = pose[:3, 0]
        output[i, 6:9] = pose[:3, 1]
        output[i, 9] = q[i]
    mask[:len(names)] = True
    if not np.all(np.isfinite(inp)) or not np.all(np.isfinite(output)):
        raise ValueError("Tokenizer frame overflows float32 storage")
    return {"input": inp, "output": output, "joint_valid_mask": mask}


# Preset arm joint names for the 5 benchmark embodiments.
EMBODIMENT_ARM_JOINTS = {
    "piper": [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ],
    "ur5-wsg": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
    "franka-panda": [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "aloha-agilex-left": [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
    ],
    "aloha-agilex-right": [
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ],
    "arx-x5": [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ],
}


def extract_embodiment_arm_features(
    urdf_path: str,
    embodiment_key: str,
    return_torch: bool = False,
):
    """Convenience wrapper for the 5 benchmark embodiments.

    Returns FK-ready features re-rooted at the arm base (see
    extract_fk_urdf_features); ``first_joint_parent_transform`` maps the
    arm-base frame back to the world frame.
    """
    if embodiment_key not in EMBODIMENT_ARM_JOINTS:
        valid = ", ".join(sorted(EMBODIMENT_ARM_JOINTS.keys()))
        raise KeyError(f"Unknown embodiment_key={embodiment_key!r}. Valid keys: {valid}")
    return extract_fk_urdf_features(
        urdf_path,
        joint_names=EMBODIMENT_ARM_JOINTS[embodiment_key],
        exclude_gripper=True,
        return_torch=return_torch,
    )
