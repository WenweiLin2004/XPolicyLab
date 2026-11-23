#!/usr/bin/env python3
"""Convert RoboTwin HDF5 trees to the standard LeRobot v3 contract.

Native RoboTwin HDF5 state/action are used directly.  No second temporal shift
is performed.  Every source HDF5 becomes one LeRobot episode.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import h5py
import numpy as np
from robomorpher.utils.data.process_data.bench_paths import robotwin_assets, robotwin_data, stored_path
from robomorpher.utils.data.process_data.decode_image_bit import decode_image_bit
from robomorpher.utils.data.process_data.action_stats import write_masked_action_stats

def flatten_state(sample, max_joints):
    q = np.asarray(sample["state_angle"], dtype=np.float32)
    g = np.asarray(sample["state_gripper"], dtype=np.float32).reshape(2)
    if q.shape != (2, max_joints):
        raise ValueError(f"state_angle shape {q.shape}, expected {(2, max_joints)}")
    return np.concatenate([q[0], g[:1], q[1], g[1:]]).astype(np.float32)

def flatten_action(sample, max_joints):
    q = np.asarray(sample["action_angle"], dtype=np.float32)
    g = np.asarray(sample["gripper"], dtype=np.float32).reshape(1, 2)
    if q.shape != (1, 2, max_joints):
        raise ValueError(f"action_angle shape {q.shape}, expected {(1, 2, max_joints)}")
    return np.concatenate([q[0, 0], g[0, :1], q[0, 1], g[0, 1:]]).astype(np.float32)

def flatten_urdf(sample, max_joints):
    f = np.asarray(sample["urdf_features"], dtype=np.float32)
    m = np.asarray(sample["joint_valid_mask"], dtype=np.float32)
    if f.shape != (2, max_joints, 6) or m.shape != (2, max_joints):
        raise ValueError(f"unexpected URDF shapes: {f.shape}, {m.shape}")
    return f.reshape(-1), m.reshape(-1)

def read_native_vectors(path, episode, max_joints):
    """Read native RoboTwin /state and /action without temporal shifting."""
    dimensions = list(episode["dimensions"])
    width = 2 * (max_joints + 1)
    with h5py.File(path, "r") as handle:
        state = np.zeros((int(episode["source_length"]), width), dtype=np.float32)
        action = np.zeros_like(state)
        for side, (arm_key, grip_key) in enumerate(episode["state_keys"]):
            if not dimensions[side]:
                continue
            sq = np.asarray(handle[f"state/{arm_key}"], dtype=np.float32)
            sg = np.asarray(handle[f"state/{grip_key}"], dtype=np.float32)
            aq = np.asarray(handle[f"action/{arm_key}"], dtype=np.float32)
            ag = np.asarray(handle[f"action/{grip_key}"], dtype=np.float32)
            if sq.shape != aq.shape or sg.shape != ag.shape or sg.ndim != 2 or sg.shape[1] != 1:
                raise ValueError(f"{path}: native state/action shape mismatch for {side}")
            if sq.shape[0] != state.shape[0] or not all(np.isfinite(x).all() for x in (sq, sg, aq, ag)):
                raise ValueError(f"{path}: invalid native state/action values")
            offset = side * (max_joints + 1)
            joints = dimensions[side]
            state[:, offset:offset + joints] = sq
            state[:, offset + max_joints] = sg[:, 0]
            action[:, offset:offset + joints] = aq
            action[:, offset + max_joints] = ag[:, 0]
    return state, action

def create_dataset(root, features, fps, robot_type, videos, writer_processes, writer_threads):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("Install the StarVLA/LeRobot environment before conversion") from exc
    old_home = os.environ.get("HF_LEROBOT_HOME")
    os.environ["HF_LEROBOT_HOME"] = str(root.parent)
    kwargs = dict(repo_id=root.name, root=root, fps=int(fps), robot_type=robot_type,
                  features=features, use_videos=videos,
                  image_writer_processes=writer_processes,
                  image_writer_threads=writer_threads)
    try:
        try:
            ds = LeRobotDataset.create(**kwargs, streaming_encoding=True, tolerance_s=0.0001, encoder_threads=4)
        except TypeError:
            ds = LeRobotDataset.create(**kwargs)
    finally:
        if old_home is None:
            os.environ.pop("HF_LEROBOT_HOME", None)
        else:
            os.environ["HF_LEROBOT_HOME"] = old_home
    actual = Path(getattr(ds, "root", root)).resolve()
    if actual != root.resolve():
        raise RuntimeError(f"LeRobot created {actual}, expected {root}")
    return ds

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")

def convert(input_dir, output_dir, *, assets_root, robot_type="robotwin", max_joints=7,
            cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
            videos=True, writer_processes=4, writer_threads=2, overwrite=False,
            action_type="joint", embodiment=None):
    action_type = str(action_type).lower()
    if action_type != "joint":
        raise ValueError("action_type must be joint")
    from robomorpher.utils.data.process_data.robotwin_source import RoboTwinSourceIndex

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; use --overwrite explicitly")
        shutil.rmtree(output_dir)

    source = RoboTwinSourceIndex(
        input_dir, max_num_joints=max_joints,
        urdf_root=assets_root, camera_names=tuple(cameras),
        allow_missing_urdf=False, embodiment=embodiment,
    )
    if not source.episodes:
        raise ValueError("No valid HDF5 episodes found")
    first_episode = source.episodes[0]
    first_camera_keys = [source._camera_key(camera) for camera in cameras]
    with h5py.File(first_episode["path"], "r") as first_file:
        image_arrays = [decode_image_bit(first_file[key][0]) for key in first_camera_keys]
    if len(image_arrays) != len(cameras):
        raise ValueError("The reader returned a different number of cameras")
    image_shape = image_arrays[0].shape
    if any(x.shape != image_shape for x in image_arrays):
        raise ValueError("The first frame has inconsistent camera shapes")

    state_dim = 2 * (max_joints + 1)
    features = {
            "observation.state": {
                "dtype": "float32", "shape": (state_dim,),
                "names": [[*[f"left_joint_{i}" for i in range(max_joints)], "left_gripper", *[f"right_joint_{i}" for i in range(max_joints)], "right_gripper"]],
            },
            "action": {
                "dtype": "float32", "shape": (state_dim,),
                "names": [[*[f"left_joint_{i}" for i in range(max_joints)], "left_gripper", *[f"right_joint_{i}" for i in range(max_joints)], "right_gripper"]],
            },
            "observation.urdf_features": {
                "dtype": "float32", "shape": (2 * max_joints * 6,),
                "names": [[f"urdf_{i}" for i in range(2 * max_joints * 6)]],
            },
            "observation.joint_valid_mask": {
                "dtype": "float32", "shape": (2 * max_joints,),
                "names": [[f"joint_valid_{i}" for i in range(2 * max_joints)]],
            },
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if videos else "image",
            "shape": (3, image_shape[0], image_shape[1]),
            "names": ["channels", "height", "width"],
        }

    ds = create_dataset(output_dir, features, 15, robot_type, videos,
                        writer_processes, writer_threads)
    frame_count = 0
    sources = []
    try:
        for episode_index, episode in enumerate(source.episodes):
            state_rows, action_rows = read_native_vectors(episode["path"], episode, max_joints)
            urdf = np.asarray(episode["urdf_features"], dtype=np.float32).reshape(-1)
            mask = np.asarray(episode["joint_valid_mask"], dtype=np.float32).reshape(-1)
            with h5py.File(episode["path"], "r") as handle:
                image_keys = [source._camera_key(camera) for camera in cameras]
                for local_index in range(len(state_rows)):
                    state = state_rows[local_index]
                    action = action_rows[local_index]
                    frame = {
                        "observation.state": state,
                        "action": action,
                        "task": str(episode["instructions"][0]),
                    }
                    frame["observation.urdf_features"] = urdf
                    frame["observation.joint_valid_mask"] = mask
                    for camera, key in zip(cameras, image_keys):
                        array = np.asarray(decode_image_bit(handle[key][local_index]))
                        if array.shape != image_shape:
                            raise ValueError(f"Image shape changed in {episode['path']}: {array.shape}")
                        frame[f"observation.images.{camera}"] = array.astype(np.uint8, copy=False)
                    ds.add_frame(frame)
                    frame_count += 1
            ds.save_episode()
            sources.append({
                "path": stored_path(episode["path"]),
                "frames": int(len(state_rows)),
                "urdf_paths": episode.get("urdf_paths"),
                "dimensions": episode.get("dimensions"),
            })
            print(f"written episode {episode_index + 1}/{len(source.episodes)} "
                  f"frames={len(state_rows)}", flush=True)
    finally:
        if hasattr(ds, "stop_image_writer"):
            ds.stop_image_writer()
        if hasattr(ds, "finalize"):
            ds.finalize()
        elif getattr(ds, "meta", None) is not None and hasattr(ds.meta, "_close_writer"):
            ds.meta._close_writer()

    j = max_joints
    slices = {
        "left_arm": {"start": 0, "end": j, "dtype": "float32", "absolute": True},
        "left_gripper": {"start": j, "end": j + 1, "dtype": "float32", "absolute": True},
        "right_arm": {"start": j + 1, "end": 2 * j + 1, "dtype": "float32", "absolute": True},
        "right_gripper": {"start": 2 * j + 1, "end": 2 * j + 2, "dtype": "float32", "absolute": True},
    }
    write_json(output_dir / "meta" / "modality.json", {
        "state": slices, "action": slices,
        "video": {camera: {"original_key": f"observation.images.{camera}"}
                  for camera in cameras},
        "annotation": {"human.task_description": {"original_key": "task"}},
    })
    write_json(output_dir / "meta" / "dataset_metadata.json", {
        "schema_version": "robotwin.lerobot.v3",
        "action_type": action_type,
        "source_format": "RoboTwin HDF5",
        "state_action_alignment": "native state/action; no second temporal shift",
        "input_dir": stored_path(input_dir), "assets_root": stored_path(assets_root),
        "max_num_joints": j, "state_dim": state_dim,
        "urdf_features_shape": [2, j, 6],
        "joint_valid_mask_shape": [2, j],
        "camera_names": list(cameras), "frames": frame_count,
        "episodes": len(sources), "sources": sources,
    })
    write_json(output_dir / "conversion_manifest.json", {
        "input_dir": stored_path(input_dir), "output_dir": stored_path(output_dir),
        "frames": frame_count, "episodes": len(sources),
        "keys": ["observation.state", "action",
                 *[f"observation.images.{camera}" for camera in cameras],
                 "observation.urdf_features", "observation.joint_valid_mask",
                 "task"],
    })
    write_masked_action_stats(output_dir, action_type=action_type)
    print(f"completed {frame_count} frames / {len(sources)} episodes -> {output_dir}",
          flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", help="Only convert episodes for this RoboTwin robot")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--robot-type", default="robotwin")
    parser.add_argument("--max-joints", type=int, default=7)
    parser.add_argument("--camera-names", nargs="+",
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--writer-processes", type=int, default=4)
    parser.add_argument("--writer-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--action-type", choices=("joint",), default="joint")
    args = parser.parse_args()
    input_dir = args.input_dir or robotwin_data()
    assets_root = args.assets_root or robotwin_assets()
    convert(input_dir, args.output_dir, assets_root=assets_root,
            robot_type=args.robot_type, max_joints=args.max_joints,
            cameras=tuple(args.camera_names), videos=not args.no_videos,
            writer_processes=args.writer_processes, writer_threads=args.writer_threads,
            overwrite=args.overwrite, action_type=args.action_type,
            embodiment=args.embodiment)

if __name__ == "__main__":
    main()
