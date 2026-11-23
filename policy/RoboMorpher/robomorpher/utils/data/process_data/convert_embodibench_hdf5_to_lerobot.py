#!/usr/bin/env python3
"""Convert EmbodiBench HDF5 collections to the shared LeRobot contract.

The HDF5 layout matches RoboTwin: native /state and /action are already
aligned, so this converter does not shift time again. Robot ids come from
seen_embodiment.txt. A single-arm episode has no right arm or right camera;
the right state stays zero and the right image repeats the left wrist.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from robomorpher.utils.data.process_data.bench_paths import embodibench_data, embodibench_task_config, embodibench_urdf
from robomorpher.utils.data.process_data.action_stats import write_masked_action_stats

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_NATIVE_CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def episode_number(path: Path) -> int:
    match = re.search(r"([0-9]+)", path.stem)
    if not match:
        raise ValueError(f"episode path has no index: {path}")
    return int(match.group(1))


def read_embodiment(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids = tuple(int(part.strip()) for part in line.split(",") if part.strip())
        if len(ids) != 2:
            raise ValueError(f"bad embodiment row in {path}: {line!r}")
        rows.append((ids[0], ids[1]))
    if not rows:
        raise ValueError(f"embodiment file is empty: {path}")
    return rows


def collections(data_root: Path, collection_config=None):
    if collection_config is None:
        collection_config = os.environ.get("ROBOMORPHER_COLLECTION_CONFIG")
    if collection_config is not None and (
        not collection_config or Path(collection_config).name != collection_config or collection_config in {".", ".."}
    ):
        raise ValueError(f"Invalid EmbodiBench collection name: {collection_config!r}")

    # Accept a native data directory, a collection, a task or a benchmark root.
    data_dirs = [data_root] if data_root.name == "data" else []
    data_dirs.append(data_root / "data")
    data_dirs.extend(sorted(data_root.glob("*/data")))
    data_dirs.extend(sorted(data_root.glob("*/*/data")))
    visited = set()
    found = []
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            continue
        config_dir = data_dir.parent.resolve()
        if config_dir in visited:
            continue
        episodes = sorted(data_dir.glob("episode*.hdf5"), key=episode_number)
        if not episodes:
            continue
        visited.add(config_dir)
        settings_file = config_dir / "collection_settings.json"
        settings = json.loads(settings_file.read_text()) if settings_file.is_file() else {}
        config = settings.get("task_config") or config_dir.name
        if collection_config and config != collection_config:
            continue
        task = settings.get("task_name") or config_dir.parent.name
        project = settings.get("project_name")
        if not project:
            import yaml
            with embodibench_task_config(config).open() as handle:
                project = yaml.safe_load(handle)["project_name"]
        seen = config_dir / "seen_embodiment.txt"
        if not seen.is_file():
            raise FileNotFoundError(f"{data_dir} requires the paired embodiment IDs in {seen}")
        ids = read_embodiment(seen)
        if len(episodes) != len(ids) or any(episode_number(path) != i for i, path in enumerate(episodes)):
            raise ValueError(f"{config_dir} episodes must match the indexed rows in {seen}")
        found.append((task, config, project, episodes, ids))
    if not found:
        raise FileNotFoundError(f"No EmbodiBench episodes for {collection_config or 'any collection'} under {data_root}")
    return found


def require_native(path: Path, handle: h5py.File) -> None:
    if "joint_action" in handle:
        raise ValueError(f"{path} still uses joint_action; rewrite the hdf5 keys first")
    if "state/left_arm_joint_states" not in handle or "vision/cam_head/colors" not in handle:
        raise ValueError(f"{path} is not in the shared state/vision hdf5 layout")


def arm_columns(handle: h5py.File, group: str, side: str):
    arm_key = f"{group}/{side}_arm_joint_states"
    grip_key = f"{group}/{side}_ee_joint_states"
    if arm_key not in handle:
        return None
    if grip_key not in handle:
        raise ValueError(f"{side} arm is present without a gripper under {group}")
    arm = np.asarray(handle[arm_key], dtype=np.float32)
    grip = np.asarray(handle[grip_key], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]
    if grip.shape[1] != 1 or arm.shape[0] != grip.shape[0]:
        raise ValueError(f"bad {group}/{side} shapes {arm.shape} {grip.shape}")
    return arm, grip


def pack(arm: np.ndarray, grip: np.ndarray, width: int) -> np.ndarray:
    rows = np.zeros((arm.shape[0], width + 1), dtype=np.float32)
    count = arm.shape[1]
    if count > width:
        raise ValueError(f"arm width {count} exceeds max joints {width}")
    rows[:, :count] = arm
    rows[:, width] = grip[:, 0]
    return rows


def side_features(urdf_path: Path, joints: int, prepare, width: int):
    names = [f"joint_{index}" for index in range(joints)]
    chain = prepare(urdf_path, names, embed_arm_base=False)
    features = np.asarray(chain["features"], dtype=np.float32)
    if features.shape != (joints, 6):
        raise ValueError(f"unexpected URDF feature shape {features.shape} for {urdf_path}")
    padded = np.zeros((width, 6), dtype=np.float32)
    mask = np.zeros(width, dtype=np.float32)
    padded[:joints] = features
    mask[:joints] = 1
    return padded, mask


def instructions_of(handle: h5py.File, path: Path) -> str:
    if "instructions" not in handle:
        raise ValueError(f"{path} has no instructions")
    raw = handle["instructions"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = json.loads(str(raw))
    if isinstance(parsed, str):
        parsed = [parsed]
    if not parsed or not str(parsed[0]).strip():
        raise ValueError(f"{path} has an empty instruction")
    return str(parsed[0])


def create_dataset(root, features, fps, writer_processes, writer_threads):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    old_home = os.environ.get("HF_LEROBOT_HOME")
    os.environ["HF_LEROBOT_HOME"] = str(root.parent)
    kwargs = dict(
        repo_id=root.name,
        root=root,
        fps=fps,
        robot_type="robotwin",
        features=features,
        use_videos=True,
        image_writer_processes=writer_processes,
        image_writer_threads=writer_threads,
    )
    try:
        try:
            dataset = LeRobotDataset.create(**kwargs, streaming_encoding=True, tolerance_s=0.0001, encoder_threads=4)
        except TypeError:
            dataset = LeRobotDataset.create(**kwargs)
    finally:
        if old_home is None:
            os.environ.pop("HF_LEROBOT_HOME", None)
        else:
            os.environ["HF_LEROBOT_HOME"] = old_home
    actual = Path(getattr(dataset, "root", root)).resolve()
    if actual != root.resolve():
        raise RuntimeError(f"LeRobot created {actual}, expected {root}")
    return dataset


def convert(output_dir, *, input_dir=None, collection_config=None, max_joints=7, fps=15, writer_processes=4, writer_threads=2, overwrite=False, action_type="joint"):
    action_type = str(action_type).lower()
    if action_type != "joint":
        raise ValueError("action_type must be joint")
    from robomorpher.utils.data.process_data.decode_image_bit import decode_image_bit
    from robomorpher.utils.data.urdf_features import prepare_tokenizer_chain

    data_root = Path(input_dir).expanduser().absolute() if input_dir else embodibench_data()
    # A collection may link its episode data while keeping its own metadata.
    if data_root.name != "data":
        data_root = data_root.resolve()
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    rows = []
    for task, config, project, episodes, ids in collections(data_root, collection_config):
        urdf_root = embodibench_urdf(project)
        for episode, robot_ids in zip(episodes, ids):
            rows.append((task, episode, robot_ids, config, urdf_root))
    if not rows:
        raise ValueError("no episodes selected")

    with h5py.File(rows[0][1], "r") as handle:
        require_native(rows[0][1], handle)
        image = decode_image_bit(handle["vision/cam_head/colors"][0])
    image_shape = image.shape
    width = 2 * (max_joints + 1)
    names = [
        *[f"left_joint_{index}" for index in range(max_joints)],
        "left_gripper",
        *[f"right_joint_{index}" for index in range(max_joints)],
        "right_gripper",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (width,), "names": [names]},
        "action": {"dtype": "float32", "shape": (width,), "names": [names]},
        "observation.urdf_features": {
            "dtype": "float32",
            "shape": (2 * max_joints * 6,),
            "names": [[f"urdf_{index}" for index in range(2 * max_joints * 6)]],
        },
        "observation.joint_valid_mask": {
            "dtype": "float32",
            "shape": (2 * max_joints,),
            "names": [[f"joint_valid_{index}" for index in range(2 * max_joints)]],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, image_shape[0], image_shape[1]),
            "names": ["channels", "height", "width"],
        }
    dataset = create_dataset(output_dir, features, fps, writer_processes, writer_threads)
    cache = {}
    sources = []
    total = 0
    try:
        for index, (task, path, (left_id, right_id), config, urdf_root) in enumerate(rows):
            with h5py.File(path, "r") as handle:
                require_native(path, handle)
                left_state = arm_columns(handle, "state", "left")
                left_action = arm_columns(handle, "action", "left")
                right_state = arm_columns(handle, "state", "right")
                right_action = arm_columns(handle, "action", "right")
                if left_state is None or left_action is None:
                    raise ValueError(f"{path} is missing the left arm")
                length = left_state[0].shape[0]
                state = np.concatenate(
                    [
                        pack(left_state[0], left_state[1], max_joints),
                        pack(*right_state, max_joints) if right_state is not None else np.zeros((length, max_joints + 1), np.float32),
                    ],
                    axis=1,
                )
                action = np.concatenate(
                    [
                        pack(left_action[0], left_action[1], max_joints),
                        pack(*right_action, max_joints) if right_action is not None else np.zeros((length, max_joints + 1), np.float32),
                    ],
                    axis=1,
                )
                if state.shape[0] != action.shape[0] or state.shape[0] < 1:
                    raise ValueError(f"bad state/action length in {path}")
                length = int(state.shape[0])
                urdf_parts = []
                mask_parts = []
                for robot_id, arm, present in (
                    (left_id, left_state[0], True),
                    (right_id, None if right_state is None else right_state[0], right_state is not None),
                ):
                    if not present:
                        urdf_parts.append(np.zeros((max_joints, 6), np.float32))
                        mask_parts.append(np.zeros(max_joints, np.float32))
                        continue
                    urdf_path = urdf_root / f"robo{robot_id}" / "single_arm.urdf"
                    if not urdf_path.is_file():
                        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
                    key = (str(urdf_path), int(arm.shape[1]))
                    if key not in cache:
                        cache[key] = side_features(urdf_path, int(arm.shape[1]), prepare_tokenizer_chain, max_joints)
                    urdf_parts.append(cache[key][0])
                    mask_parts.append(cache[key][1])
                urdf = np.concatenate(urdf_parts).reshape(-1)
                mask = np.concatenate(mask_parts)
                instruction = instructions_of(handle, path)
                for frame_index in range(length):
                    frame = {
                        "observation.state": state[frame_index],
                        "action": action[frame_index],
                        "task": instruction,
                    }
                    frame["observation.urdf_features"] = urdf
                    frame["observation.joint_valid_mask"] = mask
                    decoded = {}
                    for camera, native in _NATIVE_CAMERAS.items():
                        key = f"vision/{native}/colors"
                        if key not in handle:
                            continue
                        image = np.asarray(decode_image_bit(handle[key][frame_index]))
                        if image.shape != image_shape:
                            raise ValueError(f"image shape changed in {path}: {image.shape}")
                        decoded[camera] = image
                    if "cam_right_wrist" not in decoded:
                        if "cam_left_wrist" not in decoded:
                            raise ValueError(f"{path} has no wrist camera")
                        decoded["cam_right_wrist"] = decoded["cam_left_wrist"]
                    for camera in CAMERAS:
                        if camera not in decoded:
                            raise ValueError(f"{path} is missing {camera}")
                        frame[f"observation.images.{camera}"] = decoded[camera]
                    dataset.add_frame(frame)
                    total += 1
                dataset.save_episode()
            sources.append(
                {
                    "task": task,
                    "episode": episode_number(path),
                    "collection": config,
                    "urdf_project": urdf_root.name,
                    "source_config": config,
                    "project_name": urdf_root.name,
                    "frames": int(length),
                    "left_robot": int(left_id),
                    "right_robot": int(right_id) if right_state is not None else None,
                    "left_dim": int(left_state[0].shape[1]),
                    "right_dim": None if right_state is None else int(right_state[0].shape[1]),
                    "instruction": instruction,
                }
            )
            print(f"written {index + 1}/{len(rows)} task={task} config={config} episode={episode_number(path)} frames={length}", flush=True)
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        dataset.finalize()

    slices = {
            "left_arm": {"start": 0, "end": max_joints, "dtype": "float32", "absolute": True},
            "left_gripper": {"start": max_joints, "end": max_joints + 1, "dtype": "float32", "absolute": True},
            "right_arm": {"start": max_joints + 1, "end": 2 * max_joints + 1, "dtype": "float32", "absolute": True},
            "right_gripper": {"start": 2 * max_joints + 1, "end": width, "dtype": "float32", "absolute": True},
    }
    meta = output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": slices,
                "action": slices,
                "video": {camera: {"original_key": f"observation.images.{camera}"} for camera in CAMERAS},
                "annotation": {"human.task_description": {"original_key": "task"}},
            },
            indent=2,
        )
        + "\n"
    )
    source_configs = sorted({source["source_config"] for source in sources})
    (meta / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "embodibench.lerobot.v1",
                "action_type": action_type,
                "source_format": "EmbodiBench HDF5",
                "source_configs": source_configs,
                "source_config": source_configs[0] if len(source_configs) == 1 else None,
                "morphology_projects": sorted({source["project_name"] for source in sources}),
                "state_action_alignment": "native state/action; no second temporal shift",
                "max_num_joints": max_joints,
                "state_dim": width,
                "frames": total,
                "episodes": len(sources),
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )
    write_masked_action_stats(output_dir, action_type=action_type)
    print(f"completed {total} frames / {len(sources)} episodes -> {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-config", default=os.environ.get("ROBOMORPHER_COLLECTION_CONFIG"),
                        help="Select a collection name across tasks (default: all collections under the input directory)")
    parser.add_argument("--input-dir", help="Directory containing EmbodiBench task collections")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-joints", type=int, default=7)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--writer-processes", type=int, default=4)
    parser.add_argument("--writer-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--action-type", choices=("joint",), default="joint")
    args = parser.parse_args()
    convert(
        args.output_dir,
        input_dir=args.input_dir,
        collection_config=args.collection_config,
        max_joints=args.max_joints,
        fps=args.fps,
        writer_processes=args.writer_processes,
        writer_threads=args.writer_threads,
        overwrite=args.overwrite,
        action_type=args.action_type,
    )


if __name__ == "__main__":
    main()
