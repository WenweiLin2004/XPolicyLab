r"""Stream RoboTwin trajectories into the tokenizer's single-frame zarr layout.

Each selected source episode, arm, and frame is written once. There are no
sliding action windows. The output uses small zarr chunks and an ``index.json``
per embodiment.

Example:
    python -m robomorpher.utils.data.process_data.convert_robotwin_hdf5_to_zarr \
        --input-dir /path/to/robotwin_stack_bowls_two_demo \
        --output-dir /path/to/tokenizer_zarr

An existing output directory is always rejected, including an empty directory.
If conversion fails, partial output is retained with an incomplete manifest;
use a new destination after diagnosing the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr
from numcodecs import Blosc


ROBOMORPHER_ROOT = Path(__file__).resolve().parents[4]
if str(ROBOMORPHER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOMORPHER_ROOT))
from robomorpher.utils.data.process_data.bench_paths import robotwin_assets, stored_path

CONVERTER_SCHEMA = "eat.robotwin.single_frame.v1"
EMBODIMENT_FOLDERS = {
    "stack_bowls_two_piper": {
        "urdf": "piper/piper.urdf",
        "left_key": "piper",
        "right_key": "piper",
    },
    "stack_bowls_two_ur5": {
        "urdf": "ur5-wsg/ur5_wsg_gripper.urdf",
        "left_key": "ur5-wsg",
        "right_key": "ur5-wsg",
    },
    "stack_bowls_two_franka": {
        "urdf": "franka-panda/panda.urdf",
        "left_key": "franka-panda",
        "right_key": "franka-panda",
    },
    "stack_bowls_two_aloha": {
        "urdf": "aloha-agilex/urdf/arx5_description_isaac.urdf",
        "left_key": "aloha-agilex-left",
        "right_key": "aloha-agilex-right",
    },
    "stack_bowls_two_arx-x5": {
        "urdf": "ARX-X5/X5A.urdf",
        "left_key": "arx-x5",
        "right_key": "arx-x5",
    },
}


def _kinematics_api():
    # Import kinematics only when converting, so --help stays independent of it.
    from robomorpher.utils.data.urdf_features import (
        EMBODIMENT_ARM_JOINTS,
        build_tokenizer_frame,
        prepare_tokenizer_chain,
    )

    return EMBODIMENT_ARM_JOINTS, prepare_tokenizer_chain, build_tokenizer_frame


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _episode_sort_key(path: Path):
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def _discover_episodes(input_dir: Path, embodiments: list[str] | None):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    requested = list(EMBODIMENT_FOLDERS) if embodiments is None else list(embodiments)
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate embodiment names would duplicate source frames")
    unknown = set(requested) - EMBODIMENT_FOLDERS.keys()
    if unknown:
        raise ValueError(f"Unknown embodiments: {sorted(unknown)}")
    episodes: dict[str, list[Path]] = {}
    for name in requested:
        folder = input_dir / name / "demo_clean" / "data"
        files = sorted(folder.glob("episode*.hdf5"), key=_episode_sort_key)
        if not files:
            if embodiments is not None:
                raise FileNotFoundError(f"No episode*.hdf5 files found in {folder}")
            continue
        episodes[name] = files
    if not episodes:
        raise FileNotFoundError(f"No supported RoboTwin episodes found under {input_dir}")
    return episodes


def _inspect_trajectory(dataset, num_joints: int, rows_per_chunk: int, description: str):
    """Validate all source q values and measure motion without loading the trajectory."""
    if len(dataset.shape) != 2 or dataset.shape[1] != num_joints:
        raise ValueError(
            f"{description}: expected shape (T, {num_joints}) in preset joint-name order, "
            f"got {dataset.shape}; joint names are never truncated or inferred"
        )
    if not np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.complexfloating):
        raise ValueError(f"{description}: q must be real numeric values, got {dataset.dtype}")
    low = np.full(num_joints, np.inf, dtype=np.float64)
    high = np.full(num_joints, -np.inf, dtype=np.float64)
    for start in range(0, dataset.shape[0], rows_per_chunk):
        values = np.asarray(dataset[start : start + rows_per_chunk], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{description}: non-finite q in source frames {start}:{start + len(values)}")
        low = np.minimum(low, values.min(axis=0))
        high = np.maximum(high, values.max(axis=0))
    max_range = float(np.max(high - low)) if dataset.shape[0] else 0.0
    return int(dataset.shape[0]), max_range


def _checked_frame(frame, num_joints: int, max_joints: int):
    shapes = {"input": (max_joints, 7), "output": (max_joints, 10), "joint_valid_mask": (max_joints,)}
    result = {}
    for key, shape in shapes.items():
        if key not in frame:
            raise ValueError(f"build_tokenizer_frame did not return {key}")
        value = np.asarray(frame[key])
        if value.shape != shape:
            raise ValueError(f"build_tokenizer_frame returned {key} shape {value.shape}, expected {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"build_tokenizer_frame returned non-finite {key}")
        dtype = bool if key == "joint_valid_mask" else np.float32
        value = value.astype(dtype)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"build_tokenizer_frame {key} overflows its output dtype")
        result[key] = value
    expected_mask = np.arange(max_joints) < num_joints
    if not np.array_equal(result["joint_valid_mask"], expected_mask):
        raise ValueError("build_tokenizer_frame returned a mask inconsistent with joint_names")
    return result


def _write_arm_shard(
    path: Path,
    trajectory,
    chain: dict,
    build_frame,
    provenance: dict,
    *,
    max_joints: int,
    rows_per_chunk: int,
    frame_stride: int,
) -> int:
    num_frames = int(trajectory.shape[0])
    num_rows = (num_frames + frame_stride - 1) // frame_stride
    chunk_rows = min(rows_per_chunk, num_rows)
    group = zarr.open_group(str(path), mode="w-")
    group.attrs.update(provenance)
    data = group.create_group("data")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    arrays = {}
    for key, tail, dtype in (
        ("input", (max_joints, 7), "float32"),
        ("output", (max_joints, 10), "float32"),
        ("joint_valid_mask", (max_joints,), "bool"),
    ):
        arrays[key] = data.create_dataset(
            key,
            shape=(num_rows,) + tail,
            chunks=(chunk_rows,) + tail,
            dtype=dtype,
            compressor=compressor,
        )
    source_indices = group.create_dataset(
        "metadata/source_frame_index",
        shape=(num_rows,),
        chunks=(chunk_rows,),
        dtype="int64",
        compressor=compressor,
    )
    for row_start in range(0, num_rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, num_rows)
        frame_start = row_start * frame_stride
        frame_stop = min(row_end * frame_stride, num_frames)
        angles = np.asarray(trajectory[frame_start:frame_stop:frame_stride], dtype=np.float64)
        buffers = {
            "input": np.empty((len(angles), max_joints, 7), dtype=np.float32),
            "output": np.empty((len(angles), max_joints, 10), dtype=np.float32),
            "joint_valid_mask": np.empty((len(angles), max_joints), dtype=bool),
        }
        for offset, q in enumerate(angles):
            frame = _checked_frame(
                build_frame(chain, q, max_joints=max_joints), len(chain["joint_names"]), max_joints
            )
            for key, value in frame.items():
                buffers[key][offset] = value
        for key, buffer in buffers.items():
            arrays[key][row_start:row_end] = buffer
        source_indices[row_start:row_end] = np.arange(frame_start, frame_stop, frame_stride, dtype=np.int64)
    group.attrs["complete"] = True
    return num_rows


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    assets_dir: str | Path | None = None,
    embodiments: list[str] | None = None,
    max_joints: int = 8,
    rows_per_chunk: int = 4096,
    frame_stride: int = 1,
    active_arms_only: bool = False,
    active_threshold: float = 1e-6,
) -> dict:
    """Create a fresh single-frame dataset; retain partial results on failure."""
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (max_joints, rows_per_chunk, frame_stride)):
        raise ValueError("max_joints, rows_per_chunk, and frame_stride must be positive integers")
    if not math.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    if str(zarr.__version__).split(".")[0] != "2":
        raise RuntimeError(f"This converter requires zarr 2.x, found {zarr.__version__}")
    input_dir = Path(input_dir).expanduser().resolve()
    requested_output = Path(output_dir).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {requested_output}")
    output_dir = requested_output.resolve()
    assets_dir = Path(assets_dir or robotwin_assets()).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    episodes = _discover_episodes(input_dir, embodiments)
    presets, prepare_chain, build_frame = _kinematics_api()
    chains = {}
    urdf_hashes = {}
    for name in episodes:
        config = EMBODIMENT_FOLDERS[name]
        urdf_path = assets_dir / config["urdf"]
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
        urdf_hashes[str(urdf_path)] = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
        for side in ("left", "right"):
            joint_names = list(presets[config[f"{side}_key"]])
            if not joint_names or len(joint_names) > max_joints:
                raise ValueError(f"{name}/{side}: {len(joint_names)} joints do not fit max_joints={max_joints}")
            cache_key = (str(urdf_path), tuple(joint_names))
            if cache_key not in chains:
                chain = prepare_chain(str(urdf_path), joint_names)
                if list(chain["joint_names"]) != joint_names:
                    raise ValueError("prepare_tokenizer_chain changed the requested joint-name order")
                features = np.asarray(chain["features"])
                if features.shape != (len(joint_names), 6) or not np.all(np.isfinite(features)):
                    raise ValueError(f"{name}/{side}: invalid xyz+rpy features")
                if not chain.get("schema_version") or not chain.get("base_link"):
                    raise ValueError("prepare_tokenizer_chain must provide schema_version and base_link")
                chains[cache_key] = chain

    # Creation is exclusive even if another process races the existence check.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": CONVERTER_SCHEMA,
        "status": "incomplete",
        "input_dir": stored_path(input_dir),
        "assets_dir": str(assets_dir),
        "sample_unit": "one source episode / arm / frame",
        "action_chunk": 1,
        "frame_stride": frame_stride,
        "max_joints": max_joints,
        "rows_per_chunk": rows_per_chunk,
        "active_arms_only": active_arms_only,
        "active_threshold_rad": active_threshold,
        "active_rule": "max over joints of (max q over time - min q over time) > threshold",
        "embodiments": list(episodes),
        "shards": [],
        "skipped_arms": [],
        "total_rows": 0,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest, exclusive=True)
    try:
        for name, files in episodes.items():
            destination = output_dir / name
            destination.mkdir()
            config = EMBODIMENT_FOLDERS[name]
            urdf_path = assets_dir / config["urdf"]
            index = {
                "format_version": 1,
                "schema_version": CONVERTER_SCHEMA,
                "src": stored_path(input_dir / name),
                "rows_per_chunk": rows_per_chunk,
                "keys": {"input": [max_joints, 7], "output": [max_joints, 10], "joint_valid_mask": [max_joints]},
                "shards": [],
                "total_rows": 0,
            }
            for source_path in files:
                with h5py.File(source_path, "r") as episode:
                    for side in ("left", "right"):
                        joint_names = list(presets[config[f"{side}_key"]])
                        chain = chains[(str(urdf_path), tuple(joint_names))]
                        source_key = f"joint_action/{side}_arm"
                        description = f"{source_path}:{source_key}"
                        if source_key not in episode:
                            raise KeyError(f"Missing arm trajectory: {description}")
                        trajectory = episode[source_key]
                        num_frames, max_range = _inspect_trajectory(trajectory, len(joint_names), rows_per_chunk, description)
                        source = {"source_hdf5": stored_path(source_path), "source_dataset": source_key, "arm": side, "source_frames": num_frames}
                        if not num_frames or (active_arms_only and max_range <= active_threshold):
                            manifest["skipped_arms"].append({
                                **source,
                                "reason": "empty_trajectory" if not num_frames else "explicit_active_arms_only_filter",
                                "max_joint_range_rad": max_range,
                            })
                            continue
                        shard_name = f"data_chunk_{len(index['shards']):06d}.zarr"
                        provenance = {
                            **source,
                            "schema_version": CONVERTER_SCHEMA,
                            "kinematics_schema_version": chain["schema_version"],
                            "joint_names": joint_names,
                            "base_link": chain["base_link"],
                            "urdf_path": stored_path(urdf_path),
                            "urdf_sha256": urdf_hashes[str(urdf_path)],
                            "joint_frame_corrections": np.asarray(chain.get("joint_frame_corrections", [])).tolist(),
                            "folded_fixed_paths": chain.get("fixed_paths", []),
                            "origin_xyz_rpy": np.asarray(chain["features"]).tolist(),
                            "pose_convention": "arm-base-relative canonical active child frame; not tool TCP",
                            "q_convention": "source radians in declared order, no wrapping/sign/zero change by converter",
                            "frame_stride": frame_stride,
                            "source_frame_indices": "metadata/source_frame_index",
                            "angle_unit": "radian",
                            "position_unit": "meter",
                            "complete": False,
                        }
                        rows = _write_arm_shard(
                            destination / shard_name, trajectory, chain, build_frame, provenance,
                            max_joints=max_joints, rows_per_chunk=rows_per_chunk, frame_stride=frame_stride,
                        )
                        index["shards"].append({"path": shard_name, "rows": rows})
                        index["total_rows"] += rows
                        manifest["shards"].append({**provenance, "complete": True, "path": f"{name}/{shard_name}", "rows": rows})
                        manifest["total_rows"] += rows
                        _write_json(manifest_path, manifest)
                        print(f"{name}/{source_path.name}/{side}: {rows} single-frame samples", flush=True)
            # An empty embodiment gets an explicit zero-row index, never a fake sample.
            _write_json(destination / "index.json", index, exclusive=True)
        manifest["status"] = "complete"
        _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New destination; existing directories are rejected")
    parser.add_argument("--assets-dir", default=None, help="Defaults to RoboTwin/assets/embodiments")
    parser.add_argument("--embodiments", nargs="+", choices=sorted(EMBODIMENT_FOLDERS))
    parser.add_argument("--max-joints", type=int, default=8)
    parser.add_argument("--rows-per-chunk", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=1, help="Explicit temporal subsampling; default preserves every frame")
    parser.add_argument("--active-arms-only", action="store_true", help="Explicitly omit stationary arms; by default both arms are kept")
    parser.add_argument("--active-threshold", type=float, default=1e-6, help="Minimum per-joint temporal range in radians for active-only mode")
    args = parser.parse_args()
    result = convert(**vars(args))
    print(f"Completed: {result['total_rows']} single-frame samples in {args.output_dir}")


if __name__ == "__main__":
    main()
