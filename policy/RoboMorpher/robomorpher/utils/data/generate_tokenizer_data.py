"""Generate single-frame RoboMorpher training data from MorphGen revolute arms."""

import argparse
import ctypes
import errno
import gzip
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache, partial
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import zarr
from numcodecs import Blosc

from .urdf_features import SCHEMA_VERSION, build_tokenizer_frame, prepare_tokenizer_chain


JOINT_SLOTS = 7
FIELD_LAYOUT = {
    "input": ((JOINT_SLOTS, 7), np.float32),
    "output": ((JOINT_SLOTS, 10), np.float32),
    "joint_valid_mask": ((JOINT_SLOTS,), np.bool_),
}
MORPHGEN_FILES = ("single_urdf_generator.py", "random_augment_without_collision.py")


@lru_cache(maxsize=1)
def load_morphgen(root):
    # Both projects have a top-level 'utils' package. Load these standalone
    # upstream modules by file path without shadowing RoboMorpher's imports.
    modules = []
    for name in MORPHGEN_FILES:
        path = root / "utils" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Set --morphgen-root or MORPHGEN_ROOT to a MorphGen checkout.")
        spec = importlib.util.spec_from_file_location(f"_morphgen_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[0].SingleUrdfGenerator, modules[1].update_urdf_joint_and_visuals


def generate_arm(args, robot_id):
    generator, augment = load_morphgen(args.morphgen_root)
    morphology, augmentation, states = np.random.SeedSequence([args.seed, robot_id]).spawn(3)
    py_seed, np_seed = morphology.spawn(2)
    random.seed(int.from_bytes(py_seed.generate_state(4).tobytes(), "little"))
    np.random.seed(np_seed.generate_state(4))  # MorphGen also calls scipy.stats via NumPy's legacy RNG.
    result = generator().generate(min_joints=args.min_joints, max_joints=args.max_joints,
                                  need_gripper=False, curobo_config=False, robotwin_config=False)
    names = list(result["joints"])
    if not args.min_joints <= len(names) <= args.max_joints:
        raise ValueError(f"Robot {robot_id}: unexpected generated joint count {len(names)}")
    # The upstream transformation accepts ElementTree.parse-compatible input.
    # Supply its offsets explicitly, preserving an independent RNG for state q.
    offsets = np.random.default_rng(augmentation).uniform(-np.pi, np.pi, len(names))
    arm = augment(io.StringIO(result["urdf"]), joints_pos=offsets).getroot()
    chain = prepare_tokenizer_chain(arm, names)
    joint_by_name = {joint.get("name"): joint for joint in arm.findall("joint")}
    limits = []
    for name in names:
        joint = joint_by_name[name]
        limit = joint.find("limit")
        if joint.get("type") != "revolute" or limit is None:
            raise ValueError(f"Robot {robot_id}: expected bounded revolute joint {name}")
        limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
    low, high = np.asarray(limits, dtype=np.float64).T
    if not np.all(np.isfinite(limits)) or np.any(low >= high):
        raise ValueError(f"Robot {robot_id}: invalid URDF joint limits")
    # Keep float32 storage and the q used for FK identical, even at the limits.
    low32, high32 = low.astype(np.float32), high.astype(np.float32)
    low32 = np.where(low32 < low, np.nextafter(low32, np.float32(np.inf)), low32)
    high32 = np.where(high32 > high, np.nextafter(high32, np.float32(-np.inf)), high32)
    if not np.all(np.isfinite([low32, high32])) or np.any(low32 > high32):
        raise ValueError(f"Robot {robot_id}: joint limits cannot be represented in float32")
    first_row = robot_id * args.states_per_robot
    record = {
        "robot_id": robot_id, "row_start": first_row, "row_stop": first_row + args.states_per_robot,
        "joint_names": names, "base_link": chain["base_link"], "urdf_xml": ET.tostring(arm, encoding="unicode"),
    }
    return chain, record, np.random.default_rng(states), low32, high32


def write_shard(shard_id, *, args):
    first_robot = shard_id * args.robots_per_shard
    stop_robot = min(first_robot + args.robots_per_shard, args.num_robots)
    rows = (stop_robot - first_robot) * args.states_per_robot
    capacity = min(args.rows_per_chunk, rows)
    shard_name = f"data_chunk_{shard_id:06d}.zarr"
    manifest_name = f"robots_{shard_id:06d}.jsonl.gz"
    group = zarr.open_group(str(args.output_dir / shard_name), mode="w-", zarr_version=2)
    data = group.create_group("data")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    arrays, buffers = {}, {}
    for key, (tail, dtype) in FIELD_LAYOUT.items():
        arrays[key] = data.create_dataset(key, shape=(rows, *tail), chunks=(capacity, *tail),
                                          dtype=dtype, compressor=compressor)
        buffers[key] = np.empty((capacity, *tail), dtype=dtype)

    written = pending = 0
    data_hash = hashlib.sha256()
    with gzip.open(args.output_dir / manifest_name, "xt", encoding="utf-8", compresslevel=3) as manifest:
        for robot_id in range(first_robot, stop_robot):
            chain, record, rng, lower, upper = generate_arm(args, robot_id)
            manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
            remaining = args.states_per_robot
            while remaining:
                n = min(remaining, capacity - pending)
                angles = rng.uniform(lower, upper, size=(n, len(lower))).astype(np.float32)
                for i, q in enumerate(angles, start=pending):
                    frame = build_tokenizer_frame(chain, q, max_joints=JOINT_SLOTS)
                    for key, buffer in buffers.items():
                        buffer[i] = frame[key]
                remaining -= n
                pending += n
                if pending == capacity or written + pending == rows:
                    for key, buffer in buffers.items():
                        arrays[key][written:written + pending] = buffer[:pending]
                        data_hash.update(buffer[:pending].tobytes())
                    written += pending
                    pending = 0
    if written != rows:
        raise RuntimeError(f"{shard_name}: wrote {written} of {rows} rows")
    return {"path": shard_name, "rows": rows, "robots": manifest_name, "data_sha256": data_hash.hexdigest()}


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New corpus directory (must not exist)")
    parser.add_argument("--num-robots", type=positive_int, default=40_000_000,
                        help="Number of URDFs to generate (default: %(default)s)")
    parser.add_argument("--states-per-robot", type=positive_int, default=1,
                        help="Number of independently sampled states per URDF (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Nonnegative base seed (default: 42)")
    parser.add_argument("--min-joints", type=int, choices=(6, 7), default=6)
    parser.add_argument("--max-joints", type=int, choices=(6, 7), default=7)
    parser.add_argument("--rows-per-chunk", type=positive_int, default=4096)
    parser.add_argument("--robots-per-shard", type=positive_int, help="Default targets 131072 rows per shard")
    parser.add_argument("--workers", type=positive_int, default=1, help="CPU workers writing separate shards")
    default_root = Path(__file__).resolve().parents[2] / "third_party" / "MorphGen"
    parser.add_argument("--morphgen-root", type=Path, default=os.environ.get("MORPHGEN_ROOT") or default_root,
                        help="MorphGen checkout (default: MORPHGEN_ROOT or robomorpher/third_party/MorphGen)")
    args = parser.parse_args()
    if args.seed < 0 or args.min_joints > args.max_joints:
        parser.error("seed must be nonnegative and min-joints must not exceed max-joints")
    if args.robots_per_shard is None:
        args.robots_per_shard = max(1, 131072 // args.states_per_robot)
    args.morphgen_root = args.morphgen_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().absolute()
    return args


def publish_new_directory(stage_fd, destination):
    # Linux renameat2(RENAME_NOREPLACE) checks the destination at publication
    # time, including an empty directory or a dangling symlink from a racing run.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RuntimeError("This producer requires Linux renameat2 for exclusive corpus publication")
    rc = libc.renameat2(stage_fd, b"corpus", -100, os.fsencode(destination), 1)
    if rc < 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise SystemExit(f"Cannot overwrite existing corpus path: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def new_corpus(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise SystemExit(f"Cannot overwrite existing corpus path: {destination}")
    private = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    handle = os.open(private, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        stage = private / "corpus"
        stage.mkdir()
        yield stage
        if not os.path.samestat(os.fstat(handle), private.lstat()):
            raise RuntimeError("Private staging directory changed while generating the corpus")
        publish_new_directory(handle, destination)
    finally:
        try:
            # Anchor recursive cleanup to our open private directory, including
            # when another process moved it and replaced its former path.
            anchored = Path("/proc/self/fd") / str(handle) / "corpus"
            if anchored.is_dir() and not anchored.is_symlink():
                shutil.rmtree(anchored)
            if os.path.lexists(private) and os.path.samestat(os.fstat(handle), private.lstat()):
                private.rmdir()
        finally:
            os.close(handle)


def main():
    args = parse_args()
    load_morphgen(args.morphgen_root)
    hashes = {name: hashlib.sha256((args.morphgen_root / "utils" / name).read_bytes()).hexdigest()
              for name in MORPHGEN_FILES}
    shard_count = (args.num_robots + args.robots_per_shard - 1) // args.robots_per_shard
    index = {
        "schema_version": "robomorpher.morphgen.single_frame.v1", "kinematics_schema_version": SCHEMA_VERSION,
        "morphgen_source_sha256": hashes, "seed": args.seed, "num_robots": args.num_robots,
        "states_per_robot": args.states_per_robot, "min_joints": args.min_joints,
        "max_joints": args.max_joints, "max_num_joints": JOINT_SLOTS, "rows_per_chunk": args.rows_per_chunk,
        "robots_per_shard": args.robots_per_shard, "total_rows": args.num_robots * args.states_per_robot,
        "sampling": "MorphGen revolute arm; joint-origin offsets uniform in [-pi,pi]; independent q uniform within URDF limits; no collision filtering",
        "frame": "arm-base-relative canonical active-child xyz, rotation columns 1 and 2, raw q in radians",
        "shards": [],
    }
    destination = args.output_dir
    with new_corpus(destination) as args.output_dir:
        worker = partial(write_shard, args=args)
        if args.workers == 1 or shard_count == 1:
            index["shards"] = list(map(worker, range(shard_count)))
        else:
            with mp.get_context("spawn").Pool(min(args.workers, shard_count)) as pool:
                index["shards"] = list(pool.imap(worker, range(shard_count)))
        with (args.output_dir / "index.json").open("x", encoding="utf-8") as out:
            json.dump(index, out, indent=2)
            out.write("\n")
    print(f"Saved {index['total_rows']} frames (robots={args.num_robots}, shards={shard_count}): {destination}")


if __name__ == "__main__":
    main()
