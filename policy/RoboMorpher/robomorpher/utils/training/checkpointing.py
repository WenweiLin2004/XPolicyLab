"""Atomic checkpoint writes and manifest-owned retention shared by both trainers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import torch


def resolve_max_checkpoints(config, default=8, *, allow_unlimited=False):
    """Accept max_checkpoints, or max_num_ckpt when that key is absent."""
    current = config.get("max_checkpoints")
    previous = config.get("max_num_ckpt")
    if current is not None and previous is not None and current != previous:
        raise ValueError("Conflicting max_checkpoints and max_num_ckpt")
    value = current if current is not None else (previous if previous is not None else default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_checkpoints must be an integer")
    if not allow_unlimited and value < 1:
        raise ValueError("max_checkpoints must be >= 1")
    return value


def atomic_torch_save(payload, path):
    """Serialize beside the destination, then replace; failure preserves old bytes."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing checkpoint output through a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _contract_digest(contract):
    canonical = json.loads(json.dumps(contract, sort_keys=True, default=str))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _axis(axis):
    if axis not in {"step", "epoch"}:
        raise ValueError("Checkpoint axis must be step or epoch")
    return axis


def checkpoint_step(name, axis="step"):
    """Step counter from steps_<N>_pytorch_model.pt, or an epoch counter from epoch_<N>.pt."""
    base = Path(name).name
    if axis == "step":
        match = re.fullmatch(r"steps_([0-9]+)_pytorch_model\.pt", base)
    else:
        match = re.fullmatch(rf"{axis}_([0-9]+)\.pt", base)
    if match is None:
        raise ValueError("Invalid owned checkpoint filename")
    return int(match.group(1))


def owned_checkpoint_path(output, name, *, axis="step"):
    axis = _axis(axis)
    output = Path(output).resolve()
    if not isinstance(name, str) or name.startswith(("/", "\\")) or ".." in Path(name).parts:
        raise ValueError("Invalid owned checkpoint filename")
    rel = Path(name)
    checkpoint_step(name, axis)
    if axis == "step":
        if rel.parts != ("checkpoints", rel.name):
            raise ValueError("Invalid owned checkpoint filename")
    elif len(rel.parts) != 1:
        raise ValueError("Invalid owned checkpoint filename")
    path = (output / rel).resolve()
    if path.is_symlink() or output not in path.parents:
        raise ValueError("Checkpoint retention target escapes output directory")
    return path


def _managed_checkpoint_names(output, axis):
    output = Path(output)
    names = []
    if not output.is_dir():
        return names
    if axis == "step":
        weight_dir = output / "checkpoints"
        if weight_dir.is_dir():
            for path in weight_dir.iterdir():
                if re.fullmatch(r"steps_[0-9]+_pytorch_model\.pt", path.name):
                    names.append(f"checkpoints/{path.name}")
        return names
    for path in output.iterdir():
        if re.fullmatch(rf"{axis}_[0-9]+\.pt", path.name):
            names.append(path.name)
    return names


def policy_meta_path(weight_path):
    """Small non-weight record beside steps_<N>_pytorch_model.pt."""
    weight_path = Path(weight_path)
    match = re.fullmatch(r"steps_([0-9]+)_pytorch_model\.pt", weight_path.name)
    if match is None:
        raise ValueError("Policy weight file must be named steps_<N>_pytorch_model.pt")
    return weight_path.with_name(f"steps_{match.group(1)}_meta.pt")


def load_weight_checkpoint(path, *, mmap=True):
    """Load a model state_dict and its metadata. Optimizer is not part of either file."""
    path = Path(path)
    if re.fullmatch(r"steps_[0-9]+_pytorch_model\.pt", path.name) is None:
        raise ValueError("Policy checkpoint must be named steps_<N>_pytorch_model.pt")
    weights = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    if not isinstance(weights, dict) or any(not torch.is_tensor(value) for value in weights.values()):
        raise ValueError("Policy checkpoint must contain only model weights")
    meta_path = policy_meta_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    return weights, meta


def remove_optimizer_sidecar(weight_path):
    """Drop metadata belonging to one pruned weight file."""
    weight_path = Path(weight_path)
    match = re.fullmatch(r"steps_([0-9]+)_pytorch_model\.pt", weight_path.name)
    if match is None:
        return
    for name in (f"steps_{match.group(1)}_optimizer.pt", f"steps_{match.group(1)}_meta.pt"):
        sidecar = weight_path.with_name(name)
        if sidecar.is_symlink():
            raise ValueError("Refusing to prune a checkpoint sidecar through a symlink")
        if sidecar.is_file():
            sidecar.unlink()


def read_checkpoint_manifest(output, contract, *, axis="step"):
    axis = _axis(axis)
    output = Path(output)
    path = output / "checkpoint_manifest.json"
    if path.is_symlink():
        raise ValueError("Checkpoint manifest must not be a symlink")
    if not path.exists():
        return {"version": 1, "contract_sha256": _contract_digest(contract), "checkpoints": []}
    manifest = json.loads(path.read_text())
    if manifest.get("contract_sha256") != _contract_digest(contract):
        raise ValueError("Checkpoint manifest contract differs from this run")
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("Invalid checkpoint ownership manifest")
    seen = set()
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("Invalid checkpoint ownership manifest")
        name = entry.get("name")
        owned_checkpoint_path(output, name, axis=axis)
        number = checkpoint_step(name, axis)
        if name in seen or entry.get(axis) != number:
            raise ValueError("Invalid checkpoint ownership counter or duplicate")
        seen.add(name)
    if axis == "step" and manifest.get("last_checkpoint") is not None:
        raise ValueError("Policy checkpoints do not use last.pt")
    return manifest


def write_checkpoint_manifest(output, manifest):
    output = Path(output)
    path = output / "checkpoint_manifest.json"
    if path.is_symlink():
        raise ValueError("Checkpoint manifest must not be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint_manifest.", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_output_directory(output, resume, contract, *, axis="step", resume_counter=None):
    """Only rank zero calls; never adopt unmanaged checkpoints as owned."""
    axis = _axis(axis)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = _managed_checkpoint_names(output, axis)
    if existing and not resume:
        raise ValueError("Output directory already contains checkpoints; choose a new output_dir or resume")
    if existing and resume and not (output / "checkpoint_manifest.json").is_file():
        raise ValueError("Existing checkpoint directory has no ownership manifest; resume into a new output_dir")
    manifest = read_checkpoint_manifest(output, contract, axis=axis)
    if existing and resume and resume_counter is not None:
        resume_dir = Path(resume).expanduser().resolve().parent
        if resume_dir != output and resume_dir != (output / "checkpoints"):
            raise ValueError("Resume checkpoint belongs to another directory; choose a new output_dir")
        counters = [entry[axis] for entry in manifest["checkpoints"]]
        if counters and max(counters) > int(resume_counter):
            raise ValueError("Output directory contains newer checkpoints; resume the latest or choose a new output_dir")
    write_checkpoint_manifest(output, manifest)


def validate_checkpoint_destination(output, path, contract, *, axis="step"):
    output, path = Path(output).resolve(), Path(path)
    if path.is_symlink():
        raise ValueError("Refusing checkpoint output outside its directory or through a symlink")
    try:
        rel = path.resolve().relative_to(output).as_posix()
    except ValueError as exc:
        raise ValueError("Refusing checkpoint output outside its directory or through a symlink") from exc
    manifest = read_checkpoint_manifest(output, contract, axis=axis)
    owned_checkpoint_path(output, rel, axis=axis)
    owned = {entry["name"] for entry in manifest["checkpoints"]}
    if path.exists() and rel not in owned:
        raise ValueError("Refusing to overwrite an unmanaged checkpoint")


def register_and_prune_checkpoint(output, path, contract, max_checkpoints, *, axis="step", allow_unlimited=False, checkpoint_counter=None):
    """Only this manifest's epoch or steps_<N>_pytorch_model.pt files can be removed."""
    del checkpoint_counter
    if not allow_unlimited and max_checkpoints < 1:
        raise ValueError("max_checkpoints must be >= 1")
    axis = _axis(axis)
    output, path = Path(output).resolve(), Path(path)
    try:
        rel = path.resolve().relative_to(output).as_posix()
    except ValueError as exc:
        raise ValueError("Cannot register a missing/outside checkpoint") from exc
    target = owned_checkpoint_path(output, rel, axis=axis)
    if target.resolve() != path.resolve() or not target.is_file():
        raise ValueError("Cannot register a missing/outside checkpoint")
    manifest = read_checkpoint_manifest(output, contract, axis=axis)
    records = {entry["name"]: entry for entry in manifest["checkpoints"]}
    number = checkpoint_step(rel, axis)
    records[rel] = {"name": rel, axis: number}
    ordered = sorted(records.values(), key=lambda entry: entry[axis])
    manifest["checkpoints"] = ordered
    write_checkpoint_manifest(output, manifest)
    if allow_unlimited and max_checkpoints <= 0:
        return
    for entry in ordered[:-max_checkpoints]:
        owned = owned_checkpoint_path(output, entry["name"], axis=axis)
        if owned.exists():
            if not owned.is_file():
                raise ValueError("Refusing to prune a checkpoint that is not a regular file")
            owned.unlink()
            remove_optimizer_sidecar(owned)
    manifest["checkpoints"] = ordered[-max_checkpoints:]
    write_checkpoint_manifest(output, manifest)


def assert_no_pending_ema(model):
    for name, module in model.named_modules():
        if getattr(module, "has_pending_ema", False):
            raise RuntimeError(f"Cannot save checkpoint with pending EMA in {name}; complete accumulation first")
