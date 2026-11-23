"""Locate EmbodiBench and RoboTwin inputs from this repository.

Paths are relative to this file. A missing directory is an error.
"""
import os
from pathlib import Path

ROBOMORPHER_ROOT = Path(__file__).resolve().parents[4]


def _anchor(repo: Path) -> Path:
    """Directory that contains EmbodiBench, whether this tree is nested or not."""
    for node in (repo, *repo.parents):
        if (node / "EmbodiBench").is_dir():
            return node
        if node.name == "EmbodiBench":
            return node.parent
    return repo.parent


ANCHOR = _anchor(ROBOMORPHER_ROOT)
EMBODIBENCH_ROOT = Path(os.environ.get("EMBODIBENCH_ROOT", ANCHOR / "EmbodiBench")).expanduser().resolve()
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", ANCHOR / "RoboTwin")).expanduser().resolve()


def stored_path(path: Path) -> str:
    """Path relative to the directory that contains EmbodiBench and RoboTwin."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(ANCHOR))
    except ValueError:
        return str(resolved)


def resolve_stored_path(raw: str | Path) -> Path:
    """Open a stored path. Missing paths are an error."""
    text = str(raw)
    path = Path(text).expanduser()
    if path.exists():
        return path.resolve()
    for marker in ("EmbodiBench/", "RoboTwin/"):
        index = text.find(marker)
        if index >= 0:
            candidate = (ANCHOR / text[index:]).resolve()
            if candidate.exists():
                return candidate
    if not path.is_absolute():
        for base in (ROBOMORPHER_ROOT, EMBODIBENCH_ROOT, ROBOTWIN_ROOT, ANCHOR):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Path does not exist: {raw}")


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def embodibench_data() -> Path:
    return require_dir(EMBODIBENCH_ROOT / "data", "EmbodiBench data")


def embodibench_urdf(project: str) -> Path:
    if not project or Path(project).name != project or project in {".", ".."}:
        raise ValueError(f"Invalid EmbodiBench morphology project: {project!r}")
    return require_dir(
        EMBODIBENCH_ROOT / "MorphGen" / "embodiments" / project,
        f"MorphGen {project} URDF",
    )


def embodibench_task_config(name: str) -> Path:
    filename = f"{name}.yml"
    config = EMBODIBENCH_ROOT / "env_cfg" / "task_config" / filename
    legacy = EMBODIBENCH_ROOT / "task_config" / filename
    for path in (config, legacy):
        if path.is_file():
            return path
    raise FileNotFoundError(f"EmbodiBench task config not found: {config} or {legacy}")


def robotwin_data() -> Path:
    return require_dir(ROBOTWIN_ROOT / "data", "RoboTwin data")


def robotwin_assets() -> Path:
    return require_dir(ROBOTWIN_ROOT / "assets" / "embodiments", "RoboTwin embodiment assets")
