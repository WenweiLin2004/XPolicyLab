"""Find complete policy checkpoints and their local inference assets."""
from __future__ import annotations

import json
import re
from pathlib import Path

from omegaconf import OmegaConf


BENCHES = ("RoboTwin", "EmbodiBench")
DEFAULT_RUN = "vla_robomorpher"
RELEASE_RUN = "RoboTwin-vla_robomorpher-joint-0"
_WEIGHT = re.compile(r"steps_([0-9]+)_pytorch_model\.pt")
_OLD_DATASETS = {"robotwin_15tasks_50": "RoboTwin", "embodibench_10tasks": "EmbodiBench"}


def _explicit_path(root, value):
    path = root / Path(value).expanduser()
    if path.exists():
        return path
    # Only translate missing historical paths in this repository. Keeping an
    # existing weight's logical name also keeps its own curated sibling metadata.
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path
    if rel.parts[:2] == ("robomorpher", "checkpoints"):
        rel = Path(*rel.parts[1:])
    if rel.parts[:3] == ("checkpoints", "policy", DEFAULT_RUN):
        rel = Path("checkpoints", RELEASE_RUN, *rel.parts[3:])
    return root / rel


def _completed_weight(path):
    if path.is_file():
        if not _WEIGHT.fullmatch(path.name):
            raise ValueError(f"Expected steps_<N>_pytorch_model.pt, got {path}")
        meta = path.with_name(path.name.replace("_pytorch_model.pt", "_meta.pt"))
        if not meta.is_file():
            raise FileNotFoundError(f"Missing checkpoint metadata: {meta}")
        return path
    manifest = path / "checkpoint_manifest.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text())["checkpoints"]
        for entry in sorted(entries, key=lambda it: int(it["step"]), reverse=True):
            candidate = path / entry["name"]
            stem = f"steps_{int(entry['step'])}"
            if (candidate.name == f"{stem}_pytorch_model.pt" and candidate.is_file()
                    and candidate.with_name(f"{stem}_meta.pt").is_file()):
                return candidate
        raise FileNotFoundError(f"No completed checkpoint listed in {manifest}")
    if path.is_dir():
        ranked = []
        for folder in (path / "checkpoints", path):
            for candidate in folder.glob("steps_*_pytorch_model.pt"):
                match = _WEIGHT.fullmatch(candidate.name)
                if (match and candidate.is_file()
                        and candidate.with_name(f"steps_{match[1]}_meta.pt").is_file()):
                    ranked.append((int(match[1]), candidate))
        if ranked:
            return max(ranked)[1]
    return None


def select_checkpoint(root, value=None, *, bench=None, action_type="joint", seed=0, extra_candidates=None):
    """Resolve a YAML, explicit run/weight path, or a benchmark-qualified run name."""
    root = Path(root)
    name = str(value or DEFAULT_RUN)
    seed = 0 if seed is None else seed
    action_type = str(action_type or "joint").lower()
    if bench is not None and bench not in BENCHES:
        raise ValueError(f"Unknown benchmark {bench!r}; choose RoboTwin or EmbodiBench")
    if action_type not in ("joint", "qpos"):
        raise ValueError(f"RoboMorpher supports joint actions; got {action_type!r}")
    directory = root / "checkpoints"
    path = _explicit_path(root, name)
    configured_output = None
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not path.is_file() and Path(name).parent in (Path("."), Path("configs/policy"), Path("robomorpher/configs/policy")):
            path = root / "robomorpher" / "config" / "policy" / path.name
        if not path.is_file():
            raise FileNotFoundError(f"Missing policy config: {path}")
        config = OmegaConf.load(path)
        configured_output = OmegaConf.select(config, "trainer.output_dir") or config.get("output_dir")
        name = path.stem
    elif Path(name).is_absolute() or "/" in name or path.exists():
        weight = _completed_weight(path)
        if weight is None:
            raise FileNotFoundError(f"No complete policy checkpoint at {path}")
        return weight

    qualified = [directory / f"{b}-{name}-joint-{seed}" for b in ([bench] if bench else BENCHES)]
    matches = [(run, weight) for run in qualified if (weight := _completed_weight(run)) is not None]
    if len(matches) > 1:
        options = ", ".join(run.name for run, _ in matches)
        raise ValueError(f"Multiple benchmarks match {name!r}: {options}. Pass --bench or a full run name.")
    if matches:
        return matches[0][1]

    candidates = list(extra_candidates(name)) if extra_candidates else []
    candidates.extend([directory / name, directory / "policy" / name, directory / "policy" / f"{name}_224"])
    if configured_output:
        candidates.append(_explicit_path(root, configured_output))
    for candidate in dict.fromkeys(candidates):
        # The historical bare release is known to contain RoboTwin step-30000 weights.
        if candidate == directory / "policy" / DEFAULT_RUN and (bench == "EmbodiBench" or str(seed) != "0"):
            continue
        found = _completed_weight(candidate)
        if found is not None:
            return found
    raise FileNotFoundError("No completed policy checkpoint at " + ", ".join(map(str, [*qualified, *candidates])))


def dataset_asset_id(value):
    """Canonical name of an asset inside one of the two renamed release datasets."""
    parts = Path(value).parts
    if "data" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("data")
    if index and parts[index - 1] != "RoboMorpher":
        return None
    suffix = parts[index + 1:]
    if len(suffix) >= 3 and suffix[:2] == ("policy", "lerobot"):
        bench = _OLD_DATASETS.get(suffix[2])
        if bench:
            return Path("data", f"{bench}-{DEFAULT_RUN}-joint", *suffix[3:]).as_posix()
    if suffix and suffix[0] in {f"{bench}-{DEFAULT_RUN}-joint" for bench in BENCHES}:
        return Path("data", *suffix).as_posix()
    return None


def bundled_vlm_suffix(path) -> str | None:
    """Return the model subpath for a recognized project-relative VLM location."""
    parts = Path(path).parts
    if parts[:1] == ("robomorpher",):
        parts = parts[1:]
    if parts[:1] == ("checkpoints",):
        parts = parts[1:]
    if (len(parts) == 3 and parts[:2] in (("pretrained", "qwen"), ("pretrained", "vlm"))
            and parts[2].lower().startswith("qwen")):
        return parts[2]
    return None


def bind_checkpoint_assets(root, config, weight):
    """Relocate only known RoboMorpher assets from saved training configs."""
    root = Path(root)
    roots = [root]
    for parent in weight.resolve().parents:
        if parent.name == "policy" and parent.parent.name == "checkpoints":
            roots.append(parent.parent.parent)
            break
    for key in ("framework.qwenvl.base_vlm", "action_tokenizer.checkpoint",
                "action_tokenizer.tokenizer_path", "action_tokenizer.stats_path",
                "framework.action_model.continuous_stats_path",
                "framework.action_model.discrete_forcing.continuous_stats_path", "data.root"):
        value = OmegaConf.select(config, key)
        if not value:
            continue
        raw = Path(value).expanduser()
        rel = raw
        if raw.is_absolute():
            if "RoboMorpher" not in raw.parts:
                continue
            pos = len(raw.parts) - 1 - raw.parts[::-1].index("RoboMorpher")
            rel = Path(*raw.parts[pos+1:])
        candidates = []
        if key == "framework.qwenvl.base_vlm":
            if raw.is_absolute() and raw.exists():
                # Relocate a live path only for this checkout or the recorded training checkout.
                # An explicitly supplied model from a different checkout keeps its identity.
                source_root = raw.parents[len(rel.parts) - 1] if rel.parts else raw
                output = OmegaConf.select(config, "trainer.output_dir")
                from_training = bool(output and Path(output).is_absolute()
                                     and Path(output).resolve().is_relative_to(source_root.resolve()))
                if not from_training and all(source_root.resolve() != base.resolve() for base in roots):
                    continue
            elif not raw.is_absolute() and len(raw.parts) <= 2 and not str(value).startswith(("./", "../")):
                # Hugging Face IDs, including org/model, are passed through as given.
                continue
            suffix = bundled_vlm_suffix(rel)
            if suffix:
                candidates.append(root / "robomorpher" / "pretrained" / "vlm" / suffix)
            elif raw.is_absolute() and raw.exists():
                continue
        if rel.parts[:3] == ("checkpoints", "tokenizer", "robomorpher"):
            candidates.append(root / "robomorpher" / "pretrained" / Path(*rel.parts[1:]))
        elif rel.parts[:2] == ("checkpoints", "pretrained"):
            candidates.append(root / "robomorpher" / Path(*rel.parts[1:]))
        elif rel.parts[:1] == ("pretrained",):
            candidates.append(root / "robomorpher" / rel)
        elif dataset := dataset_asset_id(str(rel)):
            candidates.append(root / dataset)
        elif raw.is_absolute() and raw.exists() and not candidates:
            continue
        if raw.is_absolute():
            candidates.append(raw)
        candidates.extend(base / rel for base in roots)
        for candidate in candidates:
            if key == "data.root" and not (candidate / "meta" / "stats_masked.json").is_file():
                continue
            if candidate.exists():
                OmegaConf.update(config, key, str(candidate), merge=False)
                break
