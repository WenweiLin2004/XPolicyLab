"""Resolve the same RoboTwin embodiment assets for simulation, conversion and inference."""
import json
import math
from pathlib import Path

import yaml


def _yaml(path):
    return yaml.safe_load(path.read_text()) or {}


def _key(name):
    return name.casefold().replace("-", "_")


# Older RoboTwin env and dataset names use shorter names than the asset registry.
_LEGACY_NAMES = {"x5": "arx_x5", "arx": "arx_x5", "franka": "franka_panda",
                 "aloha": "aloha_agilex", "ur5": "ur5_wsg"}


def _resolve_layout(root, selector):
    env_dir = root / "env_cfg"
    task_dir = env_dir / "task_config"
    registry = _yaml(task_dir / "_embodiment_config.yml")
    if not isinstance(selector, str) or not selector or Path(selector).name != selector or selector in {".", ".."}:
        raise ValueError(f"Invalid RoboTwin environment name: {selector!r}")
    selector_key = _key(selector)
    normalized = {_key(name): name for name in registry}

    def registered(name):
        if name in registry:
            return name
        key = _key(name)
        return normalized.get(key) or normalized.get(_LEGACY_NAMES.get(key))

    def validate_layout(layout):
        if not isinstance(layout, list) or len(layout) not in (1, 3):
            raise ValueError(f"{selector}: expected embodiment [robot] or [left, right, distance], got {layout!r}")
        names = layout[:1] if len(layout) == 1 else layout[:2]
        resolved = [registered(name) if isinstance(name, str) else None for name in names]
        if not all(resolved):
            raise ValueError(f"{selector}: unregistered embodiment {names!r}; available: {sorted(registry)}")
        if len(layout) == 3:
            distance = float(layout[2])
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError(f"{selector}: invalid robot separation {layout[2]!r}")
            resolved.append(distance)
        return resolved

    env_file = env_dir / f"{selector}.yml"
    if not env_file.is_file():
        files = [p for p in env_dir.glob("*.yml") if _key(p.stem) == selector_key]
        # In RoboTwin, the franka-panda asset is configured by franka.yml.
        if not files and selector_key == "franka_panda" and (env_dir / "franka.yml").is_file():
            files = [env_dir / "franka.yml"]
        if len(files) > 1:
            raise ValueError(f"Ambiguous environment spelling {selector!r}: {files}")
        env_file = files[0] if files else None
    env = _yaml(env_file) if env_file else {}
    config = env.get("config", {})
    robot = config.get("robot", selector)
    explicit = config.get("embodiment", env.get("embodiment"))
    env_task = task_dir / f"demo_clean_{selector}.yml"
    if env_file and not env_task.is_file():
        env_task = task_dir / f"demo_clean_{env_file.stem}.yml"
    if explicit is not None:
        layout = validate_layout(explicit)
    else:
        name = _key(robot) if isinstance(robot, str) else ""
        asset = registered(name) or registered(name.removeprefix("dual_"))
        if asset is not None:
            asset_file = root / registry[asset]["file_path"] / "config.yml"
            dual_arm = _yaml(asset_file).get("dual_arm")
            if not isinstance(dual_arm, bool):
                raise ValueError(f"Set dual_arm in {asset_file} or declare config.embodiment in {env_file}")
            layout = [asset] if dual_arm else [asset, asset, 0.6]
        else:
            # Native collection names can concatenate two different registered assets.
            pairs = set()
            for i, char in enumerate(name):
                if char == "_":
                    left, right = registered(name[:i]), registered(name[i + 1:])
                    if left and right:
                        pairs.add((left, right))
            if len(pairs) == 1:
                left, right = pairs.pop()
                layout = [left, right, 0.6]
            elif len(pairs) > 1:
                raise ValueError(f"Ambiguous robot pair {robot!r}; declare config.embodiment explicitly")
            else:
                # An environment-specific task may provide the explicit asset mapping.
                if not env_task.is_file():
                    raise ValueError(f"Cannot resolve RoboTwin environment {selector!r} (robot {robot!r}); "
                                     f"register the asset or declare config.embodiment; available: {sorted(registry)}")
                layout = validate_layout(_yaml(env_task).get("embodiment"))
        # A named environment may calibrate its two arm bases for this particular pair.
        if len(layout) == 3 and env_task.is_file():
            task_layout = _yaml(env_task).get("embodiment")
            if isinstance(task_layout, list) and len(task_layout) == 3:
                if [registered(name) for name in task_layout[:2]] == layout[:2]:
                    layout = validate_layout(task_layout)
    return layout, registry, env_file, robot


def resolve_embodiment(robotwin_root, env_cfg_type):
    """Return RoboTwin's [combined_asset] or [left_asset, right_asset, separation]."""
    return _resolve_layout(Path(robotwin_root), env_cfg_type)[0]


def resolve_robot_bundle(robotwin_root, env_cfg_type):
    """Return ordered URDF paths and ordered joint-name lists for the two simulator arms."""
    root = Path(robotwin_root).resolve()
    layout, registry, env_file, robot = _resolve_layout(root, env_cfg_type)
    assets = layout * 2 if len(layout) == 1 else layout[:2]
    paths, names = [], []
    for side, asset in enumerate(assets):
        asset_dir = root / registry[asset]["file_path"]
        config = _yaml(asset_dir / "config.yml")
        rows = config.get("arm_joints_name")
        if not isinstance(rows, list) or len(rows) != 2 or not isinstance(rows[side], list) or not rows[side]:
            raise ValueError(f"{asset_dir}/config.yml lacks ordered joints for arm {side}")
        names.append(list(rows[side]))
        paths.append((asset_dir / config["urdf_path"]).resolve(strict=True))

    # Native dataset pair names may only have an asset registration. A named XPolicyLab
    # environment can additionally specify expected widths in its benchmark's registry.
    if env_file is not None:
        from XPolicyLab.utils import process_data

        dim_registry = root / "env_cfg" / "robot" / "_robot_info.json"
        dims = json.loads(dim_registry.read_text()).get(robot) if dim_registry.is_file() else None
        if dims is not None:
            if Path(process_data.__file__).resolve().parents[2] == root and "robot" in _yaml(env_file).get("config", {}):
                dims = process_data.get_robot_action_dim_info(env_file.stem)
            arm, ee = dims["arm_dim"], dims["ee_dim"]
            if [len(row) for row in names] != list(arm) or list(ee) != [1, 1]:
                raise ValueError(f"{env_cfg_type}: RoboTwin joint counts {[len(row) for row in names]} "
                                 f"do not match the registered arm/gripper dimensions {arm}/{ee}")
    return paths, names
