"""Run RoboTwin with the embodiment selected by the XPolicyLab environment config."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from robot_metadata import resolve_embodiment


def main():
    root = Path(sys.argv.pop(1)).resolve()
    deploy_file = Path(sys.argv.pop(1))
    env_cfg = sys.argv.pop(1)
    deploy = yaml.safe_load(deploy_file.read_text())
    os.chdir(root)
    sys.path.insert(0, str(root / "scripts"))
    import eval_policy_xpolicylab as simulator
    from test_render import Sapien_TEST

    sys.argv[1:1] = ["--protocol", deploy["protocol"], "--eval_batch", str(deploy["eval_batch"])]
    args_file = os.environ.get("ROBOTWIN_EVAL_ARGS_FILE")
    if args_file:
        sys.argv.extend(Path(args_file).read_text().splitlines())
    args = simulator.parse_args()
    args["env_cfg_type"] = env_cfg
    name = os.environ.get("ROBOMORPHER_TASK_CONFIG") or args["task_config"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"Expected a RoboTwin task configuration name, got {name!r}")
    args["task_config"] = name
    config_root = simulator.TASK_CONFIG_ROOT
    task_path = config_root / f"{name}.yml"
    if not task_path.is_file():
        raise FileNotFoundError(f"RoboTwin task configuration not found: {task_path}")
    task = yaml.safe_load(task_path.read_text())
    embodiment = resolve_embodiment(root, args["env_cfg_type"])
    print(f"[RoboMorpher] env_cfg={args['env_cfg_type']} task_config={name} embodiment={embodiment}", flush=True)
    run = simulator.main_batch if simulator.parse_bool(args["eval_batch"]) else simulator.main
    if task.get("embodiment") != embodiment:
        # RoboTwin reads a named task YAML and separately uses that name for
        # results. Override the robot only while its loader reads the scene.
        task["embodiment"] = embodiment
        original_load = simulator.load_task_args

        def load_selected_task(run_args):
            with TemporaryDirectory(prefix="robomorpher_robotwin_") as temp:
                config = Path(temp)
                (config / task_path.name).write_text(yaml.safe_dump(task, sort_keys=False))
                (config / "_camera_config.yml").symlink_to(config_root / "_camera_config.yml")
                simulator.TASK_CONFIG_ROOT = config
                try:
                    return original_load(run_args)
                finally:
                    simulator.TASK_CONFIG_ROOT = config_root

        simulator.load_task_args = load_selected_task
    Sapien_TEST()
    run(args)


if __name__ == "__main__":
    main()
