"""Run XPolicyLab's synthetic environment with RoboMorpher's selected robot assets."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import yaml
from XPolicyLab.policy.RoboMorpher.robot_metadata import resolve_robot_bundle
from XPolicyLab.utils import debug_env_client as debug


class RoboMorpherDebugEnv(debug.TestEnv):
    def __init__(self, cfg, robot_root):
        if cfg["bench_name"] == "EmbodiBench":
            # Real episodes supply URDFs with every observation, independently of env_cfg.
            arm = Path(__file__).with_name("debug_arm.urdf").resolve()
            paths, names = [arm, arm], [["shoulder", "elbow"], ["shoulder", "elbow"]]
        else:
            paths, names = resolve_robot_bundle(robot_root, cfg["env_cfg_type"])
        self.robot_context = {"urdf_paths": [str(path) for path in paths], "joint_names": names}
        # TestEnv looks up dimensions alongside the framework checkout. A linked
        # adapter may select a separate robot asset tree, validated by the same
        # resolver the policy uses. Supply that fixture during construction only.
        dims = {"arm_dim": [len(row) for row in names], "ee_dim": [1] * len(names)}
        with patch.object(debug, "get_robot_action_dim_info", return_value=dims):
            super().__init__(cfg)

    def get_obs(self, env_idx=0):
        obs = super().get_obs(env_idx)
        obs["robot_context"] = self.robot_context
        return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_yaml", type=Path)
    parser.add_argument("robot_root", type=Path)
    parser.add_argument("bench_name")
    parser.add_argument("task_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.deploy_yaml.read_text())
    cfg.update(bench_name=args.bench_name, task_name=args.task_name, env_cfg_type=args.env_cfg_type,
               host=args.host, port=args.port, policy_server_url=None, evaluation_id="debug-eval",
               trial_id="debug-trial", action_case_id=None, repeat_index=None,
               obs_encoded=debug.str2bool(os.environ.get("DEBUG_OBS_ENCODED", "0")))
    episodes = int(os.environ.get("ROBOMORPHER_TEST_NUM", 10))
    if episodes < 1:
        raise ValueError("ROBOMORPHER_TEST_NUM must be a positive integer")
    batch = debug.str2bool(os.environ.get("ROBOMORPHER_EVAL_BATCH", str(cfg["eval_batch"])))
    env = RoboMorpherDebugEnv(cfg, args.robot_root)
    try:
        for episode in range(episodes):
            print(f"[RoboMorpher] debug episode {episode + 1}/{episodes}")
            env.reset()
            if batch:
                env.eval_one_episode_batch()
            else:
                env.eval_one_episode()
            env.finish_episode()
    finally:
        close = getattr(env.model_client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
