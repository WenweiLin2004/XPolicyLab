# RoboMorpher

**Contributor:** RoboMorpher Team | **Paper:** RoboMorpher: Scaling Robot Morphology Representations for Cross-Embodiment Action Prediction | **arXiv:** https://arxiv.org/abs/2611.24816 | **Original code:** https://github.com/robomorpher/RoboMorpher | **Project page:** https://robomorpher.github.io/

Wenwei Lin<sup>*</sup>, Yan Qin<sup>*</sup>, Weiliang Deng<sup>*</sup>, Zixuan Li<sup>*</sup>, Kaixuan Wang<sup>*</sup>, Zanxin Chen, Yue Chen, Tianshuo Yang, Junting Chen, Zhixuan Liang, Yao Mu, Ximin Lyu, Renjing Xu, Ping Luo<sup>†</sup>, Tianxing Chen<sup>†</sup><br>
<sup>*</sup>Equal contribution · <sup>†</sup>Corresponding authors

RoboMorpher predicts joint actions for robots with different kinematics. In Stage 1, a morphology-conditioned tokenizer learns to reconstruct joint angles and link poses from 40 million morphology–state pairs. In Stage 2, the tokenizer is frozen. A Qwen3-VL policy predicts its discrete action codes from images, language and robot state, then refines the decoded trajectory into continuous joint commands.

![RoboMorpher architecture showing MorphGen, the morphology-conditioned tokenizer and the discrete-to-continuous policy](https://robomorpher.github.io/assets/architecture-animated.svg)

This XPolicyLab adapter trains and evaluates RoboMorpher on **RoboTwin** and **EmbodiBench**. XPolicyLab serves [the model](model.py) over WebSocket, and [the rollout loop](deploy.py) executes action chunks in the simulator. One checkpoint serves multiple embodiments. In RoboTwin, `env_cfg` selects the robot used by both the simulator and the policy. In EmbodiBench, each episode supplies its own URDF paths and ordered joint names. [XPolicyLab documentation](https://github.com/XPolicyLab/XPolicyLab)

## Installation

From the XPolicyLab repository root, create a Python 3.10 policy environment. The installer clones [MorphGen](https://github.com/robomorpher/MorphGen) into `robomorpher/third_party/MorphGen/` on the first run and reuses that checkout on subsequent runs. It also reuses the active PyTorch build. If PyTorch is missing, it installs PyTorch 2.8.0 and torchvision 0.23.0 for CUDA 12.8. It installs FlashAttention 2.7.4.post1 against that PyTorch build. [Adapter requirements](robomorpher/requirements.txt)

```bash
export XPOLICYLAB_ROOT="$PWD"
cd "$XPOLICYLAB_ROOT/policy/RoboMorpher"
conda create -n RoboMorpher python=3.10 -y
conda activate RoboMorpher
bash install.sh
```

Use `bash install.sh --check` to check the MorphGen checkout, package versions and model imports. `TRAIN_PYTHON` selects the interpreter for installation, conversion and training. Run the commands below from `XPolicyLab/policy/RoboMorpher/`.

## Model assets

Download Qwen3-VL-4B-Instruct for policy training and inference. The script saves the backbone and processor in `robomorpher/pretrained/vlm/Qwen3-VL-4B-Instruct/`. Set `HF_ENDPOINT` to use a Hugging Face mirror.

```bash
bash scripts/download_qwen3_vl_4b_instruct.sh
```

## Training

### Stage 1: Tokenizer Pretraining

#### Generate tokenizer data

If you have a trained tokenizer, continue with [VLA policy training](#stage-2-vla-policy-training). Run the generator from the policy directory. By default, the commands below store the corpus under `data/tokenizer/` beside the XPolicyLab checkout. Set `ROBOMORPHER_TOKENIZER_ROOT` to use another location. Choose a new output directory for each corpus.

```bash
export ROBOMORPHER_TOKENIZER_ROOT="${ROBOMORPHER_TOKENIZER_ROOT:-$XPOLICYLAB_ROOT/../data/tokenizer}"
bash scripts/generate_tokenizer_data.sh \
  --output-dir "$ROBOMORPHER_TOKENIZER_ROOT/lazy_data/morphgen_40m" \
  --num-robots 40000000 --states-per-robot 1 --workers 8
```

This generates **40 million robot URDFs and one state per URDF**. Each joint angle is sampled independently and uniformly within its URDF limits. Forward kinematics supplies the link poses. The original 40M training corpus used home-pose and cuRobo inverse-kinematics sampling. The generator writes Zarr shards, per-shard manifests with the URDFs and row ranges, and a final `index.json`.

To check the data format on CPU, generate 12 robots and 12 rows. Set `MORPHGEN_ROOT` if you use a different MorphGen checkout.

```bash
bash scripts/generate_tokenizer_data.sh \
  --output-dir "$ROBOMORPHER_TOKENIZER_ROOT/lazy_data/morphgen_demo" \
  --num-robots 12 --states-per-robot 1 --seed 42 --workers 1
```

#### Pretrain the tokenizer

Tokenizer pretraining uses morphology–state pairs independently of benchmark demonstrations. Use the same `ROBOMORPHER_TOKENIZER_ROOT` as in the generation step. Pass the directory containing the generated `index.json` as `train_data_dir`.

```bash
NUM_GPUS=8 bash train.sh tokenizer robomorpher \
  train_data_dir="$ROBOMORPHER_TOKENIZER_ROOT/lazy_data/morphgen_40m"
```

For the home-pose and cuRobo IK corpus, omit `train_data_dir`. The recipe then reads `lazy_data/home_pose_ik_40m/` under `ROBOMORPHER_TOKENIZER_ROOT`. With the variable unset, the launcher checks for `data/tokenizer/` next to the XPolicyLab checkout, then falls back to the adapter’s own `data/tokenizer/`. Set `NUM_GPUS` to the number of GPUs to use.

The default tokenizer recipe, `robomorpher`, trains for 300 epochs and saves the final checkpoint at `robomorpher/pretrained/tokenizer/robomorpher/epoch_300.pt`. Set `save_path` to choose another directory. [Tokenizer configuration](robomorpher/config/tokenizer/robomorpher.yaml)

To resume Stage 1, set `vqvae_model.load_dir` to the tokenizer checkpoint. If PyTorch 2.8 reports an unsupported pickle opcode for an older checkpoint you created or trust, prefix that training command with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. This enables full pickle deserialization for that run.

### Stage 2: VLA Policy Training

#### Prepare policy data

Set `ROBOMORPHER_RAW_DATA` to the native HDF5 data, task or collection directory to convert. When it is unset, the launcher first looks for `data/<bench>/` beside the XPolicyLab checkout, then for `data/` inside the benchmark checkout. Set `ROBOTWIN_ROOT` or `EMBODIBENCH_ROOT` to choose the benchmark checkout. The converter reads RoboTwin URDFs from `assets/embodiments/` and EmbodiBench URDFs from `MorphGen/embodiments/<project>/` within the respective checkouts.

The converter writes a LeRobot v3 dataset with aligned states and actions, instructions and three RGB views. RoboMorpher also requires the additional columns `observation.urdf_features` and `observation.joint_valid_mask`. Each arm has seven joint slots and one gripper slot. The mask identifies the joints present in each embodiment.

Use the XPolicyLab form `bash process_data.sh <bench> <ckpt_name> <env_cfg> <action_type>` or the shorter `bash process_data.sh <bench> [<ckpt_name> [<action_type>]]`. Both export every matching episode in the input directory to `data/<bench>-<ckpt_name>-<action_type>/`. Each episode supplies its training embodiment. The defaults are `vla_robomorpher` and `joint`.

```bash
bash process_data.sh RoboTwin vla_robomorpher piper joint
bash process_data.sh EmbodiBench vla_robomorpher joint
```

Set `EXPORT_DATA_PATH` to write to a different, unused directory. During training, point `ROBOMORPHER_DATA_ROOT` to that export.

EmbodiBench collections live under `data/<task>/<collection>/`. The converter includes all collections by default. If two collections share trajectories but have separate metadata, each contributes its own episodes. Keep `seen_embodiment.txt` beside the collection’s `data/` directory. The converter uses it to pair episodes with robots and reads `project_name` from `collection_settings.json` or `env_cfg/task_config/<collection>.yml` to locate the URDFs. To select one collection across tasks, set `ROBOMORPHER_COLLECTION_CONFIG` to its name.

After collecting `lift_dual_ss` with the `demo` configuration, set `EMBODIBENCH_ROOT` to your EmbodiBench checkout. This example converts the collection into `data/EmbodiBench-demo-joint/`. [EmbodiBench collection guide](https://github.com/robomorpher/EmbodiBench)

```bash
ROBOMORPHER_RAW_DATA="$EMBODIBENCH_ROOT/data/lift_dual_ss/demo" \
  bash process_data.sh EmbodiBench demo joint
```

#### Train the VLA policy

Prepare the [benchmark demonstrations](#prepare-policy-data) and the [Qwen backbone](#model-assets). The policy loads the Stage 1 tokenizer through `action_tokenizer.checkpoint` and freezes its weights. The backbone and action head then train on the selected benchmark export.

Run `bash train.sh <bench> <ckpt_name> <env_cfg> <action_type> <seed> <gpu_ids>` or omit `<env_cfg>` for the shorter form. Training uses the full prepared dataset across embodiments. The benchmark, checkpoint name and action type identify the dataset. The seed also identifies the training run. The seed and GPU list each default to `0`. The default policy recipe, `vla_robomorpher`, loads `robomorpher/pretrained/tokenizer/robomorpher/epoch_300.pt` from [tokenizer pretraining](#pretrain-the-tokenizer).

```bash
bash train.sh RoboTwin vla_robomorpher piper joint 0 0
bash train.sh EmbodiBench vla_robomorpher joint 0 0,1
```

The RoboTwin example reads `data/RoboTwin-vla_robomorpher-joint/` and writes to `checkpoints/RoboTwin-vla_robomorpher-joint-0/`. In general, checkpoints go under `checkpoints/<bench>-<ckpt_name>-<action_type>-<seed>/`. Append `action_tokenizer.checkpoint=/path/to/epoch_300.pt` to use a different tokenizer. To train on the `demo` collection above, run `bash train.sh EmbodiBench demo joint 0 0`. Put further overrides such as `trainer.batch_size=8` after the GPU list. [Policy configuration](robomorpher/config/policy/vla_robomorpher.yaml)

## Evaluation

Use the frozen tokenizer and action statistics from the policy’s training run. Keep the policy weights and matching `_meta.pt` file together. For a RoboTwin run with seed 0 and a checkpoint at step 30,000, the adapter paths are shown below. [Model configuration and checkpoint notes](robomorpher/README.md)

```text
checkpoints/RoboTwin-vla_robomorpher-joint-0/
├── checkpoint_manifest.json
└── checkpoints/
    ├── steps_30000_pytorch_model.pt
    └── steps_30000_meta.pt

data/RoboTwin-vla_robomorpher-joint/meta/stats_masked.json
```

Set `POLICY_ENV` to the policy conda environment and `EVAL_ENV` to the simulator conda environment. Both can be environment names or prefixes. This example runs one RoboTwin episode on the Piper embodiment, with the policy on GPU 0 and the simulator on GPU 1.

```bash
ROBOMORPHER_TEST_NUM=1 bash eval.sh RoboTwin beat_block_hammer vla_robomorpher piper joint 0 \
  0 1 "$POLICY_ENV" "$EVAL_ENV"
```

The arguments are `<Bench> <task> <ckpt_name> <env_cfg> <action> <seed> <policy_gpu> <sim_gpu> <policy_env> <eval_env> [task_config]`. The short name above resolves to `checkpoints/RoboTwin-vla_robomorpher-joint-0/` and selects the latest complete step. To evaluate a particular step or a checkpoint trained on another benchmark, pass its full run directory or weight path as the third argument. `ROBOMORPHER_CHECKPOINT` can also select an explicit weight file.

To evaluate the same checkpoint on a different RoboTwin robot, change only `env_cfg`. For example, replace `piper` with `franka` in the command above. The simulator and the policy resolve the robot through the RoboTwin embodiment registry and use its configured URDF and joint order. The released checkpoint supports up to seven arm joints and one gripper coordinate per arm. The optional `task_config` selects the task settings for the run. It defaults to `demo_clean`.

EmbodiBench defaults to `env_cfg/task_config/default.yml` and loads the robot described by each episode. To use a collected configuration, pass its name as the last argument. The following example evaluates a policy trained on `data/EmbodiBench-demo-joint/` using the `demo` collection.

```bash
ROBOMORPHER_TEST_NUM=1 bash eval.sh EmbodiBench lift_dual_ss demo demo joint 0 \
  0 1 "$POLICY_ENV" "$EVAL_ENV" demo
```

For separate policy and simulator machines, run the first command on the server and the second on the client. Set `SERVER_HOST` on the client to the server’s reachable address. For EmbodiBench, make the episode URDFs readable at the same paths on the policy server.

```bash
bash setup_eval_policy_server.sh RoboTwin beat_block_hammer vla_robomorpher piper joint 0 \
  0 "$POLICY_ENV" 8000 0.0.0.0

bash setup_eval_env_client.sh RoboTwin beat_block_hammer vla_robomorpher piper joint 0 \
  1 "$EVAL_ENV" '' 8000 "$SERVER_HOST"
```

## Acknowledgements

RoboMorpher builds on [StarVLA](https://github.com/starVLA/starVLA), [Discrete Forcing](https://github.com/Jbo-Wang/discrete_forcing), [LeRobot](https://github.com/huggingface/lerobot) and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).

## Citation

```bibtex
@misc{lin2026robomorpher,
  title         = {{RoboMorpher}: Scaling Robot Morphology Representations for Cross-Embodiment Action Prediction},
  author        = {Wenwei Lin and Yan Qin and Weiliang Deng and Zixuan Li and Kaixuan Wang and Zanxin Chen and Yue Chen and Tianshuo Yang and Junting Chen and Zhixuan Liang and Yao Mu and Ximin Lyu and Renjing Xu and Ping Luo and Tianxing Chen},
  year          = {2026},
  eprint        = {2611.24816},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.24816}
}
```
