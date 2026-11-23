# RoboMorpher model configuration

The XPolicyLab adapter provides one recipe for the morphology tokenizer and one for the vision-language-action policy. Run the commands below from `XPolicyLab/policy/RoboMorpher/`. See the [adapter README](../README.md) for installation, data preparation and benchmark evaluation.

## Morphology tokenizer

The tokenizer encodes morphology from a URDF and joint angles in separate branches, then combines them in a joint encoder. For each valid joint, residual quantization selects one code from each of two 2,048-entry codebooks. The decoder reconstructs the joint angle and link pose. The training recipe uses up to seven joints per arm and trains for 300 epochs. [Tokenizer recipe](config/tokenizer/robomorpher.yaml) · [Tokenizer implementation](model/tokenizer/)

Set `train_data_dir` to the corpus directory containing `index.json` and `save_path` to the checkpoint directory. The default recipe reads the original home-pose and cuRobo IK corpus. For data from the [MorphGen launcher](../scripts/generate_tokenizer_data.sh), pass the `morphgen_40m/` directory explicitly. The recipe saves a checkpoint every two epochs and keeps the latest eight. Set `vqvae_model.load_dir` to an epoch checkpoint to resume training. The policy recipe loads `robomorpher/pretrained/tokenizer/robomorpher/epoch_300.pt` by default. [Tokenizer data and training](../README.md#pretrain-the-tokenizer)

## Vision-language-action policy

The policy conditions on the instruction, current robot state and three 224 × 224 RGB views (`cam_high`, `cam_left_wrist` and `cam_right_wrist`). Qwen3-VL-4B-Instruct encodes the visual and language inputs. The action head predicts codes from the frozen tokenizer at ten points across a 50-step chunk, decodes a coarse trajectory and refines it into continuous joint commands. The default recipe uses eight MaskGIT decoding steps and four continuous refinement steps. [Policy recipe](config/policy/vla_robomorpher.yaml) · [Policy implementation](model/architectures/qwen_discrete_forcing.py)

| Configuration key | Controls |
| --- | --- |
| `framework.qwenvl.base_vlm` | Base model directory, default `./robomorpher/pretrained/vlm/Qwen3-VL-4B-Instruct` |
| `action_tokenizer.checkpoint` | Frozen morphology tokenizer loaded alongside the policy |
| `framework.action_model.action_horizon` / `framework.action_model.execute_horizon` | Number of actions predicted and executed per chunk, both 50 by default |
| `data.root` | Prepared LeRobot v3 export, including `meta/stats_masked.json` |
| `trainer.output_dir` / `trainer.seed` | Policy run directory and training seed, set by the benchmark launcher |
| `trainer.resume` | Weight file for continuing a compatible training run |

The policy uses 16 action values, with seven joint slots and one gripper slot per arm. The adapter maps XPolicyLab’s `cam_head` to the policy’s `cam_high` and reads the left and right wrist cameras by name. For each arm, the input state contains joint angles in URDF order and one gripper value in `[0, 1]`. During evaluation, joint masks restrict the output to active joints. The rollout loop executes the resulting chunk and requests a new one when it ends. [XPolicyLab model adapter](../model.py)

## Policy checkpoints

The default policy recipe runs for 30,000 optimizer steps and saves every 5,000 steps. Each saved step has a `steps_<N>_pytorch_model.pt` file with the policy weights and a `steps_<N>_meta.pt` file with the training configuration and tokenizer identity. The run’s `checkpoint_manifest.json` lists completed steps. To evaluate a run on another machine, copy the run directory, the original tokenizer checkpoint and `meta/stats_masked.json` from the training dataset. The policy also needs the Qwen backbone and processor. [Evaluation instructions](../README.md#evaluation)

To train with a custom recipe, use `bash train.sh policy <config>` or `bash train.sh tokenizer <config>`. The launcher accepts a recipe name or YAML path followed by OmegaConf `key=value` overrides. For a direct policy launch, specify `data.root` and `trainer.output_dir` to select the dataset and run directory. If omitted, the default policy recipe uses RoboTwin data and a RoboTwin run name. For a prepared benchmark dataset, the benchmark launcher selects the dataset by benchmark, run name and action type, and includes the seed in the checkpoint directory name. [Training commands](../README.md#training)
