#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 || $# -gt 10 ]]; then
  echo "Usage: setup_eval_policy_server.sh <bench> <task> <ckpt_name> <env_cfg> <action_type> <seed> <policy_gpu> <policy_env> <port> [host]" >&2
  exit 2
fi
bench_name=$1; task_name=$2; model_selector_arg=$3; env_cfg_type=$4; action_type=$5; seed=$6
policy_gpu_id=$7; policy_conda_env=$8; policy_server_port=$9; policy_server_host=${10:-localhost}
policy_dir="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
source "$policy_dir/scripts/xpolicylab_paths.sh"
resolve_xpolicylab_paths "$policy_dir" "$bench_name"
if [[ ! "$XPOLICYLAB_ROOT/policy/RoboMorpher" -ef "$policy_dir" ]]; then
  echo "[RoboMorpher] $XPOLICYLAB_ROOT/policy/RoboMorpher must point to $policy_dir" >&2
  exit 1
fi
case "$action_type" in
  joint|qpos) ;;
  *) echo "[RoboMorpher] action_type must be joint or qpos" >&2; exit 2 ;;
esac
# Use the evaluation environments' Vulkan libraries and the benchmark ICD when available.
ROBOMORPHER_SIM_ENV_ROOT="${ROBOMORPHER_SIM_ENV_ROOT:-$(dirname "$policy_conda_env")/RoboTwin}"
vulkan_lib="${ROBOMORPHER_VULKAN_LIB:-${policy_conda_env}/lib:${ROBOMORPHER_SIM_ENV_ROOT}/lib}"
export LD_LIBRARY_PATH="${vulkan_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
icd_file="$bench_root/.vulkan/nvidia_egl_icd.json"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f "$icd_file" ]]; then
  export VK_ICD_FILENAMES="$icd_file"
fi
if [[ -z "${VK_DRIVER_FILES:-}" && -f "$icd_file" ]]; then
  export VK_DRIVER_FILES="$icd_file"
fi
runtime_root="$policy_dir/.cache/runtime"
export XDG_RUNTIME_DIR="${ROBOMORPHER_XDG_RUNTIME_DIR:-$runtime_root/server_${policy_gpu_id}}"
export TMPDIR="${ROBOMORPHER_TMPDIR:-$runtime_root/server_tmp_${policy_gpu_id}}"
export XDG_CACHE_HOME="${ROBOMORPHER_XDG_CACHE_HOME:-$runtime_root/server_cache_${policy_gpu_id}}"
mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
yaml_file="${ROBOMORPHER_DEPLOY_CONFIG:-$policy_dir/deploy.yml}"
[[ -f "$yaml_file" ]] || { echo "[RoboMorpher] deploy config missing: $yaml_file" >&2; exit 2; }
selector="${ROBOMORPHER_MODEL_CONFIG:-$model_selector_arg}"
if [[ -x "$policy_conda_env/bin/python" ]]; then
  PYTHON_BIN="$policy_conda_env/bin/python"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "[RoboMorpher] policy env must be a path or conda env: $policy_conda_env" >&2; exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$policy_conda_env"
  PYTHON_BIN=python
fi
echo "[RoboMorpher] model selector=$selector"
echo "[RoboMorpher] policy env=$policy_conda_env, gpu=$policy_gpu_id, server=$policy_server_host:$policy_server_port"
override_args=()
if [[ -n "${ROBOMORPHER_CHECKPOINT:-}" ]]; then
  override_args+=(checkpoint_path="$ROBOMORPHER_CHECKPOINT")
fi
if [[ -n "${ROBOMORPHER_CONTINUOUS_INFERENCE_MODE:-}" ]]; then
  override_args+=(continuous_inference_mode="${ROBOMORPHER_CONTINUOUS_INFERENCE_MODE}")
fi
if [[ -n "${ROBOMORPHER_CONTINUOUS_INFERENCE_STEPS:-}" ]]; then
  override_args+=(continuous_inference_steps="${ROBOMORPHER_CONTINUOUS_INFERENCE_STEPS}")
fi
if [[ -n "${ROBOMORPHER_DISCRETE_INFERENCE_MODE:-}" ]]; then
  override_args+=(discrete_inference_mode="${ROBOMORPHER_DISCRETE_INFERENCE_MODE}")
fi
if [[ "${#override_args[@]}" -gt 0 ]]; then
  echo "[RoboMorpher] deploy overrides: ${override_args[*]}"
fi
device=cuda
default_precision=bf16
if [[ $policy_gpu_id == -1 ]]; then
  device=cpu
  default_precision=fp32
fi
exec env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="$policy_gpu_id" \
  "$PYTHON_BIN" -u "$XPOLICYLAB_ROOT/setup_policy_server.py" --config_path "$yaml_file" \
  --overrides port="$policy_server_port" host="$policy_server_host" \
  bench_name="$bench_name" task_name="$task_name" ckpt_name="$selector" \
  env_cfg_type="$env_cfg_type" seed="$seed" \
  policy_name=RoboMorpher action_type=joint eval_batch=false robotwin_root="$bench_root" \
  xpolicylab_root="$XPOLICYLAB_ROOT" device="$device" precision="${ROBOMORPHER_PRECISION:-$default_precision}" \
  memory_budget_gib=${ROBOMORPHER_MEMORY_BUDGET_GIB:-16} \
  reserve_free_gib=${ROBOMORPHER_RESERVE_FREE_GIB:-2} \
  "${override_args[@]}"
