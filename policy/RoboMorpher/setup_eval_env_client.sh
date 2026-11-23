#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 10 || $# -gt 11 ]]; then echo "Usage: setup_eval_env_client.sh <bench> <task> <ckpt> <env_cfg> <action_type> <seed> <env_gpu> <eval_env> <additional_info> <port> [host]" >&2; exit 2; fi
bench_name=$1; task_name=$2; ckpt_name=$3; env_cfg_type=$4; action_type=$5; seed=$6; env_gpu_id=$7; eval_env_conda_env=$8; additional_info=$9; policy_server_port=${10}; policy_server_ip=${11:-localhost}
policy_dir="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
source "$policy_dir/scripts/xpolicylab_paths.sh"
resolve_xpolicylab_paths "$policy_dir" "$bench_name"
yaml_file="${ROBOMORPHER_DEPLOY_CONFIG:-$policy_dir/deploy.yml}"
# Use the simulator environment's Vulkan loader and the benchmark ICD when available.
ROBOMORPHER_SIM_ENV_ROOT="${ROBOMORPHER_SIM_ENV_ROOT:-$(dirname "$eval_env_conda_env")/RoboTwin}"
vulkan_lib="${ROBOMORPHER_VULKAN_LIB:-${eval_env_conda_env}/lib:${ROBOMORPHER_SIM_ENV_ROOT}/lib}"
export LD_LIBRARY_PATH="${vulkan_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
icd_file="$bench_root/.vulkan/nvidia_egl_icd.json"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f "$icd_file" ]]; then
  export VK_ICD_FILENAMES="$icd_file"
fi
if [[ -z "${VK_DRIVER_FILES:-}" && -f "$icd_file" ]]; then
  export VK_DRIVER_FILES="$icd_file"
fi
runtime_root="$policy_dir/.cache/runtime"
export XDG_RUNTIME_DIR="${ROBOMORPHER_XDG_RUNTIME_DIR:-$runtime_root/xdg_${env_gpu_id}}"
export TMPDIR="${ROBOMORPHER_TMPDIR:-$runtime_root/tmp_${env_gpu_id}}"
export XDG_CACHE_HOME="${ROBOMORPHER_XDG_CACHE_HOME:-$runtime_root/cache_${env_gpu_id}}"
mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
echo "[RoboMorpher] connecting to $policy_server_ip:$policy_server_port"
export CUDA_VISIBLE_DEVICES="$env_gpu_id"
additional_info="ckpt_name=$ckpt_name,action_type=$action_type${additional_info:+,$additional_info},xpolicylab_root=$XPOLICYLAB_ROOT"
mode="${EVAL_ENV_TYPE:-sim}"
if ! command -v conda >/dev/null 2>&1 && [[ -x "$eval_env_conda_env/../../bin/conda" ]]; then
  export PATH="$(cd -- "$eval_env_conda_env/../../bin" && pwd):$PATH"
fi
if [[ "$mode" == debug || "$mode" == sim ]]; then
  if [[ "$mode" == sim && "$bench_name" == RoboTwin && -d "$eval_env_conda_env/conda-meta" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$eval_env_conda_env"
    eval_prefix=$CONDA_PREFIX
  elif [[ -x "$eval_env_conda_env/bin/python" ]]; then
    eval_prefix=$(cd -- "$eval_env_conda_env" && pwd)
  else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$eval_env_conda_env"
    eval_prefix=$CONDA_PREFIX
  fi
  if [[ "$mode" == debug ]]; then
    exec "$eval_prefix/bin/python" "$policy_dir/scripts/debug_env_client.py" "$yaml_file" \
      "$bench_root" "$bench_name" "$task_name" "$env_cfg_type" "$policy_server_ip" "$policy_server_port"
  fi
  if [[ "$bench_name" == RoboTwin ]]; then
    exec "$eval_prefix/bin/python" "$policy_dir/scripts/robotwin_eval_client.py" "$bench_root" "$yaml_file" "$env_cfg_type" \
      --bench_name "$bench_name" --task_name "$task_name" \
      --policy_name RoboMorpher --host "$policy_server_ip" --port "$policy_server_port" \
      --root_dir "$bench_root" --device_id "$env_gpu_id" --seed "$seed" --additional_info "$additional_info"
  fi
  export EMBODIBENCH_PYTHON="$eval_prefix/bin/python"
  # Batch workers use the sibling RoboTwin checkout for robot assets.
  if [[ -d "$(dirname "$bench_root")/RoboTwin" ]]; then
    export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(dirname "$bench_root")/RoboTwin}"
  fi
  shopt -s nullglob
  vulkan_real=("$eval_prefix"/lib/python*/site-packages/sapien/vulkan_library/libvulkan.so.1.*)
  if (( ${#vulkan_real[@]} )); then
    vulkan_link_dir="${ROBOMORPHER_VULKAN_DIR:-$policy_dir/.cache/vulkan_${eval_prefix##*/}}"
    mkdir -p "$vulkan_link_dir"
    ln -sfn "${vulkan_real[0]}" "$vulkan_link_dir/libvulkan.so.1"
    export LD_LIBRARY_PATH="$vulkan_link_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  args=()
  if [[ "${ROBOMORPHER_EVAL_BATCH:-false}" == true ]]; then
    args+=(--eval_batch true --num_workers "${ROBOMORPHER_NUM_WORKERS:-1}")
  fi
  if [[ -n "${ROBOMORPHER_MAX_SEED_ATTEMPTS:-}" ]]; then
    args+=(--max_seed_attempts "$ROBOMORPHER_MAX_SEED_ATTEMPTS")
  fi
  if [[ -n "${ROBOMORPHER_FREQUENCY:-}" ]]; then
    args+=(--frequency "$ROBOMORPHER_FREQUENCY")
  fi
  exec bash "$bench_root/scripts/eval_policy.sh" --bench_name "$bench_name" \
    --task_name "$task_name" --env_cfg_type "$env_cfg_type" --policy_name RoboMorpher \
    --ckpt_name "$ckpt_name" --action_type "$action_type" --host "$policy_server_ip" \
    --port "$policy_server_port" --protocol ws --root_dir "$bench_root" \
    --device_id "$env_gpu_id" --seed "$seed" --task_config "${ROBOMORPHER_TASK_CONFIG:-default}" \
    --test_num "${ROBOMORPHER_TEST_NUM:-100}" --expert_check "${ROBOMORPHER_EXPERT_CHECK:-true}" \
    --instruction_type "${ROBOMORPHER_INSTRUCTION_TYPE:-seen}" \
    --instruction_mode "${ROBOMORPHER_INSTRUCTION_MODE:-fixed_task}" \
    --additional_info "$additional_info" "${args[@]}"
fi

exec bash "$XPOLICYLAB_ROOT/utils/setup_env_client.sh" "$XPOLICYLAB_ROOT/utils" "$yaml_file" \
  "$eval_env_conda_env" "$policy_server_port" "$bench_name" "$task_name" "$env_cfg_type" \
  RoboMorpher "$additional_info" "$bench_root" "$seed" "$env_gpu_id" "$policy_server_ip"
