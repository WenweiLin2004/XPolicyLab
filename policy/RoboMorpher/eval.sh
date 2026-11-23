#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_path> <eval_env> [task_config]" >&2
  echo "  bench_name: EmbodiBench or RoboTwin" >&2
  echo "  env_cfg:    benchmark environment configuration" >&2
  echo "  action_type: joint or qpos" >&2
  exit 2
}

if [[ $# -lt 10 || $# -gt 11 ]]; then
  usage
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env=${10}
task_config_arg=${11:-}

case "$bench_name" in
  EmbodiBench|RoboTwin) ;;
  *) echo "[RoboMorpher] bench_name must be EmbodiBench or RoboTwin" >&2; exit 2 ;;
esac
if [[ "$action_type" != "joint" && "$action_type" != "qpos" ]]; then
  echo "[RoboMorpher] action_type must be joint or qpos" >&2
  exit 2
fi

policy_dir="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
source "$policy_dir/scripts/xpolicylab_paths.sh"
resolve_xpolicylab_paths "$policy_dir" "$bench_name"
if [[ "$bench_name" == "EmbodiBench" ]]; then
  task_config=${task_config_arg:-${ROBOMORPHER_TASK_CONFIG:-default}}
  episodes=${ROBOMORPHER_TEST_NUM:-100}
  host=${ROBOMORPHER_SERVER_HOST:-localhost}
  if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RoboMorpher] ROBOMORPHER_TEST_NUM must be a positive integer, got: $episodes" >&2
    exit 2
  fi

  port="${ROBOMORPHER_SERVER_PORT:-$(bash "$XPOLICYLAB_ROOT/utils/get_free_port.sh")}"
  server_log="$(mktemp -t robomorpher_eval_XXXXXX.log)"
  server_pid=""
  cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$server_log"
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  echo "[RoboMorpher] EmbodiBench task=$task_name embodiment=$env_cfg_type task_config=$task_config episodes=$episodes"
  echo "[RoboMorpher] policy_gpu=$policy_gpu_id env_gpu=$env_gpu_id server=$host:$port"
  bash "$policy_dir/setup_eval_policy_server.sh" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$policy_gpu_id" "$policy_conda_env" "$port" "$host" >"$server_log" 2>&1 &
  server_pid=$!
  if ! bash "$XPOLICYLAB_ROOT/utils/wait_for_policy_server.sh" "$host" "$port" "$server_pid" "RoboMorpher policy server" "${ROBOMORPHER_SERVER_WAIT_S:-1200}"; then
    tail -80 "$server_log" >&2
    exit 1
  fi

  additional_info="ckpt_name=$ckpt_name,action_type=$action_type,expert_check=${ROBOMORPHER_EXPERT_CHECK:-true}"
  if [[ -n "${ROBOMORPHER_FREQUENCY:-}" ]]; then additional_info+=",frequency=$ROBOMORPHER_FREQUENCY"; fi
  if [[ -n "${ROBOMORPHER_MAX_SEED_ATTEMPTS:-}" ]]; then additional_info+=",max_seed_attempts=$ROBOMORPHER_MAX_SEED_ATTEMPTS"; fi

  ROBOMORPHER_TASK_CONFIG="$task_config" \
    bash "$policy_dir/setup_eval_env_client.sh" "$bench_name" "$task_name" "$ckpt_name" \
      "$env_cfg_type" "$action_type" "$seed" "$env_gpu_id" "$eval_env" "$additional_info" "$port" "$host"
  echo "[RoboMorpher] EmbodiBench evaluation finished"
  exit 0
fi

task_config=${task_config_arg:-${ROBOMORPHER_TASK_CONFIG:-demo_clean}}
export ROBOMORPHER_TASK_CONFIG="$task_config"
policy_server_ip=${ROBOMORPHER_SERVER_HOST:-localhost}
max_restarts=${ROBOMORPHER_EXPERT_RESTARTS:-10}
if ! [[ "$max_restarts" =~ ^[1-9][0-9]*$ ]]; then
  echo "[RoboMorpher] ROBOMORPHER_EXPERT_RESTARTS must be a positive integer, got: $max_restarts" >&2
  exit 2
fi

base_info="ckpt_name=$ckpt_name,action_type=$action_type"
if [[ -n "${ROBOMORPHER_TEST_NUM:-}" ]]; then base_info+=",test_num=${ROBOMORPHER_TEST_NUM}"; fi
if [[ -n "${ROBOMORPHER_INSTRUCTION_TYPE:-}" ]]; then base_info+=",instruction_type=${ROBOMORPHER_INSTRUCTION_TYPE}"; fi
if [[ -n "${ROBOMORPHER_EXPERT_CHECK:-}" ]]; then
  case "${ROBOMORPHER_EXPERT_CHECK,,}" in
    0|false|no|off) base_info+=",expert_check=false" ;;
    1|true|yes|on) base_info+=",expert_check=true" ;;
    *) echo "[RoboMorpher] invalid ROBOMORPHER_EXPERT_CHECK=${ROBOMORPHER_EXPERT_CHECK}; use true/false" >&2; exit 2 ;;
  esac
fi
if [[ -n "${ROBOMORPHER_MAX_SEED_ATTEMPTS:-}" ]]; then
  if ! [[ "${ROBOMORPHER_MAX_SEED_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RoboMorpher] ROBOMORPHER_MAX_SEED_ATTEMPTS must be a positive integer, got: ${ROBOMORPHER_MAX_SEED_ATTEMPTS}" >&2
    exit 2
  fi
  base_info+=",max_seed_attempts=${ROBOMORPHER_MAX_SEED_ATTEMPTS}"
fi

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
log_file=""
cleanup() {
  cleanup_server
  if [[ -n "$log_file" ]]; then rm -f -- "$log_file"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${ROBOMORPHER_SERVER_PORT:-}" ]]; then
  policy_server_port="${ROBOMORPHER_SERVER_PORT}"
else
  policy_server_port=$(bash "$XPOLICYLAB_ROOT/utils/get_free_port.sh")
fi
log_file=$(mktemp -t "robomorpher_eval_${task_name//[^a-zA-Z0-9]/_}_${seed}_XXXXXX.log")
additional_info="$base_info,expert_restart_attempt=1,expert_restart_limit=$max_restarts"
echo "[RoboMorpher] RoboTwin seed=$seed task=$task_name embodiment=$env_cfg_type policy_gpu=$policy_gpu_id env_gpu=$env_gpu_id port=$policy_server_port"

cleanup_server
bash "$policy_dir/setup_eval_policy_server.sh" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$policy_gpu_id" "$policy_conda_env" "$policy_server_port" "$policy_server_ip" >"$log_file" 2>&1 &
SERVER_PID=$!

set +e
bash "$XPOLICYLAB_ROOT/utils/wait_for_policy_server.sh" "$policy_server_ip" "$policy_server_port" "$SERVER_PID" "RoboMorpher policy server" "${ROBOMORPHER_SERVER_WAIT_S:-1200}" >>"$log_file" 2>&1
wait_status=$?
if [[ $wait_status -eq 0 ]]; then
  echo "[RoboMorpher] environment client gpu=$env_gpu_id config=$task_config"
  eval_timeout="${ROBOMORPHER_EVAL_TIMEOUT_SECONDS:-0}"
  if [[ "$eval_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RoboMorpher] client timeout=${eval_timeout}s"
    timeout --signal=TERM --kill-after=30s "${eval_timeout}s" \
      bash "$policy_dir/setup_eval_env_client.sh" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$env_gpu_id" "$eval_env" "$additional_info" "$policy_server_port" "$policy_server_ip" 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
  else
    bash "$policy_dir/setup_eval_env_client.sh" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$env_gpu_id" "$eval_env" "$additional_info" "$policy_server_port" "$policy_server_ip" 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
  fi
  if [[ $status -eq 0 ]] && grep -Eiq '(^|[^[:alpha:]])Render Error([^[:alpha:]]|$)' "$log_file"; then
    echo "[RoboMorpher] renderer reported Render Error; treating evaluation as failed" >&2
    status=1
  fi
else
  status=$wait_status
fi
set -e
cleanup_server

if [[ $status -eq 0 ]]; then
  rm -f "$log_file"
  echo "[RoboMorpher] evaluation finished"
  exit 0
fi
echo "[RoboMorpher] evaluation failed (exit=$status)" >&2
tail -80 "$log_file" >&2 || true
rm -f "$log_file"
exit "$status"
