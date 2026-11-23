#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash process_data.sh <bench> [<ckpt_name> [<action_type>]]
       bash process_data.sh <bench> <ckpt_name> <env_cfg> <action_type>
  bench        RoboTwin or EmbodiBench
  ckpt_name    Dataset and policy run name (default: vla_robomorpher)
  env_cfg      Accepted for XPolicyLab compatibility. All matching embodiments are exported.
  action_type  joint (default)
EOF
}
fail() {
  echo "$1" >&2
  usage >&2
  exit 2
}
if (( $# == 1 )) && [[ $1 == -h || $1 == --help ]]; then
  usage
  exit 0
fi
(( $# >= 1 && $# <= 4 )) || fail "Unexpected number of positional arguments."
case "${1,,}" in
  robotwin) bench=RoboTwin ;;
  embodibench) bench=EmbodiBench ;;
  *) fail "Unsupported benchmark: $1" ;;
esac
ckpt_name=${2-vla_robomorpher}
if (( $# == 4 )); then
  [[ -n $3 && $3 != -* && $3 != *=* ]] || fail "Expected an environment configuration before the action type."
  action_type=$4
else
  action_type=${3-joint}
fi
[[ $ckpt_name != -* && $ckpt_name =~ ^[[:alnum:]_.-]+$ ]] || fail "Invalid checkpoint name: $ckpt_name"
[[ $action_type == joint ]] || fail "Unsupported action type: $action_type"

policy_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$policy_dir/scripts/xpolicylab_paths.sh"
resolve_xpolicylab_paths "$policy_dir" "$bench"
if [[ $bench == RoboTwin ]]; then export ROBOTWIN_ROOT="$bench_root"; else export EMBODIBENCH_ROOT="$bench_root"; fi
source_dir=${ROBOMORPHER_RAW_DATA:-}
if [[ -z "$source_dir" ]]; then
  source_dir=$(dirname "$XPOLICYLAB_ROOT")/data/$bench
  if [[ ! -d "$source_dir" ]]; then source_dir=$bench_root/data; fi
fi
if [[ ! -d "$source_dir" ]]; then
  echo "Raw $bench dataset missing at $source_dir; set ROBOMORPHER_RAW_DATA to its root." >&2
  exit 1
fi
source_dir=$(cd -L -- "$source_dir" && pwd -L)
output=$(realpath -m "${EXPORT_DATA_PATH:-$policy_dir/data/$bench-$ckpt_name-$action_type}")
cd "$policy_dir"
export PYTHONPATH="$policy_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "${TRAIN_PYTHON:-python3}" -m "robomorpher.utils.data.process_data.convert_${bench,,}_hdf5_to_lerobot" \
  --input-dir "$source_dir" --output-dir "$output"
