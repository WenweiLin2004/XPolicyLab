#!/usr/bin/env bash
# The adapter can be symlinked into a different workspace from the simulators.
resolve_xpolicylab_paths() {
  local policy_dir="$1" bench="$2" physical_parent logical_xpl workspace candidate configured_bench
  physical_parent=$(dirname "$(cd -P -- "$policy_dir" && pwd -P)") || return 1
  logical_xpl=$(dirname "$(dirname "$policy_dir")")
  candidate=${XPOLICYLAB_ROOT:-$logical_xpl}
  workspace=$(dirname "$candidate")
  case "$bench" in
    EmbodiBench) configured_bench="${EMBODIBENCH_ROOT:-}" ;;
    RoboTwin)    configured_bench="${ROBOTWIN_ROOT:-}" ;;
    *) echo "[RoboMorpher] unknown benchmark: $bench" >&2; return 2 ;;
  esac

  if [[ -n "$configured_bench" ]]; then
    bench_root=$configured_bench
  elif [[ "$(basename "$workspace")" == "$bench" && -d "$workspace" ]]; then
    bench_root=$workspace
  elif [[ -d "$workspace/$bench" ]]; then
    bench_root=$workspace/$bench
  elif [[ "$(basename "$physical_parent")" == "$bench" ]]; then
    bench_root=$physical_parent
  elif [[ -d "$physical_parent/$bench" ]]; then
    bench_root=$physical_parent/$bench
  else
    bench_root=$workspace/$bench
  fi
  if [[ ! -d "$bench_root" ]]; then
    echo "[RoboMorpher] cannot find $bench at $bench_root. Set EMBODIBENCH_ROOT or ROBOTWIN_ROOT." >&2
    return 1
  fi
  bench_root=$(cd -- "$bench_root" && pwd -P) || return 1
  if [[ -z "${XPOLICYLAB_ROOT:-}" && ! -f "$candidate/setup_policy_server.py" ]]; then
    candidate=$bench_root/XPolicyLab
    if [[ ! "$candidate/policy/RoboMorpher" -ef "$policy_dir" ]]; then
      echo "[RoboMorpher] run through XPolicyLab/policy/RoboMorpher or set XPOLICYLAB_ROOT to your checkout." >&2
      return 1
    fi
  fi
  if [[ ! -f "$candidate/setup_policy_server.py" ]]; then
    echo "[RoboMorpher] XPolicyLab entry point missing under $candidate" >&2
    return 1
  fi
  XPOLICYLAB_ROOT=$(cd -- "$candidate" && pwd -L) || return 1
  export XPOLICYLAB_ROOT
  export PYTHONPATH="$(dirname "$XPOLICYLAB_ROOT"):$XPOLICYLAB_ROOT:$policy_dir:$(dirname "$bench_root"):$bench_root${PYTHONPATH:+:$PYTHONPATH}"
}
