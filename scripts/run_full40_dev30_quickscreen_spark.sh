#!/usr/bin/env bash
# Frozen-blind 10-epoch one-step + 5-epoch H20 development quick-screen.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
stage="${1:---dry-run}"
mode="${2:---dry-run}"
run_id="${3:-}"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data_host="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"

case "$stage" in
  onestep)
    config="tandem_fno_full40_quickscreen_onestep"
    trainer="train_tandem_fno.py"
    allocator="0.20"
    ;;
  h20)
    config="tandem_fno_full40_quickscreen_h20"
    trainer="train_tandem_fno_rollout.py"
    allocator="0.15"
    ;;
  --dry-run)
    python3 scripts/plan_full40_dev30_fno_retrain.py
    exit 0
    ;;
  *) echo "usage: $0 [onestep|h20|--dry-run] [--dry-run|--execute] [run-id]" >&2; exit 2 ;;
esac

[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo "a unique run-id matching ^[a-z0-9][a-z0-9_-]{0,63}$ is required" >&2; exit 2;
}
output_host="$root/artifacts/tandem_fno_full40_dev30_quickscreen_${stage}_${run_id}"
[[ ! -e "$output_host" ]] || {
  echo "refusing existing quick-screen output; retry with a new run-id" >&2; exit 2;
}

plan="$(python3 scripts/plan_full40_dev30_fno_retrain.py)"
python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "FULL40_DEV30_FNO_RETRAIN_READY"' <<<"$plan" || {
  echo "$plan" >&2; exit 2;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$expected_image" ]] || {
  echo "PhysicsNeMo image ID mismatch" >&2; exit 2;
}

mounts=(
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$data_host,dst=/workspace/devdata,readonly"
  --mount "type=bind,src=$output_host,dst=/workspace/output"
)
overrides=("data.root=/workspace/devdata" "output_dir=/workspace/output")
if [[ "$stage" == "h20" ]]; then
  parent_run_id="${FULL40_DEV30_QUICKSCREEN_ONESTEP_RUN_ID:-}"
  parent_sha="${FULL40_DEV30_QUICKSCREEN_PARENT_SHA256:-}"
  [[ "$parent_run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
    echo "reviewed quick-screen one-step parent run-id is required" >&2; exit 2;
  }
  [[ "$parent_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "exact reviewed quick-screen parent SHA-256 is required" >&2; exit 2;
  }
  parent_host="$root/artifacts/tandem_fno_full40_dev30_quickscreen_onestep_${parent_run_id}/best"
  [[ -d "$parent_host" ]] || { echo "reviewed quick-screen parent is missing" >&2; exit 2; }
  parent_root="${parent_host%/best}"
  python3 - "$parent_root/training_history.json" "$parent_root/resolved_config.yaml" <<'PY'
import json
import sys
from pathlib import Path

import yaml

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
if [row.get("epoch") for row in history] != list(range(1, 11)):
    raise SystemExit("quick-screen one-step parent is not a completed 10-epoch run")
if (
    config.get("training", {}).get("epochs") != 10
    or config.get("training", {}).get("seed") != 20261003
    or config.get("data", {}).get("root") != "/workspace/devdata"
):
    raise SystemExit("quick-screen one-step resolved config differs")
PY
  mapfile -t parent_models < <(find "$parent_host" -maxdepth 1 -type f -name 'FNO.*.mdlus' -print)
  [[ "${#parent_models[@]}" -eq 1 ]] || {
    echo "parent must contain exactly one PhysicsNeMo FNO generation" >&2; exit 2;
  }
  actual_parent_sha="$(sha256sum "${parent_models[0]}" | awk '{print $1}')"
  [[ "$actual_parent_sha" == "$parent_sha" ]] || {
    echo "quick-screen parent FNO SHA-256 mismatch" >&2; exit 2;
  }
  parent_host="$(realpath -e "$parent_host")"
  [[ "$parent_host" == "$root"/artifacts/tandem_fno_full40_dev30_quickscreen_onestep_*/best ]] || {
    echo "resolved parent escapes quick-screen artifact root" >&2; exit 2;
  }
  mounts+=(--mount "type=bind,src=$parent_host,dst=/workspace/parent,readonly")
  overrides+=("training.initial_checkpoint=/workspace/parent")
fi

command=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g
  --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src:/workspace/scripts
  --env OMP_NUM_THREADS=4 "${mounts[@]}" --workdir /workspace "$image"
  python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20
  --allocator-fraction "$allocator" --margin-gib 4 --
  python -u "/workspace/scripts/$trainer" --config-name "$config"
  hydra.run.dir=/tmp/hydra hydra.output_subdir=null "${overrides[@]}")

if [[ "$mode" == "--dry-run" ]]; then
  printf 'DEV30_QUICKSCREEN_STAGE_CANDIDATE_ONLY_NO_FROZEN_NO_EXECUTION\ncommand:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "$stage" == "h20" ]]; then
    printf 'fixed_parent_sha256=%s\n' "$actual_parent_sha"
  fi
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "second argument must be --dry-run or --execute" >&2; exit 2; }
[[ "${FULL40_DEV30_QUICKSCREEN_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_DEV30_QUICKSCREEN" ]] || {
  echo "explicit reviewed quick-screen token is required" >&2; exit 2;
}
mkdir "$output_host"
if [[ "$stage" == "h20" ]]; then
  printf '%s  %s\n' "$actual_parent_sha" "${parent_models[0]##*/}" > "$output_host/parent_fno_sha256.txt"
fi
exec "${command[@]}"
