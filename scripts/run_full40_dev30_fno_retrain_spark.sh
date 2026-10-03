#!/usr/bin/env bash
# Frozen-blind official PhysicsNeMo trainer for the immutable dev30 release.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
stage="${1:---dry-run}"
mode="${2:---dry-run}"
requested_run_id="${3:-}"
parent_run_id="${FULL40_DEV30_ONESTEP_RUN_ID:-seed20261003_30epoch}"
approval="${FULL40_DEV30_TRAIN_APPROVAL_TOKEN:-}"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data_host="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"

case "$stage" in
  onestep)
    config="tandem_fno_full40_onestep"
    trainer="train_tandem_fno.py"
    default_run_id="seed20261003_30epoch"
    allocator="0.20"
    ;;
  h20)
    config="tandem_fno_full40_h20"
    trainer="train_tandem_fno_rollout.py"
    default_run_id="seed20261003_10epoch"
    allocator="0.15"
    ;;
  --dry-run)
    python3 scripts/plan_full40_dev30_fno_retrain.py
    exit 0
    ;;
  *) echo "usage: $0 [onestep|h20|--dry-run] [--dry-run|--execute] [run-id]" >&2; exit 2 ;;
esac

run_id="${requested_run_id:-$default_run_id}"
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo "run-id must match ^[a-z0-9][a-z0-9_-]{0,63}$" >&2; exit 2;
}
output_host="$root/artifacts/tandem_fno_full40_dev30_${stage}_${run_id}"
[[ ! -e "$output_host" ]] || {
  echo "refusing existing output; retry with a new run-id: $output_host" >&2; exit 2;
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
  [[ "$parent_run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
    echo "FULL40_DEV30_ONESTEP_RUN_ID is invalid" >&2; exit 2;
  }
  parent_host="$root/artifacts/tandem_fno_full40_dev30_onestep_${parent_run_id}/best"
  [[ -d "$parent_host" ]] || { echo "reviewed dev30 one-step parent is missing" >&2; exit 2; }
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
  printf 'DEV30_FROZEN_BLIND_DRY_RUN_NO_EXECUTION\ncommand:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "second argument must be --dry-run or --execute" >&2; exit 2; }
[[ "$approval" == "EXECUTE_REVIEWED_FULL40_DEV30_FNO_RETRAIN" ]] || {
  echo "explicit reviewed dev30 training token is required" >&2; exit 2;
}
mkdir "$output_host"
exec "${command[@]}"
