#!/usr/bin/env bash
# Review-gated launcher; dry-run only until the full40 profile and plan pass.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
stage="${1:---dry-run}"
mode="${2:---dry-run}"
requested_run_id="${3:-}"
onestep_parent_run_id="${FULL40_ONESTEP_RUN_ID:-seed20261003_30epoch}"
approval="${FULL40_TRAIN_APPROVAL_TOKEN:-}"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data="data/curated/tandem_cylinders_matched_start_full40_v1"

case "$stage" in
  onestep)
    config="tandem_fno_full40_onestep"
    trainer="scripts/train_tandem_fno.py"
    default_run_id="seed20261003_30epoch"
    allocator="0.20"
    ;;
  h20)
    config="tandem_fno_full40_h20"
    trainer="scripts/train_tandem_fno_rollout.py"
    default_run_id="seed20261003_10epoch"
    allocator="0.15"
    ;;
  --dry-run)
    python3 scripts/plan_full40_fno_retrain.py
    exit 0
    ;;
  *) echo "usage: $0 [onestep|h20|--dry-run] [--dry-run|--execute] [run-id]" >&2; exit 2 ;;
esac

run_id="${requested_run_id:-$default_run_id}"
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo "run-id must match ^[a-z0-9][a-z0-9_-]{0,63}$" >&2; exit 2;
}
output="artifacts/tandem_fno_full40_${stage}_${run_id}"
[[ ! -e "$output" ]] || {
  echo "refusing existing output path; retry with a new independent run-id: $output" >&2
  exit 2
}
overrides=("data.root=$data" "output_dir=$output")
if [[ "$stage" == "h20" ]]; then
  [[ "$onestep_parent_run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
    echo "FULL40_ONESTEP_RUN_ID is invalid" >&2; exit 2;
  }
  onestep_parent="artifacts/tandem_fno_full40_onestep_${onestep_parent_run_id}"
  [[ -d "$onestep_parent/best" ]] || {
    echo "reviewed one-step best checkpoint is required before H20: $onestep_parent" >&2
    exit 2
  }
  overrides+=("training.initial_checkpoint=$onestep_parent/best")
fi

plan="$(python3 scripts/plan_full40_fno_retrain.py)"
python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "FULL40_FNO_RETRAIN_READY_FOR_REVIEWED_RUN"' <<<"$plan" || {
  echo "$plan" >&2; exit 2;
}
actual_image="$(docker image inspect "$image" --format '{{.Id}}')"
[[ "$actual_image" == "$expected_image" ]] || { echo "PhysicsNeMo image ID mismatch" >&2; exit 2; }
command=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g
  --shm-size 2g --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace "$image"
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction "$allocator"
  --margin-gib 4 -- python -u "$trainer" --config-name "$config"
  "${overrides[@]}")

if [[ "$mode" == "--dry-run" ]]; then
  printf 'REVIEWED_DATA_READY_BUT_NO_TRAINING_STARTED\ncommand:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "second argument must be --dry-run or --execute" >&2; exit 2; }
[[ "$approval" == "EXECUTE_REVIEWED_FULL40_FNO_RETRAIN" ]] || {
  echo "explicit reviewed approval token is required" >&2; exit 2;
}
mkdir -p "$output"
exec "${command[@]}"
