#!/usr/bin/env bash
# Frozen-blind H1/H10/H50/H100 validation10 diagnostic; never a formal Gate.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
run_id="${2:-dev30_validation_diagnostic}"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data_host="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
checkpoint_relative="${CHECKPOINT_DIR:-artifacts/tandem_fno_full40_dev30_h20_seed20261003_10epoch/best}"
output_relative="${DEV30_VALIDATION_OUTPUT:-artifacts/tandem_cylinders/full40_dev30_validation_${run_id}}"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo "run-id must match ^[a-z0-9][a-z0-9_-]{0,63}$" >&2; exit 2;
}
case "$checkpoint_relative" in
  artifacts/tandem_fno_full40_dev30_h20_*/best)
    candidate_kind="dev30_h20_development"
    ;;
  artifacts/tandem_fno_full40_dev30_quickscreen_h20_*/best)
    candidate_kind="dev30_quickscreen_h20_stage_candidate"
    ;;
  *) echo "checkpoint must be a reviewed dev30 H20 or quick-screen H20 best directory" >&2; exit 2 ;;
esac
[[ "$output_relative" == artifacts/tandem_cylinders/full40_dev30_validation_* ]] || {
  echo "output must remain in the dedicated dev30 validation artifact root" >&2; exit 2;
}
checkpoint_host="$root/$checkpoint_relative"
output_host="$root/$output_relative"
[[ ! -e "$output_host" ]] || {
  echo "refusing existing diagnostic output; retry with a new run-id" >&2; exit 2;
}
[[ -d "$checkpoint_host" ]] || { echo "dev30 H20 checkpoint directory is missing" >&2; exit 2; }
checkpoint_host="$(realpath -e "$checkpoint_host")"
output_host="$(realpath -m "$output_host")"
case "$checkpoint_host" in
  "$root"/artifacts/tandem_fno_full40_dev30_h20_*/best) ;;
  "$root"/artifacts/tandem_fno_full40_dev30_quickscreen_h20_*/best) ;;
  *) echo "resolved checkpoint escapes reviewed dev30 H20 artifact roots" >&2; exit 2 ;;
esac
[[ "$output_host" == "$root"/artifacts/tandem_cylinders/full40_dev30_validation_* ]] || {
  echo "resolved output escapes the dedicated dev30 validation artifact root" >&2; exit 2;
}

plan="$(python3 scripts/plan_full40_dev30_fno_retrain.py)"
python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "FULL40_DEV30_FNO_RETRAIN_READY"' <<<"$plan" || {
  echo "$plan" >&2; exit 2;
}
[[ "$(find "$checkpoint_host" -maxdepth 1 -name 'FNO.*.mdlus' | wc -l)" -eq 1 ]] || {
  echo "checkpoint must contain exactly one PhysicsNeMo FNO model generation" >&2; exit 2;
}
if [[ "$candidate_kind" == "dev30_quickscreen_h20_stage_candidate" ]]; then
  checkpoint_run="${checkpoint_host%/best}"
  python3 - "$checkpoint_run/training_history.json" "$checkpoint_run/resolved_config.yaml" <<'PY'
import json
import sys
from pathlib import Path

import yaml

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
if [row.get("epoch") for row in history] != list(range(1, 6)):
    raise SystemExit("quick-screen H20 candidate is not a completed five-epoch run")
if (
    config.get("training", {}).get("epochs") != 5
    or config.get("training", {}).get("seed") != 20261003
    or config.get("data", {}).get("root") != "/workspace/devdata"
):
    raise SystemExit("quick-screen H20 resolved config differs")
PY
fi
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$expected_image" ]] || {
  echo "PhysicsNeMo image ID mismatch" >&2; exit 2;
}

evaluate=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g
  --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src:/workspace/scripts
  --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$data_host,dst=/workspace/devdata,readonly"
  --mount "type=bind,src=$checkpoint_host,dst=/workspace/checkpoint,readonly"
  --mount "type=bind,src=$output_host,dst=/workspace/output"
  --workdir /workspace "$image"
  python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20
  --allocator-fraction 0.15 --margin-gib 4 --
  python -u /workspace/scripts/evaluate_tandem_fno.py
  --data /workspace/devdata --normalization-data /workspace/devdata
  --config /workspace/conf/tandem_fno_full40_h20.yaml
  --checkpoint-dir /workspace/checkpoint --split validation
  --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4
  --action-mode observed --visualizations-per-horizon 0
  --output /workspace/output/evaluation.json
  --segment-metrics-output /workspace/output/segments.json)

if [[ "$mode" == "--dry-run" ]]; then
  printf 'DEV30_VALIDATION10_DIAGNOSTIC_ONLY_NO_FROZEN_NO_EXECUTION\ncommand:'
  printf ' %q' "${evaluate[@]}"
  printf '\n'
  printf 'candidate_kind=%s formal_gate=false ppo_authorized=false\n' "$candidate_kind"
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "first argument must be --dry-run or --execute" >&2; exit 2; }
[[ "${DEV30_VALIDATION_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_DEV30_VALIDATION_DIAGNOSTIC" ]] || {
  echo "explicit reviewed dev30 diagnostic token is required" >&2; exit 2;
}
mkdir "$output_host"
"${evaluate[@]}"
python3 scripts/audit_dev30_validation_diagnostic.py \
  --report "$output_host/evaluation.json" \
  --segments "$output_host/segments.json" \
  --data "$data_host" --checkpoint-dir "$checkpoint_host" \
  --candidate-kind "$candidate_kind" \
  --output "$output_host/diagnostic.json"
