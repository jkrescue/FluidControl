#!/usr/bin/env bash
# Evaluate exactly one reviewed full40 checkpoint on validation10 only.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
checkpoint="${CHECKPOINT_DIR:-artifacts/tandem_fno_full40_h20_seed20261003_10epoch/best}"
output="${VALIDATION_OUTPUT:-artifacts/tandem_cylinders/full40_fno_validation_pending_review}"
data="data/curated/tandem_cylinders_matched_start_full40_v1"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"

[[ "$checkpoint" == artifacts/tandem_fno_full40_* && "$output" == artifacts/* ]] || {
  echo "checkpoint/output must remain in reviewed project artifact roots" >&2; exit 2;
}
[[ ! -e "$output" ]] || { echo "refusing existing validation output" >&2; exit 2; }
plan="$(python3 scripts/plan_full40_fno_retrain.py)"
python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "FULL40_FNO_RETRAIN_READY_FOR_REVIEWED_RUN"' <<<"$plan" || {
  echo "$plan" >&2; exit 2;
}
[[ -d "$checkpoint" ]] || { echo "checkpoint directory is missing" >&2; exit 2; }
[[ "$(find "$checkpoint" -maxdepth 1 -name 'FNO.*.mdlus' | wc -l)" -eq 1 ]] || {
  echo "checkpoint view must contain exactly one FNO model generation" >&2; exit 2;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$expected_image" ]] || {
  echo "PhysicsNeMo image ID mismatch" >&2; exit 2;
}

eval_command=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g
  --user "$(id -u):$(id -g)" --env HOME=/tmp
  --env PYTHONPATH=/workspace/src:/workspace/scripts
  --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace "$image"
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction 0.15
  --margin-gib 4 -- python -u scripts/evaluate_tandem_fno.py
  --data "$data" --normalization-data "$data"
  --config conf/tandem_fno_full40_h20.yaml --checkpoint-dir "$checkpoint"
  --split validation --horizons 1 10 50 100 --segment-stride 25
  --evaluation-batch-size 4 --action-mode observed
  --visualizations-per-horizon 0 --output "$output/evaluation.json"
  --segment-metrics-output "$output/segments.json")

if [[ "$mode" == "--dry-run" ]]; then
  printf 'VALIDATION10_ONLY_NO_FROZEN_NO_EXECUTION\ncommand:'
  printf ' %q' "${eval_command[@]}"
  printf '\n'
  exit 0
fi
[[ "$mode" == "--execute" && "${FULL40_VALIDATION_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_FULL40_VALIDATION" ]] || {
  echo "reviewed validation approval token is required" >&2; exit 2;
}
mkdir -p "$output"
"${eval_command[@]}"
python3 scripts/audit_full40_validation_gate.py \
  --report "$output/evaluation.json" --segments "$output/segments.json" \
  --predeclaration artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json \
  --output "$output/gate.json"
