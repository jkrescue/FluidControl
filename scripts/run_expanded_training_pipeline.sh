#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
data="data/curated/tandem_cylinders_expanded_v1"
evidence="artifacts/tandem_cylinders"
pipeline_log="$evidence/expanded_training_pipeline.log"
mkdir -p "$evidence"
exec > >(tee -a "$pipeline_log") 2>&1

timestamp() { date '+%F %T %Z'; }

wait_for_finalize() {
    echo "[$(timestamp)] Waiting for expanded normalization and manifest"
    while [[ ! -s "$data/manifest.json" || ! -s "$data/normalization.json" ]]; do
        if ! tmux has-session -t expanded_finalize 2>/dev/null; then
            echo "Expanded finalization ended without manifest/normalization" >&2
            exit 1
        fi
        sleep 15
    done
}

run_dataset_gates() {
    echo "[$(timestamp)] Running full curated-data validation"
    .venv-curator/bin/python scripts/validate_tandem_curated.py \
        --data "$data" --output "$evidence/expanded_curated_validation.json"
    .venv-curator/bin/python scripts/audit_tandem_actions.py \
        --data "$data" --output "$evidence/expanded_action_audit_curated.json"
    echo "[$(timestamp)] Running official PhysicsNeMo DataPipe validation"
    PYTHONPATH=src .venv/bin/python scripts/validate_tandem_datapipe.py \
        --data "$data" --output "$evidence/expanded_datapipe_validation.json"
    PYTHONPATH=src .venv/bin/python scripts/inspect_tandem_model.py \
        conf/tandem_fno_expanded.yaml | tee "$evidence/expanded_model_architecture.json"
}

tune_one_step_batch() {
    local batch output
    for batch in 224 192 160 128 96; do
        output="artifacts/tandem_fno_expanded_smoke_b${batch}"
        rm -rf "$output"
        echo "[$(timestamp)] One-step smoke test with per-GPU batch $batch"
        if OUTPUT_DIR="$output" BATCH_SIZE="$batch" SMOKE=true \
            bash scripts/run_tandem_fno_expanded.sh; then
            echo "$batch" >"$evidence/expanded_one_step_batch_size.txt"
            echo "[$(timestamp)] Selected one-step per-GPU batch $batch"
            return
        fi
    done
    echo "No one-step batch candidate passed" >&2
    exit 1
}

tune_rollout_batch() {
    local batch output
    for batch in 16 12 8 4; do
        output="artifacts/tandem_fno_rollout_expanded_smoke_b${batch}"
        rm -rf "$output"
        echo "[$(timestamp)] Ten-step rollout smoke test with per-GPU batch $batch"
        if OUTPUT_DIR="$output" BATCH_SIZE="$batch" SMOKE=false EPOCHS=1 \
            bash scripts/run_tandem_fno_rollout_expanded.sh \
                training.max_train_batches=2 training.max_validation_batches=2; then
            echo "$batch" >"$evidence/expanded_rollout_batch_size.txt"
            echo "[$(timestamp)] Selected rollout per-GPU batch $batch"
            return
        fi
    done
    echo "No rollout batch candidate passed" >&2
    exit 1
}

if [[ "${RESUME_AFTER_ONE_STEP:-false}" == "true" ]]; then
    echo "[$(timestamp)] Resuming after completed one-step training; preserving completed data gates"
else
    wait_for_finalize
    run_dataset_gates
fi
one_step_epoch="$(python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/tandem_fno_expanded_v1/training_history.json")
epochs = []
if path.exists():
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        epochs.extend(int(record["epoch"]) for record in records)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
print(max(epochs, default=0))
PY
)"
if [[ "$one_step_epoch" -ge 80 && -d artifacts/tandem_fno_expanded_v1/best ]]; then
    echo "[$(timestamp)] One-step training already completed at epoch $one_step_epoch; preserving checkpoints"
else
    tune_one_step_batch
    one_step_batch="$(cat "$evidence/expanded_one_step_batch_size.txt")"
    echo "[$(timestamp)] Starting formal one-step PhysicsNeMo FNO training"
    OUTPUT_DIR=artifacts/tandem_fno_expanded_v1 BATCH_SIZE="$one_step_batch" SMOKE=false \
        bash scripts/run_tandem_fno_expanded.sh
fi

echo "[$(timestamp)] Evaluating formal one-step model"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/evaluate_tandem_fno.py \
    --data "$data" --config conf/tandem_fno_expanded.yaml \
    --checkpoint-dir artifacts/tandem_fno_expanded_v1/best \
    --output artifacts/tandem_fno_expanded_v1/evaluation.json \
    --visualization-dir artifacts/tandem_fno_expanded_v1/rollout_visualizations \
    --horizons 1 10 50 100 --visualizations-per-horizon 3

tune_rollout_batch
rollout_batch="$(cat "$evidence/expanded_rollout_batch_size.txt")"
echo "[$(timestamp)] Starting formal ten-step rollout fine-tuning"
OUTPUT_DIR=artifacts/tandem_fno_rollout_expanded_v1 BATCH_SIZE="$rollout_batch" SMOKE=false \
    bash scripts/run_tandem_fno_rollout_expanded.sh

echo "[$(timestamp)] Evaluating rollout-fine-tuned model"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/evaluate_tandem_fno.py \
    --data "$data" --config conf/tandem_fno_expanded.yaml \
    --checkpoint-dir artifacts/tandem_fno_rollout_expanded_v1/best \
    --output artifacts/tandem_fno_rollout_expanded_v1/evaluation.json \
    --visualization-dir artifacts/tandem_fno_rollout_expanded_v1/rollout_visualizations \
    --horizons 1 10 50 100 --visualizations-per-horizon 3

.venv/bin/python scripts/compare_rollout_evaluations.py \
    artifacts/tandem_fno_expanded_v1/evaluation.json \
    artifacts/tandem_fno_rollout_expanded_v1/evaluation.json \
    --output artifacts/tandem_fno_rollout_expanded_v1/baseline_comparison.json \
    --markdown artifacts/tandem_fno_rollout_expanded_v1/baseline_comparison.md
echo "[$(timestamp)] EXPANDED_TRAINING_PIPELINE_OK"
