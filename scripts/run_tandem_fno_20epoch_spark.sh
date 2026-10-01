#!/usr/bin/env bash
# Longer independent-v2 FNO study on GPU0; no host environment mutation.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
output="artifacts/tandem_fno_expanded_spark_20epoch"
log_dir="artifacts/tandem_cylinders"
mkdir -p "${log_dir}"
if [[ -e "${output}/training_history.json" ]]; then
    echo "Refusing to overwrite an existing 20-epoch run" >&2
    exit 2
fi
[[ -s data/curated/tandem_cylinders_expanded_independent_v2/manifest.json ]] || exit 1
python3 scripts/audit_tandem_split_integrity.py \
    --data data/curated/tandem_cylinders_expanded_independent_v2 \
    --cases-root cfd/tandem_cylinders/cases \
    --output "${log_dir}/expanded_split_integrity_20epoch.json"
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < 150 )); then
    echo "Only ${free_gib} GiB disk free; refusing to train" >&2
    exit 75
fi

EPOCHS=20 BATCH_SIZE=64 OUTPUT_DIR="${output}" \
    bash scripts/run_tandem_fno_spark.sh 2>&1 |
    tee "${log_dir}/expanded_fno_20epoch_spark.log"
python3 scripts/validate_tandem_fno_stage.py \
    --training-history "${output}/training_history.json" --min-epoch 20
echo PHYSICSNEMO_SPARK_20EPOCH_OK

MODEL_DIR="${output}" bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee "${log_dir}/expanded_fno_20epoch_heldout_spark.log"
python3 scripts/validate_tandem_fno_stage.py \
    --evaluation "${output}/heldout_evaluation.json" --action-mode observed
echo PHYSICSNEMO_SPARK_20EPOCH_HELDOUT_OK

MODEL_DIR="${output}" ACTION_MODE=zero VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee "${log_dir}/expanded_fno_20epoch_action_zero_spark.log"
MODEL_DIR="${output}" ACTION_MODE=sign_flip VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee "${log_dir}/expanded_fno_20epoch_action_sign_flip_spark.log"
python3 scripts/validate_tandem_fno_stage.py \
    --evaluation "${output}/heldout_evaluation_zero.json" --action-mode zero
python3 scripts/validate_tandem_fno_stage.py \
    --evaluation "${output}/heldout_evaluation_sign_flip.json" --action-mode sign_flip
python3 scripts/summarize_tandem_action_sensitivity.py \
    "${output}/heldout_evaluation.json" \
    "${output}/heldout_evaluation_zero.json" \
    "${output}/heldout_evaluation_sign_flip.json" \
    --output "${output}/action_sensitivity_summary.json" \
    --markdown "${output}/action_sensitivity_summary.md"
echo PHYSICSNEMO_SPARK_20EPOCH_ACTION_ABLATIONS_OK
