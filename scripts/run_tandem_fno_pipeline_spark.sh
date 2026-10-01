#!/usr/bin/env bash
# Wait for the validated real CFD dataset, then exercise the official PhysicsNeMo path.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
curator_log="artifacts/tandem_cylinders/expanded_spark_curator.log"
dataset="data/curated/tandem_cylinders_expanded_v1"
audit="artifacts/tandem_cylinders/expanded_datapipe_validation_spark.json"

while ! rg -q '^EXPANDED_SPARK_CURATOR_OK$' "${curator_log}" 2>/dev/null; do
    if ! tmux has-session -t fluid-control-curator-full 2>/dev/null; then
        echo "Curator stopped before completing the full dataset" >&2
        exit 1
    fi
    sleep 60
done
[[ -s "${dataset}/manifest.json" && -s "${dataset}/normalization.json" ]] || exit 1
mapfile -t names < <(python3 cfd/tandem_cylinders/make_expanded_control_dataset.py --list)
[[ "${#names[@]}" -eq 32 ]] || exit 1
.venv-curator-py312/bin/python scripts/validate_expanded_curated_cases.py \
    --data "${dataset}" --cases-root cfd/tandem_cylinders/cases "${names[@]}"
echo EXPANDED_LABEL_AUDIT_OK

free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < 150 )); then
    echo "Only ${free_gib} GiB disk free; refusing to train" >&2
    exit 75
fi

docker run --rm --network none --cpus 4 --memory 16g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/src \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace fluid-control-physicsnemo:2.2.2 \
    python -u scripts/validate_tandem_datapipe.py \
      --data "${dataset}" --output "${audit}"
[[ -s "${audit}" ]] || exit 1

SMOKE=true BATCH_SIZE=64 OUTPUT_DIR=artifacts/tandem_fno_expanded_spark_smoke \
    bash scripts/run_tandem_fno_spark.sh 2>&1 |
    tee artifacts/tandem_cylinders/expanded_fno_smoke_spark.log
[[ -s artifacts/tandem_fno_expanded_spark_smoke/training_history.json ]] || exit 1
python3 scripts/validate_tandem_fno_stage.py --training-history artifacts/tandem_fno_expanded_spark_smoke/training_history.json --min-epoch 1

echo PHYSICSNEMO_SPARK_SMOKE_OK

EPOCHS=5 BATCH_SIZE=64 OUTPUT_DIR=artifacts/tandem_fno_expanded_spark_5epoch \
    bash scripts/run_tandem_fno_spark.sh 2>&1 |
    tee artifacts/tandem_cylinders/expanded_fno_5epoch_spark.log
python3 scripts/validate_tandem_fno_stage.py --training-history artifacts/tandem_fno_expanded_spark_5epoch/training_history.json --min-epoch 5
echo PHYSICSNEMO_SPARK_5EPOCH_OK

bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee artifacts/tandem_cylinders/expanded_fno_heldout_spark.log
python3 scripts/validate_tandem_fno_stage.py --evaluation artifacts/tandem_fno_expanded_spark_5epoch/heldout_evaluation.json --action-mode observed
echo PHYSICSNEMO_SPARK_HELDOUT_OK

ACTION_MODE=zero VISUALIZATIONS_PER_HORIZON=0 bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee artifacts/tandem_cylinders/expanded_fno_action_zero_spark.log
ACTION_MODE=sign_flip VISUALIZATIONS_PER_HORIZON=0 bash scripts/run_tandem_fno_eval_spark.sh 2>&1 |
    tee artifacts/tandem_cylinders/expanded_fno_action_sign_flip_spark.log
python3 scripts/validate_tandem_fno_stage.py --evaluation artifacts/tandem_fno_expanded_spark_5epoch/heldout_evaluation_zero.json --action-mode zero
python3 scripts/validate_tandem_fno_stage.py --evaluation artifacts/tandem_fno_expanded_spark_5epoch/heldout_evaluation_sign_flip.json --action-mode sign_flip
echo PHYSICSNEMO_SPARK_ACTION_ABLATIONS_OK
