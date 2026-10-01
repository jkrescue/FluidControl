#!/usr/bin/env bash
# Wait for the live FNO pipeline; gate its held-out results before CPU-only RL.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model="artifacts/tandem_fno_expanded_spark_20epoch"
pipeline_log="artifacts/tandem_cylinders/expanded_fno_20epoch_pipeline_spark.log"
readiness="${model}/control_readiness.json"
pilot="artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark"
[[ ! -e "${readiness}" && ! -e "${pilot}" ]] || { echo "Readiness or pilot already exists; refusing to overwrite" >&2; exit 2; }

while tmux has-session -t fluid-control-fno20 2>/dev/null; do
    sleep 30
done
rg -q '^PHYSICSNEMO_SPARK_20EPOCH_ACTION_ABLATIONS_OK$' "${pipeline_log}" || {
    echo "20-epoch FNO pipeline did not complete all action ablations" >&2
    exit 1
}
python3 scripts/validate_tandem_fno_stage.py \
    --training-history "${model}/training_history.json" --min-epoch 20
for mode in observed zero sign_flip; do
    suffix="_${mode}"
    if [[ "${mode}" == observed ]]; then suffix=""; fi
    python3 scripts/validate_tandem_fno_stage.py \
        --evaluation "${model}/heldout_evaluation${suffix}.json" \
        --action-mode "${mode}"
done
python3 scripts/assess_tandem_control_readiness.py \
    "${model}/heldout_evaluation.json" \
    "${model}/heldout_evaluation_zero.json" \
    --output "${readiness}"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["status"]=="CANDIDATE_SURROGATE_SCREEN_PASS" else 1)' "${readiness}" || {
    echo "Surrogate screen failed; PPO pilot intentionally not started"
    exit 0
}
echo "Independent surrogate gate passed; starting CPU-only HydroGym PPO pilot"
bash scripts/run_tandem_hydrogym_ppo_pilot_spark.sh
