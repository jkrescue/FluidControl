#!/usr/bin/env bash
# After independent Curator QC, run a one-epoch smoke then formal FNO training.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
curator_pid="${CURATOR_PID:?CURATOR_PID is required}"
poll_seconds="${POLL_SECONDS:-30}"
[[ "${curator_pid}" =~ ^[1-9][0-9]*$ && "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CURATOR_PID and POLL_SECONDS must be positive integers" >&2; exit 2;
}
log_dir="artifacts/tandem_cylinders"
curator_log="${log_dir}/gate_b_aug_v3_curator.log"
failed_marker="${log_dir}/GATE_B_AUG_V3_ONE_STEP_FAILED"
complete_marker="${log_dir}/GATE_B_AUG_V3_ONE_STEP_COMPLETE"
on_error() {
    local code=$?
    trap - ERR
    echo "V3_ONE_STEP_FAILED exit=${code} at $(date -Is)" >&2
    touch "${failed_marker}"
    exit "${code}"
}
trap on_error ERR

while ! rg -q '^GATE_B_AUG_V3_CURATED_OK$' "${curator_log}" 2>/dev/null; do
    if ! kill -0 "${curator_pid}" 2>/dev/null; then
        echo "Curator process ended without the required data marker" >&2
        exit 1
    fi
    sleep "${poll_seconds}"
done
echo "Validated v3 Curator dataset is ready at $(date -Is)"

EPOCHS=1 BATCH_SIZE=64 \
OUTPUT_DIR=artifacts/tandem_fno_gate_b_aug_v3_smoke \
    bash scripts/run_gate_b_aug_v3_onestep_spark.sh \
    > "${log_dir}/gate_b_aug_v3_onestep_smoke.log" 2>&1
grep -q '"event": "gpu_guard_complete", "exit_code": 0' \
    "${log_dir}/gate_b_aug_v3_onestep_smoke.log"
jq -e 'length == 1 and .[-1].epoch == 1' \
    artifacts/tandem_fno_gate_b_aug_v3_smoke/training_history.json >/dev/null
echo "V3_ONE_STEP_SMOKE_OK $(date -Is)"

EPOCHS=30 BATCH_SIZE=64 \
OUTPUT_DIR=artifacts/tandem_fno_gate_b_aug_v3_30epoch \
    bash scripts/run_gate_b_aug_v3_onestep_spark.sh \
    > "${log_dir}/gate_b_aug_v3_onestep_formal.log" 2>&1
grep -q '"event": "gpu_guard_complete", "exit_code": 0' \
    "${log_dir}/gate_b_aug_v3_onestep_formal.log"
jq -e 'length == 30 and .[-1].epoch == 30' \
    artifacts/tandem_fno_gate_b_aug_v3_30epoch/training_history.json >/dev/null
touch "${complete_marker}"
echo "V3_ONE_STEP_FORMAL_OK $(date -Is)"
