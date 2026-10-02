#!/usr/bin/env bash
# Wait for the formal v3 one-step checkpoint and run all frozen audit panels.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
training_pid="${TRAINING_SUPERVISOR_PID:?TRAINING_SUPERVISOR_PID is required}"
[[ "${training_pid}" =~ ^[1-9][0-9]*$ ]] || exit 2
failed="artifacts/tandem_cylinders/GATE_B_AUG_V3_EVAL_FAILED"
on_error() {
    local code=$?
    trap - ERR
    touch "${failed}"
    echo "V3_EVAL_FAILED exit=${code} at $(date -Is)" >&2
    exit "${code}"
}
trap on_error ERR

while [[ ! -e artifacts/tandem_cylinders/GATE_B_AUG_V3_ONE_STEP_COMPLETE ]]; do
    if ! kill -0 "${training_pid}" 2>/dev/null; then
        echo "Training supervisor stopped without a complete marker" >&2
        exit 1
    fi
    sleep 30
done
bash scripts/evaluate_gate_b_aug_v3_onestep_spark.sh
