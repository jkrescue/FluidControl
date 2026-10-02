#!/usr/bin/env bash
# Advance to CEM only after the canonical multistep finalizer produces a passing gate.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
formal="${FORMAL_ROOT:-artifacts/distributed_runs/gateb_multistep_20261002/formal}"
run_name="${RUN_NAME:-tandem_fno_total_drag_rollout_seed20261003}"
poll_seconds="${POLL_SECONDS:-60}"
complete="${formal}/MULTISTEP_GATE_B_COMPLETE"
failed="${formal}/MULTISTEP_GATE_B_FAILED"
audit="${formal}/${run_name}/gate_b_audit.json"

[[ "${formal}" != /* && "${run_name}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
    echo "Formal root must be project-relative and run name must be safe" >&2
    exit 2
}
while [[ ! -f "${complete}" && ! -f "${failed}" ]]; do
    sleep "${poll_seconds}"
done
if [[ -f "${failed}" ]]; then
    echo "Multistep finalization failed; CEM was not started" >&2
    touch "${formal}/CEM_SKIPPED_FINALIZER_FAILED"
    exit 1
fi
status="$(jq -er '.status' "${audit}")"
if [[ "${status}" != "GATE_B_PASS" ]]; then
    echo "Gate-B status is ${status}; CEM was not started"
    touch "${formal}/CEM_SKIPPED_GATE_B"
    exit 0
fi

bash scripts/run_tandem_cem_stage_c_spark.sh
touch "${formal}/CEM_STAGE_C_COMPLETE"
