#!/usr/bin/env bash
# Retrieve a completed stateless-worker run and execute the canonical Gate-B suite.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
worker="${WORKER_HOST:-USER@WORKER_HOST}"
remote_pid="${REMOTE_PID:?REMOTE_PID is required}"
remote_root="${REMOTE_ROOT:-/tmp/fluid_control_gateb_20261002}"
run_name="${RUN_NAME:-tandem_fno_total_drag_rollout_seed20261003}"
destination="${DESTINATION:-artifacts/distributed_runs/gateb_multistep_20261002/formal}"
poll_seconds="${POLL_SECONDS:-60}"

[[ "${destination}" != /* && "${run_name}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
    echo "Destination must be project-relative and run name must be safe" >&2
    exit 2
}
[[ "${remote_pid}" =~ ^[1-9][0-9]*$ && "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Remote PID and poll interval must be positive integers" >&2
    exit 2
}

mkdir -p "${destination}"
failure_marker="${destination}/MULTISTEP_GATE_B_FAILED"
complete_marker="${destination}/MULTISTEP_GATE_B_COMPLETE"
on_error() {
    local code=$?
    trap - ERR
    echo "FINALIZE_FAILED exit=${code} at $(date -Is)"
    touch "${failure_marker}"
    exit "${code}"
}
trap on_error ERR

while true; do
    if ssh -o BatchMode=yes "${worker}" kill -0 "${remote_pid}" 2>/dev/null; then
        sleep "${poll_seconds}"
        continue
    fi
    if ssh -o BatchMode=yes "${worker}" true 2>/dev/null; then
        break
    fi
    echo "Worker temporarily unreachable at $(date -Is); retrying" >&2
    sleep "${poll_seconds}"
done

rsync -a \
    "${worker}:${remote_root}/artifacts/${run_name}" \
    "${destination}/"
rsync -a \
    "${worker}:/tmp/fluid_control_multistep_formal.log" \
    "${destination}/runner.log"

grep -q '"event": "gpu_guard_complete", "exit_code": 0' \
    "${destination}/runner.log"
jq -e 'length == 10 and .[-1].epoch == 10' \
    "${destination}/${run_name}/training_history.json" >/dev/null

model="${destination}/${run_name}"
for mode in observed zero sign_flip shuffle; do
    MODEL_DIR="${model}" ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh
done
MODEL_DIR="${model}" \
EVALUATION_DATA=data/curated/tandem_cylinders_phase_v1 \
NORMALIZATION_DATA=data/curated/tandem_cylinders_expanded_independent_v2 \
EVALUATION_LABEL=phase_independent ACTION_MODE=observed \
VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh

python3 scripts/audit_tandem_gate_b.py \
    --observed "${model}/heldout_evaluation.json" \
    --zero "${model}/heldout_evaluation_zero.json" \
    --sign-flip "${model}/heldout_evaluation_sign_flip.json" \
    --shuffle "${model}/heldout_evaluation_shuffle.json" \
    --independent "${model}/heldout_evaluation_phase_independent.json" \
    --output "${model}/gate_b_audit.json" \
    --markdown "${model}/gate_b_audit.md"

touch "${complete_marker}"
echo "FINALIZE_OK $(date -Is)"
