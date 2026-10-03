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
remote_log="${REMOTE_LOG:-/tmp/fluid_control_multistep_formal.log}"
expected_epochs="${EXPECTED_EPOCHS:-10}"
evaluation_data="${EVALUATION_DATA:-data/curated/tandem_cylinders_expanded_independent_v2}"
normalization_data="${NORMALIZATION_DATA:-data/curated/tandem_cylinders_expanded_independent_v2}"
parent_validation_json="${PARENT_VALIDATION_JSON:-}"

[[ "${destination}" != /* && "${run_name}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
    echo "Destination must be project-relative and run name must be safe" >&2
    exit 2
}
[[ "${remote_pid}" =~ ^[1-9][0-9]*$ && "${poll_seconds}" =~ ^[1-9][0-9]*$ \
    && "${expected_epochs}" =~ ^[1-9][0-9]*$ \
    && "${remote_log}" == /tmp/fluid_control_*.log ]] || {
    echo "Remote PID, poll interval, expected epochs, or runner log is invalid" >&2
    exit 2
}
[[ "${evaluation_data}" == data/curated/* && "${normalization_data}" == data/curated/* \
    && -d "${evaluation_data}/test" && -s "${normalization_data}/normalization.json" ]] || {
    echo "Evaluation data or train-only normalization is missing" >&2; exit 2;
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
    "${worker}:${remote_log}" \
    "${destination}/runner.log"

grep -q '"event": "gpu_guard_complete", "exit_code": 0' \
    "${destination}/runner.log"
jq -e --argjson epochs "${expected_epochs}" \
    'length == $epochs and .[-1].epoch == $epochs' \
    "${destination}/${run_name}/training_history.json" >/dev/null

model="${destination}/${run_name}"
for file in "${model}"/best/*.mdlus "${model}/training_history.json"; do
    [[ -f "${file}" ]] || { echo "Missing transferred model or history: ${file}" >&2; exit 1; }
    relative="${file#"${model}/"}"
    remote_sha="$(ssh -o BatchMode=yes "${worker}" \
        "sha256sum '${remote_root}/artifacts/${run_name}/${relative}'" | awk '{print $1}')"
    local_sha="$(sha256sum "${file}" | awk '{print $1}')"
    [[ -n "${remote_sha}" && "${remote_sha}" == "${local_sha}" ]] || {
        echo "Worker-to-primary checksum mismatch: ${relative}" >&2; exit 1;
    }
    echo "TRANSFER_SHA256_OK ${relative} ${local_sha}"
done
if [[ -n "${parent_validation_json}" ]]; then
    [[ "${parent_validation_json}" == artifacts/* && -s "${parent_validation_json}" ]] || {
        echo "Parent validation JSON must be an existing project artifact" >&2; exit 2;
    }
    docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
        --shm-size 2g --user "$(id -u):$(id -g)" \
        --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
        --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
        --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
        fluid-control-physicsnemo:2.2.2 \
        python -u scripts/spark_gpu_guard.py \
          --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
          python -u scripts/evaluate_tandem_fno.py \
            --data "${evaluation_data}" --normalization-data "${normalization_data}" \
            --config conf/tandem_fno_total_drag.yaml \
            --checkpoint-dir "/workspace/${model}/best" \
            --output "/workspace/${model}/validation_long_horizon.json" \
            --split validation --horizons 1 10 50 100 --segment-stride 25 \
            --action-mode observed --evaluation-batch-size 4 \
            --visualizations-per-horizon 0
    python3 - "${parent_validation_json}" "${model}/validation_long_horizon.json" \
        "${model}/validation_decision.json" <<'PY'
import json
import math
import sys
from pathlib import Path

parent_path, candidate_path, decision_path = map(Path, sys.argv[1:])
parent = json.loads(parent_path.read_text())
candidate = json.loads(candidate_path.read_text())
if parent.get("split") != "validation" or candidate.get("split") != "validation":
    raise SystemExit("validation-first gate received a non-validation report")
for key in ("evaluation_data", "normalization_data", "segment_stride", "action_mode"):
    if parent.get(key) != candidate.get(key):
        raise SystemExit(f"validation comparison mismatch: {key}")
parent_error = float(parent["summary"]["100"]["total_drag_nrmse"])
candidate_error = float(candidate["summary"]["100"]["total_drag_nrmse"])
if not all(math.isfinite(value) and value >= 0 for value in (parent_error, candidate_error)):
    raise SystemExit("non-finite validation error")
proceed = candidate_error < parent_error
report = {
    "status": "VALIDATION_IMPROVED_PROCEED_TO_FROZEN_GATE_B" if proceed else "VALIDATION_NO_GAIN_SKIP_FROZEN_TEST",
    "parent_validation_100step_nrmse": parent_error,
    "candidate_validation_100step_nrmse": candidate_error,
    "selection_rule": "strictly lower 100-step NRMSE on identical validation split",
    "scientific_scope": "surrogate accuracy only, no CFD control-benefit claim",
}
decision_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
PY
    if [[ "$(jq -r '.status' "${model}/validation_decision.json")" == VALIDATION_NO_GAIN_SKIP_FROZEN_TEST ]]; then
        touch "${destination}/MULTISTEP_VALIDATION_COMPLETE_NO_GAIN"
        echo "VALIDATION_NO_GAIN_SKIP_FROZEN_TEST $(date -Is)"
        exit 0
    fi
fi
for mode in observed zero sign_flip shuffle; do
    MODEL_DIR="${model}" EVALUATION_DATA="${evaluation_data}" \
    NORMALIZATION_DATA="${normalization_data}" \
    ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh
done
MODEL_DIR="${model}" \
EVALUATION_DATA=data/curated/tandem_cylinders_phase_v1 \
NORMALIZATION_DATA="${normalization_data}" \
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

if [[ "${evaluation_data}" == data/curated/tandem_cylinders_gate_b_aug_v3 ]]; then
    MODEL_DIR="${model}" \
    EVALUATION_DATA=data/curated/tandem_cylinders_expanded_independent_v2 \
    NORMALIZATION_DATA="${normalization_data}" \
    EVALUATION_LABEL=legacy_four ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh
    python3 - "${model}" <<'PY'
import json,sys
from pathlib import Path
model=Path(sys.argv[1])
audit=json.loads((model/'gate_b_audit.json').read_text())
fresh=json.loads((model/'heldout_evaluation.json').read_text())
legacy=json.loads((model/'heldout_evaluation_legacy_four.json').read_text())
cases={row['case']:row['horizons']['100']['total_drag_nrmse'] for row in fresh['cases']}
if len(cases)!=5 or 'expanded_test_05' not in cases:
    raise SystemExit('Fresh independent test case 05 is missing')
report={'gate_status':audit['status'],
        'formal_five_case_100step_nrmse':fresh['summary']['100']['total_drag_nrmse'],
        'legacy_four_case_100step_nrmse':legacy['summary']['100']['total_drag_nrmse'],
        'fresh_test_05_100step_nrmse':cases['expanded_test_05'],
        'case_100step_nrmse':cases,
        'interpretation':'surrogate accuracy only; no CFD control-benefit claim'}
(model/'gate_b_v3_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
fi

touch "${complete_marker}"
echo "FINALIZE_OK $(date -Is)"
