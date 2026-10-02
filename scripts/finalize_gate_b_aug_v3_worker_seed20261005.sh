#!/usr/bin/env bash
# Retrieve the compute-only v3 FNO seed and run the frozen test only on primary.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
worker="${WORKER_HOST:-USER@WORKER_HOST}"
remote_root="/tmp/fluid_control_gateb_20261002"
remote_pid="${REMOTE_SUPERVISOR_PID:?REMOTE_SUPERVISOR_PID is required}"
[[ "${remote_pid}" =~ ^[1-9][0-9]*$ ]] || exit 2
run="tandem_fno_gate_b_aug_v3_seed20261005_30epoch"
destination="artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal"
model="${destination}/${run}"
data="data/curated/tandem_cylinders_gate_b_aug_v3"
phase="data/curated/tandem_cylinders_phase_v1"
legacy="data/curated/tandem_cylinders_expanded_independent_v2"
mkdir -p "${destination}"
failed="${destination}/GATE_B_AUG_V3_SEED20261005_FAILED"
complete="${destination}/GATE_B_AUG_V3_SEED20261005_COMPLETE"
trap 'touch "${failed}"; echo "V3_SEED20261005_FINALIZE_FAILED $(date -Is)" >&2' ERR
fail() { echo "$*" >&2; touch "${failed}"; exit 1; }

while ! ssh -o BatchMode=yes "${worker}" \
    test -e "${remote_root}/artifacts/tandem_cylinders/GATE_B_AUG_V3_SEED20261005_COMPLETE"; do
    if ssh -o BatchMode=yes "${worker}" \
        test -e "${remote_root}/artifacts/tandem_cylinders/GATE_B_AUG_V3_SEED20261005_FAILED"; then
        fail "Worker reported training failure"
    fi
    if ! ssh -o BatchMode=yes "${worker}" kill -0 "${remote_pid}" 2>/dev/null; then
        fail "Worker supervisor stopped before the completion marker"
    fi
    sleep 60
done

[[ ! -e "${model}" ]] || fail "Refusing to overwrite worker result"
rsync -a "${worker}:${remote_root}/artifacts/${run}" "${destination}/"
rsync -a "${worker}:${remote_root}/artifacts/tandem_cylinders/gate_b_aug_v3_seed20261005_worker.log" \
    "${destination}/runner.log"
grep -q '"event": "gpu_guard_complete", "exit_code": 0' "${destination}/runner.log"
jq -e 'length == 30 and .[-1].epoch == 30' "${model}/training_history.json" >/dev/null

for file in "${model}"/best/*.mdlus "${model}/training_history.json"; do
    [[ -f "${file}" ]] || fail "Missing worker artifact ${file}"
    relative="${file#"${model}/"}"
    remote_sha="$(ssh -o BatchMode=yes "${worker}" \
        "sha256sum '${remote_root}/artifacts/${run}/${relative}'" | awk '{print $1}')"
    local_sha="$(sha256sum "${file}" | awk '{print $1}')"
    [[ -n "${remote_sha}" && "${remote_sha}" == "${local_sha}" ]] ||
        fail "Worker transfer checksum mismatch: ${relative}"
    echo "TRANSFER_SHA256_OK ${relative} ${local_sha}"
done

for mode in observed zero sign_flip shuffle; do
    MODEL_DIR="${model}" EVALUATION_DATA="${data}" NORMALIZATION_DATA="${data}" \
    ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh
done
MODEL_DIR="${model}" EVALUATION_DATA="${phase}" NORMALIZATION_DATA="${data}" \
EVALUATION_LABEL=phase_independent ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh

python3 scripts/audit_tandem_gate_b.py \
    --observed "${model}/heldout_evaluation.json" \
    --zero "${model}/heldout_evaluation_zero.json" \
    --sign-flip "${model}/heldout_evaluation_sign_flip.json" \
    --shuffle "${model}/heldout_evaluation_shuffle.json" \
    --independent "${model}/heldout_evaluation_phase_independent.json" \
    --output "${model}/gate_b_audit.json" \
    --markdown "${model}/gate_b_audit.md"

MODEL_DIR="${model}" EVALUATION_DATA="${legacy}" NORMALIZATION_DATA="${data}" \
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
touch "${complete}"
echo "V3_SEED20261005_FINALIZE_OK $(date -Is)"
