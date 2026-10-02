#!/usr/bin/env bash
# Continue the predeclared 10-step rollout replication after both one-step audits.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model="artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch"
data="data/curated/tandem_cylinders_gate_b_aug_v3"
log="artifacts/tandem_cylinders/gate_b_aug_v3_rollout_seed20261002.log"
failed="artifacts/tandem_cylinders/GATE_B_AUG_V3_ROLLOUT_SEED20261002_FAILED"
complete="artifacts/tandem_cylinders/GATE_B_AUG_V3_ROLLOUT_SEED20261002_COMPLETE"
trap 'code=$?; touch "${failed}"; echo "PRIMARY_ROLLOUT_FAILED exit=${code} $(date -Is)" >&2' ERR

while [[ ! -e artifacts/tandem_fno_gate_b_aug_v3_30epoch/GATE_B_V3_ONE_STEP_AUDIT_COMPLETE \
      || ! -e artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal/GATE_B_AUG_V3_SEED20261005_COMPLETE ]]; do
    [[ ! -e artifacts/tandem_cylinders/GATE_B_AUG_V3_EVAL_FAILED \
       && ! -e artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal/GATE_B_AUG_V3_SEED20261005_FAILED ]] || {
        echo 'A prerequisite audit failed; refusing automatic training' >&2
        exit 1
    }
    sleep 60
done

[[ ! -e "${model}" && ! -e "${complete}" ]] || {
    echo "Refusing to overwrite an existing primary rollout: ${model}" >&2
    exit 1
}
[[ -d artifacts/tandem_fno_gate_b_aug_v3_30epoch/best \
   && -s "${data}/normalization.json" ]] || exit 1

OUTPUT_DIR="${model}" DATA_ROOT="${data}" \
INITIAL_CHECKPOINT=artifacts/tandem_fno_gate_b_aug_v3_30epoch/best \
EPOCHS=10 BATCH_SIZE=4 ROLLOUT_STEPS=10 SEED=20261002 \
    bash scripts/run_tandem_fno_total_drag_rollout_spark.sh >"${log}" 2>&1
grep -q '"event": "gpu_guard_complete", "exit_code": 0' "${log}"
jq -e 'length == 10 and .[-1].epoch == 10' "${model}/training_history.json" >/dev/null

for mode in observed zero sign_flip shuffle; do
    MODEL_DIR="${model}" EVALUATION_DATA="${data}" NORMALIZATION_DATA="${data}" \
    ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh >>"${log}" 2>&1
done
MODEL_DIR="${model}" EVALUATION_DATA=data/curated/tandem_cylinders_phase_v1 \
NORMALIZATION_DATA="${data}" EVALUATION_LABEL=phase_independent \
ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh >>"${log}" 2>&1

python3 scripts/audit_tandem_gate_b.py \
    --observed "${model}/heldout_evaluation.json" \
    --zero "${model}/heldout_evaluation_zero.json" \
    --sign-flip "${model}/heldout_evaluation_sign_flip.json" \
    --shuffle "${model}/heldout_evaluation_shuffle.json" \
    --independent "${model}/heldout_evaluation_phase_independent.json" \
    --output "${model}/gate_b_audit.json" \
    --markdown "${model}/gate_b_audit.md" >>"${log}" 2>&1

MODEL_DIR="${model}" \
EVALUATION_DATA=data/curated/tandem_cylinders_expanded_independent_v2 \
NORMALIZATION_DATA="${data}" EVALUATION_LABEL=legacy_four \
ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh >>"${log}" 2>&1

python3 - "${model}" <<'PY' >>"${log}" 2>&1
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
audit = json.loads((model / 'gate_b_audit.json').read_text())
fresh = json.loads((model / 'heldout_evaluation.json').read_text())
legacy = json.loads((model / 'heldout_evaluation_legacy_four.json').read_text())
cases = {row['case']: row['horizons']['100']['total_drag_nrmse'] for row in fresh['cases']}
if len(cases) != 5 or 'expanded_test_05' not in cases:
    raise SystemExit('Fresh independent CFD case 05 is missing')
report = {
    'gate_status': audit['status'],
    'formal_five_case_100step_nrmse': fresh['summary']['100']['total_drag_nrmse'],
    'legacy_four_case_100step_nrmse': legacy['summary']['100']['total_drag_nrmse'],
    'fresh_test_05_100step_nrmse': cases['expanded_test_05'],
    'case_100step_nrmse': cases,
    'interpretation': 'surrogate accuracy only; no closed-loop CFD drag-reduction claim',
}
(model / 'gate_b_v3_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
PY
touch "${complete}"
echo "PRIMARY_ROLLOUT_COMPLETE $(date -Is)" >>"${log}"
