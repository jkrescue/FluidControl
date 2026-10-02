#!/usr/bin/env bash
# Preserve the unchanged Gate-B limit and test the new independent CFD case.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model="artifacts/tandem_fno_gate_b_aug_v3_30epoch"
data="data/curated/tandem_cylinders_gate_b_aug_v3"
legacy="data/curated/tandem_cylinders_expanded_independent_v2"
phase="data/curated/tandem_cylinders_phase_v1"
output_marker="${model}/GATE_B_V3_ONE_STEP_AUDIT_COMPLETE"

[[ -e artifacts/tandem_cylinders/GATE_B_AUG_V3_ONE_STEP_COMPLETE \
    && -d "${model}/best" && -s "${data}/manifest.json" ]] || {
    echo "Formal augmented one-step model or data is incomplete" >&2; exit 1;
}
[[ ! -e "${output_marker}" && ! -e "${model}/gate_b_audit.json" ]] || {
    echo "Refusing to overwrite completed model audit" >&2; exit 1;
}

for mode in observed zero sign_flip shuffle; do
    figures=0
    [[ "${mode}" == observed ]] && figures=1
    MODEL_DIR="${model}" EVALUATION_DATA="${data}" NORMALIZATION_DATA="${data}" \
    ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON="${figures}" \
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

# Preserve the original four-case cohort for a like-for-like comparison. The
# formal v3 gate above additionally includes fresh, untouched test_05.
MODEL_DIR="${model}" EVALUATION_DATA="${legacy}" NORMALIZATION_DATA="${data}" \
EVALUATION_LABEL=legacy_four ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh

python3 - "${model}" <<'PY'
import json,sys
from pathlib import Path
model=Path(sys.argv[1])
audit=json.loads((model/"gate_b_audit.json").read_text())
v3=json.loads((model/"heldout_evaluation.json").read_text())
legacy=json.loads((model/"heldout_evaluation_legacy_four.json").read_text())
cases={row["case"]:row["horizons"]["100"]["total_drag_nrmse"] for row in v3["cases"]}
if "expanded_test_05" not in cases or len(cases)!=5:
    raise SystemExit("fresh test 05 is missing from formal v3 evaluation")
report={
    "gate_status":audit["status"],
    "checkpoint_epoch":audit["checkpoint_epoch"],
    "formal_five_case_100step_nrmse":v3["summary"]["100"]["total_drag_nrmse"],
    "legacy_four_case_100step_nrmse":legacy["summary"]["100"]["total_drag_nrmse"],
    "fresh_test_05_100step_nrmse":cases["expanded_test_05"],
    "case_100step_nrmse":cases,
    "interpretation":"surrogate accuracy only; no closed-loop CFD drag-reduction claim",
}
(model/"gate_b_v3_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
PY
touch "${output_marker}"
echo GATE_B_V3_ONE_STEP_AUDIT_COMPLETE
