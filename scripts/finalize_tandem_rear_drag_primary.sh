#!/usr/bin/env bash
# Validation-first finalization of the predeclared rear-drag loss ablation.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model="artifacts/tandem_fno_gate_b_aug_v3_rear_drag_seed20261007_10epoch"
parent="artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch"
data="data/curated/tandem_cylinders_gate_b_aug_v3"
train_log="artifacts/tandem_cylinders/gate_b_aug_v3_rear_drag_seed20261007.log"
decision="${model}/validation_decision.json"
failed="artifacts/tandem_cylinders/GATE_B_AUG_V3_REAR_DRAG_FINALIZE_FAILED"
complete="artifacts/tandem_cylinders/GATE_B_AUG_V3_REAR_DRAG_FINALIZE_COMPLETE"
deadline_epoch="$(date -u -d '2026-10-02 21:45:00 UTC' +%s)"

on_exit() {
    local code=$?
    if (( code != 0 )); then
        touch "${failed}"
        echo "REAR_DRAG_FINALIZE_FAILED exit=${code} $(date -u -Is)" >&2
    fi
}
trap on_exit EXIT

[[ ! -e "${decision}" && ! -e "${complete}" ]] || {
    echo "Refusing to overwrite a completed decision" >&2; exit 2;
}
[[ -s "${parent}/validation_long_horizon.json" && -d "${model}" ]] || {
    echo "Parent validation or new training directory missing" >&2; exit 2;
}

while ! grep -q '"event": "gpu_guard_complete", "exit_code": 0' "${train_log}"; do
    if grep -Eq '"event": "gpu_guard_complete", "exit_code": [1-9]' "${train_log}"; then
        echo "Training failed GPU guard" >&2; exit 1;
    fi
    if (( $(date -u +%s) >= deadline_epoch )); then
        echo "Training did not finish by research cutoff" >&2; exit 1;
    fi
    sleep 60
done
jq -e 'length == 10 and .[-1].epoch == 10' "${model}/training_history.json" >/dev/null
[[ -d "${model}/best" ]] || { echo "Best checkpoint missing" >&2; exit 1; }

docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
    fluid-control-physicsnemo:2.2.2 \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/evaluate_tandem_fno.py \
        --data "${data}" --normalization-data "${data}" \
        --config conf/tandem_fno_total_drag.yaml \
        --checkpoint-dir "/workspace/${model}/best" \
        --output "/workspace/${model}/validation_long_horizon.json" \
        --split validation --horizons 1 10 50 100 --segment-stride 25 \
        --action-mode observed --evaluation-batch-size 4 \
        --visualizations-per-horizon 0

python3 - "${parent}" "${model}" "${decision}" <<'PY'
import json
import sys
from pathlib import Path

parent, model, path = map(Path, sys.argv[1:])
parent_nrmse = json.loads((parent / "validation_long_horizon.json").read_text())["summary"]["100"]["total_drag_nrmse"]
new_nrmse = json.loads((model / "validation_long_horizon.json").read_text())["summary"]["100"]["total_drag_nrmse"]
if not (0 <= parent_nrmse < float("inf") and 0 <= new_nrmse < float("inf")):
    raise SystemExit("non-finite validation NRMSE")
report = {
    "status": "VALIDATION_IMPROVED_PROCEED_TO_FROZEN_GATE_B" if new_nrmse < parent_nrmse else "VALIDATION_NO_GAIN_SKIP_FROZEN_TEST",
    "parent_validation_100step_nrmse": parent_nrmse,
    "rear_weighted_validation_100step_nrmse": new_nrmse,
    "selection_rule": "new validation 100-step NRMSE must be strictly below same-seed parent",
    "scientific_status": "surrogate accuracy only; no physical control-benefit claim",
}
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
PY

if [[ "$(jq -r '.status' "${decision}")" == VALIDATION_NO_GAIN_SKIP_FROZEN_TEST ]]; then
    touch "${complete}"
    echo "REAR_DRAG_VALIDATION_NO_GAIN $(date -u -Is)"
    exit 0
fi

for mode in observed zero sign_flip shuffle; do
    MODEL_DIR="${model}" EVALUATION_DATA="${data}" NORMALIZATION_DATA="${data}" \
    ACTION_MODE="${mode}" VISUALIZATIONS_PER_HORIZON=0 \
        bash scripts/run_tandem_fno_total_drag_eval_spark.sh
done
MODEL_DIR="${model}" EVALUATION_DATA=data/curated/tandem_cylinders_phase_v1 \
NORMALIZATION_DATA="${data}" EVALUATION_LABEL=phase_independent \
ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh

python3 scripts/audit_tandem_gate_b.py \
    --observed "${model}/heldout_evaluation.json" \
    --zero "${model}/heldout_evaluation_zero.json" \
    --sign-flip "${model}/heldout_evaluation_sign_flip.json" \
    --shuffle "${model}/heldout_evaluation_shuffle.json" \
    --independent "${model}/heldout_evaluation_phase_independent.json" \
    --output "${model}/gate_b_audit.json" \
    --markdown "${model}/gate_b_audit.md"

MODEL_DIR="${model}" EVALUATION_DATA=data/curated/tandem_cylinders_expanded_independent_v2 \
NORMALIZATION_DATA="${data}" EVALUATION_LABEL=legacy_four \
ACTION_MODE=observed VISUALIZATIONS_PER_HORIZON=0 \
    bash scripts/run_tandem_fno_total_drag_eval_spark.sh

python3 - "${model}" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
audit = json.loads((model / "gate_b_audit.json").read_text())
fresh = json.loads((model / "heldout_evaluation.json").read_text())
legacy = json.loads((model / "heldout_evaluation_legacy_four.json").read_text())
cases = {row["case"]: row["horizons"]["100"]["total_drag_nrmse"] for row in fresh["cases"]}
if len(cases) != 5 or "expanded_test_05" not in cases:
    raise SystemExit("fresh independent case 05 missing")
report = {
    "gate_status": audit["status"],
    "formal_five_case_100step_nrmse": fresh["summary"]["100"]["total_drag_nrmse"],
    "legacy_four_case_100step_nrmse": legacy["summary"]["100"]["total_drag_nrmse"],
    "fresh_test_05_100step_nrmse": cases["expanded_test_05"],
    "case_100step_nrmse": cases,
    "interpretation": "surrogate accuracy only; no CFD control-benefit claim",
}
(model / "gate_b_v3_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
PY

touch "${complete}"
echo "REAR_DRAG_FINALIZE_OK $(date -u -Is)"
