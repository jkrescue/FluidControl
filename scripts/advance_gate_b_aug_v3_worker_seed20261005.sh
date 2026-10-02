#!/usr/bin/env bash
# Compute-only independent v3 FNO seed; never copies the held-out test split.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
smoke_pid="${SMOKE_PID:?SMOKE_PID is required}"
[[ "${smoke_pid}" =~ ^[1-9][0-9]*$ ]] || exit 2
smoke="artifacts/tandem_fno_gate_b_aug_v3_seed20261005_smoke"
formal="artifacts/tandem_fno_gate_b_aug_v3_seed20261005_30epoch"
log="artifacts/tandem_cylinders/gate_b_aug_v3_seed20261005_worker.log"
failed="artifacts/tandem_cylinders/GATE_B_AUG_V3_SEED20261005_FAILED"
complete="artifacts/tandem_cylinders/GATE_B_AUG_V3_SEED20261005_COMPLETE"
trap 'touch "${failed}"; echo "V3_SEED20261005_FAILED $(date -Is)" >&2' ERR

while [[ ! -s "${smoke}/training_history.json" ]]; do
    kill -0 "${smoke_pid}" 2>/dev/null || {
        echo "One-epoch seed smoke ended without history" >&2; exit 1;
    }
    sleep 30
done
python3 - "${smoke}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
history=json.loads((root/'training_history.json').read_text())
if len(history)!=1 or history[0]['epoch']!=1 or not list((root/'best').glob('FNO.*.mdlus')):
    raise SystemExit('Incomplete one-epoch seed smoke')
PY
echo "V3_SEED20261005_SMOKE_OK $(date -Is)" >> "${log}"

SEED=20261005 EPOCHS=30 OUTPUT_DIR="${formal}" \
    bash scripts/run_gate_b_aug_v3_onestep_spark.sh >> "${log}" 2>&1
python3 - "${formal}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
history=json.loads((root/'training_history.json').read_text())
if len(history)!=30 or history[-1]['epoch']!=30 or not list((root/'best').glob('FNO.*.mdlus')):
    raise SystemExit('Incomplete formal seed training')
PY
touch "${complete}"
echo "V3_SEED20261005_FORMAL_OK $(date -Is)" >> "${log}"
