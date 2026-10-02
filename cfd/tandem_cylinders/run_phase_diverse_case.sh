#!/usr/bin/env bash
# Run one pre-generated independent-phase OpenFOAM trajectory without overwriting it.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "${name}" in
    phase_train_82_multisine|phase_train_84_ramp|phase_validation_86_multisine|phase_test_88_ramp) ;;
    *) echo "Unknown phase-diverse case: ${name}" >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
[[ -f "${target}/case_config.json" ]] || { echo "Missing generated case" >&2; exit 1; }
[[ ! -e "${log}" ]] || { echo "Refusing to overwrite solver log: ${log}" >&2; exit 1; }

readarray -t bounds < <(python3 - "${target}/case_config.json" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{cfg['start_time']:g}")
print(f"{cfg['end_time']:g}")
PY
)
start_time="${bounds[0]}"
end_time="${bounds[1]}"
[[ -f "${target}/${start_time}/U" ]] || { echo "Missing restart U" >&2; exit 1; }
if find "${target}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk -v start="${start_time}" '$1 ~ /^[0-9.]+$/ && $1 > start {found=1} END {exit !found}'; then
    echo "Refusing existing output time newer than ${start_time}" >&2
    exit 1
fi

echo "Starting ${name}: ${start_time} -> ${end_time}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1; then
    grep -qx End "${log}" || { echo "Solver log has no End marker" >&2; exit 1; }
    echo "Completed ${name}"
    tail -n 4 "${log}"
else
    result=$?
    echo "Failed ${name} (exit ${result})" >&2
    tail -n 50 "${log}" >&2
    exit "${result}"
fi
