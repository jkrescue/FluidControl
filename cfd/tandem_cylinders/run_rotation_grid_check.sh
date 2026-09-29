#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="control_grid_p100_medium"
target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"

if [[ ! -f "${target}/case_config.json" ]]; then
    echo "Missing configured case: ${target}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite solver log: ${log}" >&2
    exit 1
fi

bash "${root}/run_openfoam.sh" blockMesh -case "/case/cases/${name}" >"${target}/log.blockMesh" 2>&1
bash "${root}/run_openfoam.sh" checkMesh -case "/case/cases/${name}" >"${target}/log.checkMesh" 2>&1
grep -q 'Mesh OK' "${target}/log.checkMesh"
bash "${root}/run_openfoam.sh" setFields -case "/case/cases/${name}" >"${target}/log.setFields" 2>&1

echo "Starting ${name}; solver log: ${log}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"$log" 2>&1; then
    echo "Completed ${name}"
    tail -n 4 "$log"
else
    result=$?
    echo "Failed ${name} (exit ${result}); final log lines:" >&2
    tail -n 40 "$log" >&2
    exit "$result"
fi
