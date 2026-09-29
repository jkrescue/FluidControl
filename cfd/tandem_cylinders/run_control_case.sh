#!/usr/bin/env bash
set -euo pipefail

# Run one generated control case inside the pinned, project-bound OpenFOAM image.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
label="${1:-}"
case "$label" in
    m100|m050|z000|p050|p100) ;;
    *) echo 'Usage: bash run_control_case.sh m100|m050|z000|p050|p100' >&2; exit 2 ;;
esac
name="control_small_${label}"
target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
if [[ ! -f "${target}/system/controlDict" || ! -f "${target}/constant/polyMesh/boundary" ]]; then
    echo "Missing configured case or mesh: ${target}" >&2
    exit 1
fi
if ! grep -q 'internalField.*nonuniform' "${target}/0/U"; then
    echo "Seeded initial field missing: ${target}/0/U" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite solver log: $log" >&2
    exit 1
fi
if find "$target" -maxdepth 1 -type d -name '[1-9]*' -print -quit | grep -q .; then
    echo "Refusing to overwrite existing time directories: $target" >&2
    exit 1
fi

echo "Starting ${name}; solver log: ${log}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" > "$log" 2>&1; then
    echo "Completed ${name}"
    tail -n 4 "$log"
else
    result=$?
    echo "Failed ${name} (exit ${result}); final log lines:" >&2
    tail -n 35 "$log" >&2
    exit "$result"
fi
