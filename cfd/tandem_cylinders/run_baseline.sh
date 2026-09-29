#!/usr/bin/env bash
set -euo pipefail

# One foreground, restart-safe baseline run. All outputs stay inside this case.
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
case "${1:-}" in
    single) name=single_baseline ;;
    tandem) name=tandem_baseline ;;
    temporal) name=tandem_dt005 ;;
    medium) name=tandem_medium_dt005 ;;
    backward) name=tandem_backward_dt005 ;;
    medium_backward) name=tandem_medium_backward_dt005 ;;
    *) echo 'Usage: bash run_baseline.sh single|tandem|temporal|medium|backward|medium_backward' >&2; exit 2 ;;
esac
target="${case_dir}/cases/${name}"
log="${target}/log.pimpleFoam"
if [[ ! -f "${target}/system/controlDict" || ! -f "${target}/constant/polyMesh/boundary" ]]; then
    echo "Case or mesh missing: ${target}" >&2
    exit 1
fi
if ! grep -q 'internalField.*nonuniform' "${target}/0/U"; then
    echo "Seeded 0/U is missing: ${target}/0/U" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite existing solver log: $log" >&2
    exit 1
fi
if find "$target" -maxdepth 1 -type d -name '[1-9]*' -print -quit | grep -q .; then
    echo "Refusing to run over existing nonzero time directories: $target" >&2
    exit 1
fi

echo "Starting ${name}; solver log: ${log}"
if bash "${case_dir}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" > "$log" 2>&1; then
    echo "Completed ${name}"
    tail -n 4 "$log"
else
    result=$?
    echo "Failed ${name} (exit ${result}); final log lines:" >&2
    tail -n 35 "$log" >&2
    exit "$result"
fi
