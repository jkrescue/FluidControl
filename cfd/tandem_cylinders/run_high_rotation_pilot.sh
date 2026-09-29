#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "$name" in
    high_rotation_qm200_dt005|high_rotation_qp200_dt005|high_rotation_qm250_dt005|high_rotation_qp250_dt005|high_rotation_qm250_dt0025|high_rotation_qp250_dt0025|high_rotation_qm250_medium_dt0025|high_rotation_qp250_medium_dt0025) ;;
    *) echo 'Usage: run_high_rotation_pilot.sh <high_rotation_q[m|p]..._dt...>' >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
if [[ ! -f "${target}/case_config.json" || ! -f "${target}/80/U" ]]; then
    echo "Missing configured pilot: ${target}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite solver log: ${log}" >&2
    exit 1
fi
if find "$target" -maxdepth 1 -type d \( -name '8[1-9]*' -o -name '9[0-9]*' -o -name '100' \) -print -quit | grep -q .; then
    echo "Refusing to overwrite existing output times: ${target}" >&2
    exit 1
fi

echo "Starting ${name}; solver log: ${log}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"$log" 2>&1; then
    echo "Completed ${name}"
    tail -n 4 "$log"
else
    result=$?
    echo "Failed ${name} (exit ${result}); final log lines:" >&2
    tail -n 50 "$log" >&2
    exit "$result"
fi
