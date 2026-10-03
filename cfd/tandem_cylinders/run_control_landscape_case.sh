#!/usr/bin/env bash
# Run one frozen matched-start control-landscape case in the pinned solver.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "${name}" in
    landscape_val_zero_20261003|landscape_val_p100_20261003|landscape_val_m100_20261003|landscape_val_sine_20261003|landscape_long_val_zero_20261003|landscape_long_val_p100_20261003|landscape_long_val_m100_20261003) ;;
    *) echo "Unknown control-landscape case: ${name}" >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
[[ -f "${target}/case_config.json" && -f "${target}/80/U" ]] || {
    echo "Missing generated case or restart field" >&2; exit 1;
}
[[ ! -e "${log}" ]] || { echo "Refusing to overwrite solver log: ${log}" >&2; exit 1; }
if find "${target}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '$1 ~ /^[0-9.]+$/ && $1 > 80 {found=1} END {exit !found}'; then
    echo "Refusing existing output newer than restart t=80" >&2
    exit 1
fi

end_time="$(jq -er '.end_time' "${target}/case_config.json")"
[[ "${end_time}" == 116.0 || "${end_time}" == 160.0 ]] || {
    echo "Unexpected frozen end time: ${end_time}" >&2; exit 2;
}
echo "Starting ${name}: t=80..${end_time}, OpenFOAM v2512"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1; then
    grep -qx End "${log}" || { echo "Solver log has no End marker" >&2; exit 1; }
    echo "Completed ${name}"
else
    code=$?
    echo "Failed ${name} (exit ${code})" >&2
    tail -n 40 "${log}" >&2
    exit "${code}"
fi
