#!/usr/bin/env bash
# Run one frozen, train-only signed-pulse CFD case with the pinned solver.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "${name}" in
    train_signed_pulse_p_v4_20261003|train_signed_pulse_m_v4_20261003) ;;
    *) echo "Unknown signed-pulse training case: ${name}" >&2; exit 2 ;;
esac
target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
[[ -f "${target}/case_config.json" && -f "${target}/82/U" ]] || {
    echo "Missing generated case or restart field" >&2; exit 1;
}
[[ ! -e "${log}" ]] || { echo "Refusing to overwrite solver log: ${log}" >&2; exit 1; }
if find "${target}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '$1 ~ /^[0-9.]+$/ && $1 > 82 {found=1} END {exit !found}'; then
    echo "Refusing existing output newer than restart t=82" >&2
    exit 1
fi
[[ "$(jq -er '.start_time' "${target}/case_config.json")" == 82.0 ]] || exit 2
[[ "$(jq -er '.end_time' "${target}/case_config.json")" == 162.0 ]] || exit 2
echo "Starting ${name}: t=82..162, OpenFOAM v2512"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1; then
    grep -qx End "${log}" || { echo "Solver log has no End marker" >&2; exit 1; }
    echo "Completed ${name}"
else
    code=$?
    echo "Failed ${name} (exit ${code})" >&2
    tail -n 40 "${log}" >&2
    exit "${code}"
fi
