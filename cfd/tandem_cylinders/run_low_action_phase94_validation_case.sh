#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:?case name required}"
case "${name}" in
  validation_signed_low_pulse_p_phase94_v1_20261003|validation_signed_low_pulse_m_phase94_v1_20261003) ;;
  *) echo "unexpected case: ${name}" >&2; exit 2 ;;
esac
target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
[[ -s "${target}/case_config.json" && -s "${target}/94/U" && ! -e "${log}" ]]

set +e
bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1
code=$?
set -e
printf '%s\n' "${code}" > "${target}/solver_exit_code"
if (( code != 0 )); then
  tail -n 60 "${log}" >&2
  exit "${code}"
fi
grep -qx End "${log}"
printf 'CFD_COMPLETE %s\n' "${name}"
