#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "$name" in
    mpc_replay_*) ;;
    *) echo 'Usage: run_mpc_replay_case.sh mpc_replay_NAME' >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
if [[ ! -f "${target}/case_config.json" || ! -f "${target}/80/U" ]]; then
    echo "Missing configured MPC replay case: ${target}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite solver log: ${log}" >&2
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
