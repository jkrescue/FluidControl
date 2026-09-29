#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "$name" in
    expanded_train_0[0-9]|expanded_train_1[0-9]|expanded_train_2[0-3]|expanded_validation_0[0-3]|expanded_test_0[0-3]) ;;
    *) echo 'Usage: run_expanded_dynamic_case.sh <expanded_train_00..23|expanded_validation_00..03|expanded_test_00..03>' >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
log="${target}/log.pimpleFoam"
if [[ ! -f "${target}/case_config.json" || ! -f "${target}/80/U" ]]; then
    echo "Missing configured expanded case: ${target}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite solver log: ${log}" >&2
    exit 1
fi
if find "$target" -maxdepth 1 -type d \( -name '8[1-9]*' -o -name '9[0-9]*' -o -name '1[0-9][0-9]*' \) -print -quit | grep -q .; then
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
