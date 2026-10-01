#!/usr/bin/env bash
# Resume the fixed expanded-v1 OpenFOAM dataset on a DGX Spark without overwriting cases.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${root}/../.." && pwd -P)"
cd "${project_root}"

min_free_gib="${MIN_FREE_GIB:-150}"
workers="${EXPANDED_CFD_WORKERS:-3}"
if ! [[ "${min_free_gib}" =~ ^[0-9]+$ && "${workers}" =~ ^[1-4]$ ]]; then
    echo "MIN_FREE_GIB must be an integer and EXPANDED_CFD_WORKERS must be 1..4" >&2
    exit 2
fi

mapfile -t names < <(python3 "${root}/make_expanded_control_dataset.py" --list)
pending=()
active=()
for name in "${names[@]}"; do
    log="${root}/cases/${name}/log.pimpleFoam"
    if [[ -e "${log}" ]]; then
        if grep -qx End "${log}"; then
            echo "Reusing completed CFD: ${name}"
        elif pgrep -f "pimpleFoam -case /case/cases/${name}" >/dev/null; then
            echo "Waiting for already active CFD: ${name}"
            active+=("${name}")
        else
            echo "Incomplete existing CFD; refusing to overwrite: ${name}" >&2
            exit 1
        fi
    else
        pending+=("${name}")
    fi
done

if (( ${#pending[@]} )); then
    free_gib="$(df -BG --output=avail "${project_root}" | tail -n 1 | tr -dc '0-9')"
    if (( free_gib < min_free_gib )); then
        echo "Only ${free_gib} GiB free; requires ${min_free_gib} GiB" >&2
        exit 75
    fi
    echo "Starting ${#pending[@]} cases with ${workers} OpenFOAM workers; free=${free_gib} GiB"
    EXPANDED_CFD_WORKERS="${workers}" bash "${root}/run_expanded_cfd_chunk.sh" "${pending[@]}"
fi

for name in "${active[@]}"; do
    log="${root}/cases/${name}/log.pimpleFoam"
    until grep -qx End "${log}"; do
        if ! pgrep -f "pimpleFoam -case /case/cases/${name}" >/dev/null; then
            echo "Active CFD stopped without End: ${name}" >&2
            exit 1
        fi
        sleep 30
    done
done

python3 "${root}/validate_expanded_control_dataset.py" \
    --write "${project_root}/artifacts/tandem_cylinders/expanded_v1_cfd_qc.json"
echo EXPANDED_SPARK_CFD_OK
