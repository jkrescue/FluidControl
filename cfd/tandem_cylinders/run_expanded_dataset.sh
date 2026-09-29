#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workers="${1:-4}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: run_expanded_dataset.sh [positive-worker-count]' >&2
    exit 2
fi

artifact_root="${root}/../../artifacts/tandem_cylinders"
batch_log="${artifact_root}/expanded_cfd_batch.log"
if [[ -e "$batch_log" ]]; then
    echo "Refusing to overwrite batch log: ${batch_log}" >&2
    exit 1
fi
mkdir -p "$artifact_root"

mapfile -t names < <(python3 "${root}/make_expanded_control_dataset.py" --list)
if [[ "${#names[@]}" -ne 32 ]]; then
    echo "Expected 32 expanded cases, found ${#names[@]}" >&2
    exit 1
fi
for name in "${names[@]}"; do
    if [[ ! -f "${root}/cases/${name}/case_config.json" ]]; then
        echo "Missing generated case: ${name}" >&2
        exit 1
    fi
    if [[ -e "${root}/cases/${name}/log.pimpleFoam" ]]; then
        echo "Refusing to reuse case with an existing solver log: ${name}" >&2
        exit 1
    fi
done

printf '%s\n' "${names[@]}" \
    | xargs -n1 -P "$workers" bash "${root}/run_expanded_dynamic_case.sh" \
    2>&1 | tee "$batch_log"

echo "EXPANDED_CFD_BATCH_OK cases=${#names[@]} workers=${workers}" | tee -a "$batch_log"
