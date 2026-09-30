#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# == 0 )); then
    echo 'Usage: run_expanded_cfd_chunk.sh CASE [CASE ...]' >&2
    exit 2
fi
names=("$@")
workers="${EXPANDED_CFD_WORKERS:-4}"
log="${root}/../../artifacts/tandem_cylinders/expanded_cfd_streaming.log"

printf '%s\n' "${names[@]}" \
    | xargs -n1 -P "$workers" bash "${root}/run_expanded_dynamic_case.sh" \
    2>&1 | tee -a "$log"

python3 "${root}/validate_expanded_control_dataset.py" "${names[@]}" \
    2>&1 | tee -a "$log"
echo "EXPANDED_CFD_CHUNK_OK ${names[*]}" | tee -a "$log"
