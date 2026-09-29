#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"

output="data/curated/tandem_cylinders_expanded_v1"
log="artifacts/tandem_cylinders/expanded_curator.log"
if [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite curated data: ${output}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite Curator log: ${log}" >&2
    exit 1
fi
mkdir -p "$(dirname "$log")"

set -o pipefail
/usr/bin/time -v .venv-curator/bin/python scripts/curate_tandem_cfd.py \
    --profile expanded_v1 \
    --cases-root cfd/tandem_cylinders/cases \
    --output "$output" \
    --nx 256 --ny 128 \
    --jobs 2 --backend process_pool \
    2>&1 | tee "$log"

echo "EXPANDED_CURATOR_OK" | tee -a "$log"
