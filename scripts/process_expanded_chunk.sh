#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if (( $# == 0 )); then
    echo 'Usage: process_expanded_chunk.sh CASE [CASE ...]' >&2
    exit 2
fi
names=("$@")
output="${project_root}/data/curated/tandem_cylinders_expanded_v1"
chunk_log="${project_root}/artifacts/tandem_cylinders/expanded_streaming_chunks.log"
cd "$project_root"

for name in "${names[@]}"; do
    bash scripts/export_tandem_vtk.sh "$name"
done

.venv-curator/bin/python scripts/curate_tandem_cfd.py \
    --profile expanded_v1 \
    --cases-root cfd/tandem_cylinders/cases \
    --output "$output" \
    --nx 256 --ny 128 \
    --cases "${names[@]}" \
    --defer-finalize \
    --jobs 2 --backend process_pool \
    2>&1 | tee -a "$chunk_log"

.venv-curator/bin/python scripts/validate_expanded_curated_cases.py \
    --data "$output" "${names[@]}" \
    2>&1 | tee -a "$chunk_log"

for name in "${names[@]}"; do
    case_dir="${project_root}/cfd/tandem_cylinders/cases/${name}"
    rm -rf "${case_dir}/VTK_curator"
    while IFS= read -r directory; do
        base="$(basename "$directory")"
        if [[ "$base" != "80" ]]; then
            rm -rf "$directory"
        fi
    done < <(find "$case_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
        -regex '.*/[0-9]+([.][0-9]+)?' | sort -V)
done

echo "EXPANDED_CHUNK_OK ${names[*]}" | tee -a "$chunk_log"
