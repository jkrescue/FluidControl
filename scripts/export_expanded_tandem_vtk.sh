#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
workers="${1:-4}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: export_expanded_tandem_vtk.sh [positive-worker-count]' >&2
    exit 2
fi

log="${project_root}/artifacts/tandem_cylinders/expanded_vtk_export.log"
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite export log: ${log}" >&2
    exit 1
fi
mkdir -p "$(dirname "$log")"

python3 "${project_root}/cfd/tandem_cylinders/make_expanded_control_dataset.py" --list \
    | xargs -n1 -P "$workers" bash "${project_root}/scripts/export_tandem_vtk.sh" \
    2>&1 | tee "$log"

expected=32
cases=0
files=0
while IFS= read -r name; do
    count="$(find "${project_root}/cfd/tandem_cylinders/cases/${name}/VTK_curator" \
        -name internal.vtu -type f | wc -l | tr -d ' ')"
    [[ "$count" == 801 ]] || { echo "${name}: expected 801 files, found ${count}" >&2; exit 1; }
    cases=$((cases + 1))
    files=$((files + count))
done < <(python3 "${project_root}/cfd/tandem_cylinders/make_expanded_control_dataset.py" --list)
[[ "$cases" == "$expected" ]] || { echo "Expected ${expected} cases, found ${cases}" >&2; exit 1; }
echo "EXPANDED_VTK_EXPORT_OK cases=${cases} files=${files}" | tee -a "$log"
