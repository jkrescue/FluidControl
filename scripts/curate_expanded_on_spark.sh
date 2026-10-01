#!/usr/bin/env bash
# Finish the fixed expanded-v1 dataset without deleting raw CFD or existing HDF5.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
output="data/curated/tandem_cylinders_expanded_v1"
python=".venv-curator-py312/bin/python"
[[ -x "${python}" ]] || { echo "Missing isolated Curator environment" >&2; exit 1; }
mapfile -t names < <(python3 cfd/tandem_cylinders/make_expanded_control_dataset.py --list)
[[ "${#names[@]}" -eq 32 ]] || { echo "Expected 32 fixed cases" >&2; exit 1; }

while [[ ! -s artifacts/tandem_cylinders/expanded_v1_cfd_qc.json ]]; do
    if ! tmux has-session -t fluid-control-expanded-resume 2>/dev/null; then
        echo "CFD producer stopped before writing full QC manifest" >&2
        exit 1
    fi
    sleep 30
done

free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < 150 )); then
    echo "Only ${free_gib} GiB disk free; at least 150 GiB required" >&2
    exit 75
fi

for name in "${names[@]}"; do
    vtk="cfd/tandem_cylinders/cases/${name}/VTK_curator"
    if [[ ! -d "${vtk}" ]]; then
        bash scripts/export_tandem_vtk.sh "${name}"
    fi
    count="$(find "${vtk}" -name internal.vtu -type f | wc -l | tr -d ' ')"
    [[ "${count}" -eq 801 ]] || { echo "${name}: expected 801 VTK frames, found ${count}" >&2; exit 1; }
done

pending=()
for name in "${names[@]}"; do
    case "${name}" in
        expanded_train_*) split=train ;;
        expanded_validation_*) split=validation ;;
        expanded_test_*) split=test ;;
    esac
    if [[ ! -s "${output}/${split}/${name}.h5" ]]; then
        pending+=("${name}")
    fi
done

if (( ${#pending[@]} )); then
    "${python}" scripts/curate_tandem_cfd.py \
        --profile expanded_v1 --cases-root cfd/tandem_cylinders/cases \
        --output "${output}" --nx 256 --ny 128 \
        --cases "${pending[@]}" --defer-finalize --jobs 4 --backend process_pool
fi
"${python}" scripts/validate_expanded_curated_cases.py \
    --data "${output}" --cases-root cfd/tandem_cylinders/cases "${names[@]}"
"${python}" scripts/curate_tandem_cfd.py \
    --profile expanded_v1 --output "${output}" --nx 256 --ny 128 --finalize-only
"${python}" scripts/validate_tandem_curated.py \
    --data "${output}" --output artifacts/tandem_cylinders/expanded_curated_validation_spark.json
echo EXPANDED_SPARK_CURATOR_OK
