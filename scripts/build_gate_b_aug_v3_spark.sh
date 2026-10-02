#!/usr/bin/env bash
# Assemble an immutable 26/4/5 Curator dataset from QC-passed OpenFOAM cases.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
source_data="data/curated/tandem_cylinders_expanded_independent_v2"
target_data="data/curated/tandem_cylinders_gate_b_aug_v3"
python=".venv-curator-py312/bin/python"
names=(expanded_train_24 expanded_train_25 expanded_test_05)

[[ -x "${python}" && -s "${source_data}/manifest.json" ]] || {
    echo "Missing isolated Curator environment or validated v2 dataset" >&2; exit 1;
}
rg -q '^EXPANDED_CFD_CHUNK_OK expanded_train_24 expanded_train_25 expanded_test_05$' \
    artifacts/tandem_cylinders/gate_b_aug_cfd.log || {
    echo "New OpenFOAM trajectories have not passed CFD QC" >&2; exit 1;
}
[[ ! -e "${target_data}/manifest.json" ]] || {
    echo "Refusing to overwrite completed augmented dataset" >&2; exit 1;
}
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 150 )) || {
    echo "Only ${free_gib} GiB disk free; 150 GiB required" >&2; exit 75;
}

for split in train validation test; do
    mkdir -p "${target_data}/${split}"
    for source_path in "${source_data}/${split}"/*.h5; do
        target_path="${target_data}/${split}/${source_path##*/}"
        if [[ -e "${target_path}" ]]; then
            [[ "${source_path}" -ef "${target_path}" ]] || {
                echo "Unexpected existing target: ${target_path}" >&2; exit 1;
            }
        else
            ln "${source_path}" "${target_path}"
        fi
    done
done
for split in train validation test; do
    count="$(find "${target_data}/${split}" -maxdepth 1 -name '*.h5' -type f | wc -l)"
    case "${split}:${count}" in
        train:24|validation:4|test:4) ;;
        *) echo "Unexpected inherited ${split} count: ${count}" >&2; exit 1 ;;
    esac
done
echo GATE_B_V3_BASE_HARDLINKS_OK

for name in "${names[@]}"; do
    vtk="cfd/tandem_cylinders/cases/${name}/VTK_curator"
    if [[ ! -d "${vtk}" ]]; then
        bash scripts/export_tandem_vtk.sh "${name}"
    fi
    count="$(find "${vtk}" -name internal.vtu -type f | wc -l)"
    [[ "${count}" -eq 801 ]] || { echo "${name}: expected 801 VTK frames" >&2; exit 1; }
done
echo GATE_B_V3_VTK_OK

for name in "${names[@]}"; do
    split=train
    [[ "${name}" == expanded_test_* ]] && split=test
    [[ ! -e "${target_data}/${split}/${name}.h5" ]] || {
        echo "Refusing to overwrite curated case: ${name}" >&2; exit 1;
    }
done
"${python}" scripts/curate_tandem_cfd.py \
    --profile gate_b_aug_v3 --cases-root cfd/tandem_cylinders/cases \
    --output "${target_data}" --nx 256 --ny 128 \
    --cases "${names[@]}" --defer-finalize --jobs 2 --backend process_pool
"${python}" scripts/validate_expanded_curated_cases.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases "${names[@]}"
echo GATE_B_V3_NEW_LABELS_OK

"${python}" scripts/curate_tandem_cfd.py \
    --profile gate_b_aug_v3 --output "${target_data}" \
    --nx 256 --ny 128 --finalize-only
mapfile -t all_names < <(
    find "${target_data}/train" "${target_data}/validation" "${target_data}/test" \
        -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort
)
[[ "${#all_names[@]}" -eq 35 ]]
"${python}" scripts/validate_expanded_curated_cases.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases "${all_names[@]}"
"${python}" scripts/validate_tandem_curated.py \
    --data "${target_data}" \
    --output artifacts/tandem_cylinders/gate_b_aug_v3_curated_validation.json
python3 scripts/audit_tandem_split_integrity.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases \
    --expected-train 26 --expected-validation 4 --expected-test 5 \
    --output artifacts/tandem_cylinders/gate_b_aug_v3_split_integrity.json
"${python}" scripts/audit_curated_control_response.py \
    --data "${target_data}" --expected-count 35 \
    --output artifacts/tandem_cylinders/gate_b_aug_v3_control_response.json
echo GATE_B_AUG_V3_CURATED_OK
