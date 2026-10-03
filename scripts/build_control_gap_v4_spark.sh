#!/usr/bin/env bash
# Build immutable 28/4/5 Curator data without altering the validated v3 files.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
source_data="data/curated/tandem_cylinders_gate_b_aug_v3"
target_data="data/curated/tandem_cylinders_control_gap_v4"
curator=".venv-curator-py312/bin/python"
cases_root="cfd/tandem_cylinders/cases"
names=(train_signed_pulse_p_v4_20261003 train_signed_pulse_m_v4_20261003)
qc="artifacts/tandem_cylinders/signed_pulse_train_cfd_qc_20261003.json"

[[ -x "${curator}" && -s "${source_data}/manifest.json" ]] || {
    echo "Missing isolated official Curator environment or validated v3 dataset" >&2; exit 1;
}
[[ -s "${qc}" ]] && jq -e '.status == "SIGNED_PULSE_REAL_CFD_TRAIN_PAIR_OK" and (.cases|length)==2' "${qc}" >/dev/null || {
    echo "Signed-pulse OpenFOAM cases have not passed fail-closed CFD QC" >&2; exit 1;
}
[[ ! -e "${target_data}/manifest.json" ]] || {
    echo "Refusing to overwrite finalized v4 data profile" >&2; exit 1;
}
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 150 )) || { echo "Only ${free_gib} GiB free; 150 GiB required" >&2; exit 75; }

for split in train validation test; do
    mkdir -p "${target_data}/${split}"
    for source_path in "${source_data}/${split}"/*.h5; do
        target_path="${target_data}/${split}/${source_path##*/}"
        if [[ -e "${target_path}" ]]; then
            [[ "${source_path}" -ef "${target_path}" ]] || {
                echo "Unexpected non-hardlinked inherited file: ${target_path}" >&2; exit 1;
            }
        else
            ln "${source_path}" "${target_path}"
        fi
    done
done
for split_count in train:26 validation:4 test:5; do
    split="${split_count%%:*}"
    expected="${split_count##*:}"
    inherited=0
    for source_path in "${source_data}/${split}"/*.h5; do
        [[ "${source_path}" -ef "${target_data}/${split}/${source_path##*/}" ]] || exit 1
        inherited=$((inherited + 1))
    done
    [[ "${inherited}" -eq "${expected}" ]] || {
        echo "Inherited ${split} count ${inherited} != ${expected}" >&2; exit 1;
    }
done
echo CONTROL_GAP_V4_INHERITED_V3_HARDLINKS_OK

for name in "${names[@]}"; do
    vtk="${cases_root}/${name}/VTK_curator"
    if [[ ! -d "${vtk}" ]]; then
        bash scripts/export_tandem_vtk.sh "${name}"
    fi
    count="$(find "${vtk}" -name internal.vtu -type f | wc -l)"
    [[ "${count}" -eq 801 ]] || { echo "${name}: VTK count ${count} != 801" >&2; exit 1; }
done
echo CONTROL_GAP_V4_VTK_1602_OK

for name in "${names[@]}"; do
    target="${target_data}/train/${name}.h5"
    if [[ ! -e "${target}" ]]; then
        "${curator}" scripts/curate_tandem_cfd.py \
            --profile control_gap_v4 --cases-root "${cases_root}" \
            --output "${target_data}" --nx 256 --ny 128 \
            --cases "${name}" --defer-finalize
    fi
done
"${curator}" scripts/validate_expanded_curated_cases.py \
    --data "${target_data}" --cases-root "${cases_root}" "${names[@]}"
echo CONTROL_GAP_V4_NEW_CURATED_LABELS_OK

"${curator}" scripts/curate_tandem_cfd.py \
    --profile control_gap_v4 --output "${target_data}" \
    --nx 256 --ny 128 --finalize-only
"${curator}" scripts/validate_tandem_curated.py \
    --data "${target_data}" \
    --output artifacts/tandem_cylinders/control_gap_v4_curated_validation.json
python3 scripts/audit_tandem_split_integrity.py \
    --data "${target_data}" --cases-root "${cases_root}" \
    --expected-train 28 --expected-validation 4 --expected-test 5 \
    --output artifacts/tandem_cylinders/control_gap_v4_split_integrity.json
echo CONTROL_GAP_V4_CURATED_OK
