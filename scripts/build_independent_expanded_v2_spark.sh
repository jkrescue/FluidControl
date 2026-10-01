#!/usr/bin/env bash
# Build a leakage-free 24/4/4 dataset without modifying the original 32 trajectories.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
source_data="data/curated/tandem_cylinders_expanded_v1"
target_data="data/curated/tandem_cylinders_expanded_independent_v2"
curator_log="artifacts/tandem_cylinders/expanded_spark_curator.log"
replacement_log="artifacts/tandem_cylinders/expanded_edge_replacements_cfd.log"
python=".venv-curator-py312/bin/python"

wait_for_marker() {
    local marker="$1" log="$2" session="$3"
    while ! rg -q "^${marker}$" "${log}" 2>/dev/null; do
        if ! tmux has-session -t "${session}" 2>/dev/null; then
            echo "Missing ${marker}; producer ${session} has stopped" >&2
            exit 1
        fi
        sleep 60
    done
}

wait_for_marker EXPANDED_SPARK_CURATOR_OK "${curator_log}" fluid-control-curator-full
wait_for_marker "EXPANDED_CFD_CHUNK_OK expanded_validation_04 expanded_test_04" \
    "${replacement_log}" fluid-control-edge-replacements

[[ -s "${source_data}/manifest.json" && -s "${source_data}/normalization.json" ]]
[[ ! -e "${target_data}/manifest.json" ]] || {
    echo "Refusing to overwrite completed independent dataset" >&2
    exit 1
}
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < 150 )); then
    echo "Only ${free_gib} GiB disk free; refusing dataset assembly" >&2
    exit 75
fi

for split in train validation test; do
    mkdir -p "${target_data}/${split}"
    for source_path in "${source_data}/${split}"/*.h5; do
        name="${source_path##*/}"
        if [[ "${split}/${name}" == validation/expanded_validation_03.h5 ||
              "${split}/${name}" == test/expanded_test_03.h5 ]]; then
            continue
        fi
        target_path="${target_data}/${split}/${name}"
        if [[ -e "${target_path}" ]]; then
            [[ "${source_path}" -ef "${target_path}" ]] || {
                echo "Existing target is not the validated source hardlink: ${target_path}" >&2
                exit 1
            }
        else
            ln "${source_path}" "${target_path}"
        fi
    done
done
for split in train validation test; do
    count="$(find "${target_data}/${split}" -maxdepth 1 -name '*.h5' -type f | wc -l)"
    case "${split}:${count}" in
        train:24|validation:3|test:3) ;;
        *) echo "Unexpected pre-replacement count ${split}:${count}" >&2; exit 1 ;;
    esac
done
echo "INDEPENDENT_V2_BASE_HARDLINKS_OK"

for name in expanded_validation_04 expanded_test_04; do
    vtk="cfd/tandem_cylinders/cases/${name}/VTK_curator"
    if [[ ! -d "${vtk}" ]]; then
        bash scripts/export_tandem_vtk.sh "${name}"
    fi
    count="$(find "${vtk}" -name internal.vtu -type f | wc -l)"
    [[ "${count}" -eq 801 ]] || { echo "${name}: expected 801 VTK frames" >&2; exit 1; }
done

if [[ -e "${target_data}/validation/expanded_validation_04.h5" ||
      -e "${target_data}/test/expanded_test_04.h5" ]]; then
    echo "Refusing to overwrite partially curated replacements" >&2
    exit 1
fi
"${python}" scripts/curate_tandem_cfd.py \
    --profile expanded_with_replacements \
    --cases-root cfd/tandem_cylinders/cases \
    --output "${target_data}" --nx 256 --ny 128 \
    --cases expanded_validation_04 expanded_test_04 \
    --defer-finalize --jobs 2 --backend process_pool
"${python}" scripts/validate_expanded_curated_cases.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases \
    expanded_validation_04 expanded_test_04

"${python}" scripts/curate_tandem_cfd.py \
    --profile expanded_independent_v2 --output "${target_data}" \
    --nx 256 --ny 128 --finalize-only
mapfile -t names < <(
    find "${target_data}/train" "${target_data}/validation" "${target_data}/test" \
        -maxdepth 1 -type f -name '*.h5' -printf '%f\n' |
        sed 's/\.h5$//' | sort
)
[[ "${#names[@]}" -eq 32 ]]
"${python}" scripts/validate_expanded_curated_cases.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases "${names[@]}"
"${python}" scripts/validate_tandem_curated.py \
    --data "${target_data}" \
    --output artifacts/tandem_cylinders/expanded_independent_v2_curated_validation.json
python3 scripts/audit_tandem_split_integrity.py \
    --data "${target_data}" --cases-root cfd/tandem_cylinders/cases \
    --output artifacts/tandem_cylinders/expanded_independent_v2_split_integrity.json
"${python}" scripts/audit_curated_control_response.py \
    --data "${target_data}" --expected-count 32 \
    --output artifacts/tandem_cylinders/expanded_independent_v2_control_response.json
echo EXPANDED_INDEPENDENT_V2_OK

bash scripts/run_tandem_fno_pipeline_spark.sh \
    >> artifacts/tandem_cylinders/expanded_fno_pipeline_spark.log 2>&1
echo EXPANDED_INDEPENDENT_V2_FNO_PIPELINE_OK
