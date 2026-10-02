#!/usr/bin/env bash
# Curate and validate the fixed independent-phase CFD panel on DGX Spark.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
output="data/curated/tandem_cylinders_phase_v1"
python=".venv-curator-py312/bin/python"
[[ -x "${python}" ]] || { echo "Missing isolated Curator environment" >&2; exit 1; }
mapfile -t names < <(python3 cfd/tandem_cylinders/make_phase_diverse_control_dataset.py --list)
[[ "${#names[@]}" -eq 4 ]] || { echo "Expected 4 fixed phase cases" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 1; }

python3 cfd/tandem_cylinders/validate_phase_diverse_cfd.py \
    --output artifacts/tandem_cylinders/phase_diverse_v1_cfd_qc.json
for name in "${names[@]}"; do
    vtk="cfd/tandem_cylinders/cases/${name}/VTK_curator"
    if [[ ! -d "${vtk}" ]]; then
        bash scripts/export_tandem_vtk.sh "${name}"
    fi
    count="$(find "${vtk}" -name internal.vtu -type f | wc -l | tr -d ' ')"
    [[ "${count}" -eq 241 ]] || {
        echo "${name}: expected 241 VTK frames, found ${count}" >&2
        exit 1
    }
done

"${python}" scripts/curate_tandem_cfd.py \
    --profile phase_v1 --cases-root cfd/tandem_cylinders/cases \
    --output "${output}" --nx 256 --ny 128 --jobs 4 --backend process_pool
"${python}" scripts/validate_tandem_curated.py \
    --data "${output}" \
    --output artifacts/tandem_cylinders/phase_curated_validation_spark.json
echo PHASE_DIVERSE_SPARK_CURATOR_OK
