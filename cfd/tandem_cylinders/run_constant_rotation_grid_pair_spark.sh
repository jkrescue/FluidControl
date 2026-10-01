#!/usr/bin/env bash
# Matched Re=100, omega=+1 coarse/medium OpenFOAM grid-sensitivity run.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(cd "${root}/../.." && pwd -P)"
cd "${project}"
coarse="cfd/tandem_cylinders/cases/control_small_p100"
medium="cfd/tandem_cylinders/cases/control_grid_p100_medium"
[[ ! -e "${coarse}" && ! -e "${medium}" ]] || {
    echo "Refusing to overwrite a pre-existing grid-check case" >&2
    exit 2
}
free_gib="$(df -BG --output=avail "${project}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 300 )) || { echo "Need 300 GiB disk free; found ${free_gib}" >&2; exit 75; }
available_kib="$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) || {
    echo "Need 40 GiB MemAvailable before extra CFD work" >&2
    exit 75
}
mkdir -p artifacts/tandem_cylinders

python3 "${root}/make_small_control_dataset.py" p100
python3 "${root}/make_rotation_grid_check.py"

bash "${root}/run_openfoam.sh" blockMesh -case /case/cases/control_small_p100 \
    >"${coarse}/log.blockMesh" 2>&1
bash "${root}/run_openfoam.sh" checkMesh -case /case/cases/control_small_p100 \
    >"${coarse}/log.checkMesh" 2>&1
rg -q 'Mesh OK' "${coarse}/log.checkMesh"
bash "${root}/run_openfoam.sh" setFields -case /case/cases/control_small_p100 \
    >"${coarse}/log.setFields" 2>&1
echo GRID_CHECK_COARSE_MESH_READY
bash "${root}/run_control_case.sh" p100
echo GRID_CHECK_COARSE_SOLVE_OK

bash "${root}/run_rotation_grid_check.sh"
echo GRID_CHECK_MEDIUM_SOLVE_OK
python3 "${root}/compare_convergence.py" control_small_p100 control_grid_p100_medium |
    tee artifacts/tandem_cylinders/constant_p100_grid_comparison.json
echo GRID_CHECK_COMPARISON_READY
