#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cfd_root="${project_root}/cfd/tandem_cylinders"
name="${1:-}"

if [[ -z "$name" ]]; then
    echo 'Usage: bash scripts/export_tandem_vtk.sh CASE_NAME' >&2
    exit 2
fi

case_dir="${cfd_root}/cases/${name}"
if [[ ! -d "$case_dir" ]]; then
    echo "Case does not exist: ${case_dir}" >&2
    exit 1
fi
if [[ -e "${case_dir}/VTK_curator" ]]; then
    echo "Refusing to overwrite existing VTK export: ${case_dir}/VTK_curator" >&2
    exit 1
fi

bash "${cfd_root}/run_openfoam.sh" foamToVTK \
    -case "/case/cases/${name}" \
    -time '80:160' \
    -fields '(U p)' \
    -no-boundary \
    -name VTK_curator \
    >"${case_dir}/log.foamToVTK_curator" 2>&1

count="$(find "${case_dir}/VTK_curator" -name internal.vtu -type f | wc -l | tr -d ' ')"
if [[ "$count" != 801 ]]; then
    echo "Expected 801 VTK snapshots for ${name}, found ${count}" >&2
    exit 1
fi
echo "Exported ${name}: ${count} VTK snapshots"
