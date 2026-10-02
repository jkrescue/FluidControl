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

start_time="80"
end_time="160"
expected_frames="801"
if [[ -f "${case_dir}/case_config.json" ]]; then
    readarray -t export_config < <(python3 - "${case_dir}/case_config.json" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{cfg.get('start_time', 80):g}")
print(f"{cfg.get('end_time', 160):g}")
print(int(cfg.get("expected_frames", 801)))
PY
    )
    start_time="${export_config[0]}"
    end_time="${export_config[1]}"
    expected_frames="${export_config[2]}"
fi

bash "${cfd_root}/run_openfoam.sh" foamToVTK \
    -case "/case/cases/${name}" \
    -time "${start_time}:${end_time}" \
    -fields '(U p)' \
    -no-boundary \
    -name VTK_curator \
    >"${case_dir}/log.foamToVTK_curator" 2>&1

count="$(find "${case_dir}/VTK_curator" -name internal.vtu -type f | wc -l | tr -d ' ')"
if [[ "$count" != "${expected_frames}" ]]; then
    echo "Expected ${expected_frames} VTK snapshots for ${name}, found ${count}" >&2
    exit 1
fi
echo "Exported ${name}: ${count} VTK snapshots"
