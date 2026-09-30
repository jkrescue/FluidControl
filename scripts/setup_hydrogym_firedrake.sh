#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${HYDROGYM_SOURCE_DIR:-${repo_root}/.tools/hydrogym}"
commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
image="fluid-control/hydrogym-firedrake:${commit}"
min_free_gib=25

available_bytes="$(df -B1 --output=avail "${repo_root}" | tail -n 1 | tr -d ' ')"
if (( available_bytes < min_free_gib * 1024 * 1024 * 1024 )); then
  echo "HydroGym setup requires at least ${min_free_gib} GiB free." >&2
  exit 1
fi

if [[ ! -d "${source_dir}/.git" ]]; then
  mkdir -p "$(dirname "${source_dir}")"
  git clone https://github.com/dynamicslab/hydrogym.git "${source_dir}"
fi

git -C "${source_dir}" fetch origin "${commit}"
git -C "${source_dir}" checkout --detach "${commit}"
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${commit}"

docker build \
  --progress=plain \
  --build-arg "HYDROGYM_COMMIT=${commit}" \
  --file "${repo_root}/docker/hydrogym/Dockerfile.firedrake" \
  --tag "${image}" \
  "${source_dir}"

docker run --rm "${image}" bash -lc '
  source /home/USER/firedrake/bin/activate
  python - <<"PY"
import firedrake
import hydrogym
import mpi4py
import numpy
import stable_baselines3
import torch
from petsc4py import PETSc

print("numpy", numpy.__version__)
print("mpi4py", mpi4py.__version__)
print("petsc", PETSc.Sys.getVersion())
print("torch", torch.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("HYDROGYM_FIREDRAKE_IMAGE_OK")
PY
'
