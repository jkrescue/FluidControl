#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
image="fluid-control/hydrogym-firedrake:${commit}"
output_dir="${repo_root}/artifacts/hydrogym/rotary_smoke"
cache_dir="${repo_root}/artifacts/hydrogym/cache"

rm -rf "${output_dir}"
mkdir -p "${output_dir}" "${cache_dir}"

docker run --rm \
  --mount "type=bind,src=${output_dir},dst=/home/USER/hydrogym/examples/firedrake/getting_started/output" \
  --mount "type=bind,src=${cache_dir},dst=/home/USER/.cache" \
  "${image}" bash -lc '
    source /home/USER/firedrake/bin/activate
    cd /home/USER/hydrogym/examples/firedrake/getting_started
    python test_firedrake_env.py \
      --environment rotary_cylinder \
      --reynolds 100 \
      --mesh-resolution medium \
      --num-steps 3 \
      --num-episodes 1 \
      --seed 42 \
      --verbose
  '
