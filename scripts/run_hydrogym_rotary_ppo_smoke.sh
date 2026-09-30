#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
image="fluid-control/hydrogym-firedrake:${commit}"
output_dir="${repo_root}/artifacts/hydrogym/ppo_smoke"
cache_dir="${repo_root}/artifacts/hydrogym/cache"

rm -rf "${output_dir}"
mkdir -p "${output_dir}" "${cache_dir}"

docker run --rm \
  --mount "type=bind,src=${output_dir},dst=/work" \
  --mount "type=bind,src=${cache_dir},dst=/home/USER/.cache" \
  --env PYTHONHASHSEED=42 \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "${image}" bash -lc '
    source /home/USER/firedrake/bin/activate
    cd /home/USER/hydrogym/examples/firedrake/getting_started
    python train_sb3_firedrake.py \
      --env rotary_cylinder \
      --reynolds 100 \
      --mesh medium \
      --dt 0.01 \
      --obs-type pressure_probes \
      --num-substeps 1 \
      --algo PPO \
      --total-timesteps 256 \
      --n-steps 64 \
      --batch-size 32 \
      --learning-rate 0.0003 \
      --gamma 0.99 \
      --save-freq 10000 \
      --log-dir /work/logs
  '
