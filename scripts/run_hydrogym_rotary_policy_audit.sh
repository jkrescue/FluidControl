#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
image="fluid-control/hydrogym-firedrake:${commit}"
ppo_root="${repo_root}/artifacts/hydrogym/ppo_smoke/logs"
run_dir="$(find "${ppo_root}" -mindepth 1 -maxdepth 1 -type d -name 'PPO_Firedrake_rotary_cylinder_*' | sort | tail -n 1)"
output_dir="${repo_root}/artifacts/hydrogym/policy_audit"
cache_dir="${repo_root}/artifacts/hydrogym/cache"

test -n "${run_dir}"
test -f "${run_dir}/model_final.zip"
test -f "${run_dir}/vec_normalize_final.pkl"
rm -rf "${output_dir}"
mkdir -p "${output_dir}" "${cache_dir}"

docker run --rm \
  --mount "type=bind,src=${repo_root}/scripts/evaluate_hydrogym_rotary_policies.py,dst=/work/evaluate.py,readonly" \
  --mount "type=bind,src=${run_dir},dst=/work/model,readonly" \
  --mount "type=bind,src=${output_dir},dst=/work/output" \
  --mount "type=bind,src=${cache_dir},dst=/home/USER/.cache" \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "${image}" bash -lc '
    source /home/USER/firedrake/bin/activate
    python /work/evaluate.py \
      --model /work/model/model_final.zip \
      --vec-normalize /work/model/vec_normalize_final.pkl \
      --output /work/output \
      --steps 200 \
      --warmup 50 \
      --seed 42
  '
