#!/usr/bin/env bash
# GPU0 PPO run weighted toward the matched t=80 restart while retaining diverse starts.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
model_dir="artifacts/tandem_fno_expanded_spark_20epoch"
readiness="${model_dir}/control_readiness.json"
output="artifacts/hydrogym/tandem_ppo_t80_targeted_16384_spark"
log="${output}.log"
fraction="0.20"
[[ ! -e "${output}" && ! -e "${log}" ]] || { echo "Output or log already exists" >&2; exit 2; }
[[ -s "${readiness}" && -d "${model_dir}/best" ]] || { echo "Gate or checkpoint missing" >&2; exit 1; }
[[ "$(git -C .tools/hydrogym rev-parse HEAD)" == "4ab9854dea3d84e38a59c25e0f5835a00cf8225f" ]] || exit 1
docker image inspect "${image}" >/dev/null
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 150 )) || { echo "Insufficient disk free: ${free_gib} GiB" >&2; exit 75; }
available_gib="$(awk '/^MemAvailable:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
(( available_gib >= 40 )) || { echo "Insufficient memory available: ${available_gib} GiB" >&2; exit 75; }
mkdir -p artifacts/hydrogym

docker run --rm --network none --gpus 'device=0' --cpus 6 --memory 64g \
    --shm-size 2g --pids-limit 256 \
    --user "$(id -u):$(id -g)" \
    --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction "${fraction}" --margin-gib 4 -- \
    python -u scripts/train_tandem_hydrogym_ppo_pilot.py \
        --data data/curated/tandem_cylinders_expanded_independent_v2 \
        --config conf/tandem_fno_expanded.yaml \
        --checkpoint-dir "${model_dir}/best" \
        --readiness "${readiness}" \
        --output "${output}" --timesteps 16384 --episode-steps 32 \
        --checkpoint-interval 2048 --checkpoint-eval-steps 16 \
        --training-profile t80_targeted \
        --device cuda:0 --gpu-memory-fraction "${fraction}" \
    2>&1 | tee "${log}"
