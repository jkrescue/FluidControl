#!/usr/bin/env bash
# Isolated ARM64 HydroGym/Firedrake PPO using official public restart fields.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-hydrogym-spark:2025.10.2"
cache="${root}/artifacts/hydrogym/firedrake_arm64_smoke/cache"
output="${1:-${root}/artifacts/hydrogym/rotary_physical_v1}"
timesteps="${2:-8192}"
eval_steps="${3:-600}"
warmup="${4:-100}"

[[ "${output}" == /* ]] || output="${root}/${output}"
[[ -d "${cache}" ]] || { echo "Missing public HydroGym checkpoint cache" >&2; exit 1; }
[[ -f "${root}/scripts/train_hydrogym_rotary_spark.py" ]] || exit 1
docker image inspect "${image}" >/dev/null
mkdir -p "${output}"
if [[ -e "${output}/model_final.zip" ]]; then
    echo "Refusing to overwrite completed model in ${output}" >&2
    exit 1
fi

docker run --rm --network none --cpus 4 --memory 8g \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --mount "type=bind,src=${root}/scripts/train_hydrogym_rotary_spark.py,dst=/work/train.py,readonly" \
    --mount "type=bind,src=${cache},dst=/work/cache,readonly" \
    --mount "type=bind,src=${output},dst=/work/output" \
    "${image}" python3 -u /work/train.py \
      --cache /work/cache --output /work/output \
      --timesteps "${timesteps}" --eval-steps "${eval_steps}" --warmup "${warmup}"
