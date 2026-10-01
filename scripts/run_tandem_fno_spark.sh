#!/usr/bin/env bash
# Single-GPU DGX Spark launcher; host Python environments are untouched.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo:2.2.2"
output="${OUTPUT_DIR:-artifacts/tandem_fno_expanded_spark_v1}"
batch_size="${BATCH_SIZE:-4}"
smoke="${SMOKE:-false}"
fraction="0.20"
[[ "${output}" == /* ]] || output="${root}/${output}"
[[ "${output}" == "${root}/"* ]] || { echo "Output must be inside project" >&2; exit 1; }
[[ -s data/curated/tandem_cylinders_expanded_v1/manifest.json ]] || {
    echo "Curated expanded-v1 manifest is not yet ready" >&2; exit 1;
}
docker image inspect "${image}" >/dev/null
mkdir -p "${output}"

overrides=(
    "data.root=/workspace/data/curated/tandem_cylinders_expanded_v1"
    "output_dir=/workspace/${output#${root}/}"
    "training.batch_size=${batch_size}"
    "training.gpu_memory_fraction=${fraction}"
    "training.smoke=${smoke}"
    "data.num_streams=1"
)
if [[ -n "${EPOCHS:-}" ]]; then
    overrides+=("training.epochs=${EPOCHS}")
fi

docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONPATH=/workspace/src --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction "${fraction}" --margin-gib 4 -- \
      torchrun --standalone --nproc_per_node=1 scripts/train_tandem_fno.py \
      --config-name tandem_fno_expanded "${overrides[@]}" "$@"
