#!/usr/bin/env bash
# Guarded single-GPU rollout fine-tuning for an isolated DGX Spark worker.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="${PHYSICSNEMO_IMAGE:-fluid-control-physicsnemo:2.2.2}"
output_dir="${OUTPUT_DIR:-artifacts/tandem_fno_total_drag_rollout_seed20261003}"
data_root="${DATA_ROOT:-data/curated/tandem_cylinders_expanded_independent_v2}"
epochs="${EPOCHS:-10}"
batch_size="${BATCH_SIZE:-4}"
rollout_steps="${ROLLOUT_STEPS:-10}"
smoke="${SMOKE:-false}"

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "This runner is restricted to the ARM64 DGX Spark workers" >&2
    exit 2
}
[[ "${output_dir}" != /* && "${data_root}" != /* ]] || {
    echo "Output and data paths must be project-relative" >&2
    exit 2
}
[[ "${epochs}" =~ ^[1-9][0-9]*$ && "${batch_size}" =~ ^[1-9][0-9]*$ \
    && "${rollout_steps}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Epochs, batch size and rollout steps must be positive integers" >&2
    exit 2
}
[[ "${smoke}" == "true" || "${smoke}" == "false" ]] || {
    echo "SMOKE must be true or false" >&2
    exit 2
}
[[ -d "${data_root}/train" && -d "${data_root}/validation" ]] || {
    echo "Missing curated train/validation data: ${data_root}" >&2
    exit 2
}
[[ -d artifacts/tandem_fno_total_drag_spark_30epoch/best ]] || {
    echo "Missing initial epoch-30 checkpoint" >&2
    exit 2
}
docker image inspect "${image}" >/dev/null
mkdir -p "${output_dir}"

overrides=(
    "data.root=${data_root}"
    "output_dir=${output_dir}"
    "training.epochs=${epochs}"
    "training.batch_size=${batch_size}"
    "training.rollout_steps=${rollout_steps}"
    "training.smoke=${smoke}"
    "training.gpu_memory_fraction=0.20"
    "data.num_streams=1"
)
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
    overrides+=("training.max_train_batches=${MAX_TRAIN_BATCHES}")
fi
if [[ -n "${MAX_VALIDATION_BATCHES:-}" ]]; then
    overrides+=("training.max_validation_batches=${MAX_VALIDATION_BATCHES}")
fi

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/train_tandem_fno_rollout.py \
        --config-name tandem_fno_total_drag_rollout \
        "${overrides[@]}"
