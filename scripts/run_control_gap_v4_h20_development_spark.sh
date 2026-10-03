#!/usr/bin/env bash
# Validation-only v4 development ablation; no frozen-test access.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data="data/curated/tandem_cylinders_control_gap_v4"
parent="artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/best"
name="tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch"
smoke="${DEVELOPMENT_SMOKE:-false}"

[[ "$(uname -m)" == aarch64 ]] || { echo "ARM64 DGX Spark required" >&2; exit 2; }
[[ "${smoke}" == true || "${smoke}" == false ]] || { echo "Invalid DEVELOPMENT_SMOKE" >&2; exit 2; }
[[ -s "${data}/manifest.json" && -s "${data}/normalization.json" && -d "${parent}" ]] || {
    echo "Required curated v4 data or v3 parent checkpoint missing" >&2; exit 2;
}
actual_image="$(docker image inspect "${image}" --format '{{.Id}}')"
[[ "${actual_image}" == "${expected_image}" ]] || { echo "PhysicsNeMo image ID mismatch" >&2; exit 2; }

output="artifacts/${name}"
epochs=5
extra=()
if [[ "${smoke}" == true ]]; then
    output="${output}_full_horizon_smoke"
    epochs=1
    extra+=(training.max_train_batches=1 training.max_validation_batches=1)
fi
[[ ! -e "${output}/training_history.json" ]] || {
    echo "Refusing to overwrite completed development run" >&2; exit 2;
}
mkdir -p "${output}"

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
    "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.15 --margin-gib 4 -- \
      python -u scripts/train_tandem_fno_rollout.py \
        --config-name tandem_fno_total_drag_rollout_h20_rear_drag \
        "data.root=${data}" "output_dir=${output}" \
        "training.initial_checkpoint=${parent}" \
        "training.epochs=${epochs}" training.batch_size=4 \
        training.rollout_steps=20 training.seed=20261003 \
        training.gpu_memory_fraction=0.15 data.num_streams=1 \
        "${extra[@]}"
