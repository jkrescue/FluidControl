#!/usr/bin/env bash
# Guarded held-out rollout evaluation of the Spark PhysicsNeMo FNO pilot.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model_dir="${MODEL_DIR:-artifacts/tandem_fno_expanded_spark_5epoch}"
[[ "${model_dir}" == /* ]] || model_dir="${root}/${model_dir}"
[[ "${model_dir}" == "${root}/"* ]] || { echo "Model directory must be inside project" >&2; exit 1; }
[[ -d "${model_dir}/best" ]] || { echo "Missing PhysicsNeMo best checkpoint" >&2; exit 1; }
docker image inspect fluid-control-physicsnemo:2.2.2 >/dev/null
action_mode="${ACTION_MODE:-observed}"
case "${action_mode}" in
    observed|zero|sign_flip|shuffle) ;;
    *) echo "Invalid ACTION_MODE: ${action_mode}" >&2; exit 2 ;;
esac
visualizations="${VISUALIZATIONS_PER_HORIZON:-1}"
[[ "${visualizations}" =~ ^[0-9]+$ ]] || { echo "Invalid visualization count" >&2; exit 2; }
result_suffix=""
if [[ "${action_mode}" != observed ]]; then result_suffix="_${action_mode}"; fi

docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONPATH=/workspace/src --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace fluid-control-physicsnemo:2.2.2 \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/evaluate_tandem_fno.py \
        --data data/curated/tandem_cylinders_expanded_v1 \
        --config conf/tandem_fno_expanded.yaml \
        --checkpoint-dir "/workspace/${model_dir#${root}/}/best" \
        --output "/workspace/${model_dir#${root}/}/heldout_evaluation${result_suffix}.json" \
        --visualization-dir "/workspace/${model_dir#${root}/}/heldout_figures${result_suffix}" \
        --split test --horizons 1 10 50 --segment-stride 50 \
        --action-mode "${action_mode}" \
        --segment-metrics-output "/workspace/${model_dir#${root}/}/heldout_segments${result_suffix}.json" \
        --evaluation-batch-size 4 --visualizations-per-horizon "${visualizations}"
