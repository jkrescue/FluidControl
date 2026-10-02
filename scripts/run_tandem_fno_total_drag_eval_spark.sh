#!/usr/bin/env bash
# Guarded held-out evaluation of the seven-output total-drag PhysicsNeMo FNO.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model_dir="${MODEL_DIR:-artifacts/tandem_fno_total_drag_spark_30epoch}"
action_mode="${ACTION_MODE:-observed}"
visualizations="${VISUALIZATIONS_PER_HORIZON:-1}"
evaluation_data="${EVALUATION_DATA:-data/curated/tandem_cylinders_expanded_independent_v2}"
normalization_data="${NORMALIZATION_DATA:-data/curated/tandem_cylinders_expanded_independent_v2}"
evaluation_label="${EVALUATION_LABEL:-}"

[[ "${model_dir}" == /* ]] || model_dir="${root}/${model_dir}"
[[ "${model_dir}" == "${root}/"* ]] || {
    echo "Model directory must be inside project" >&2; exit 1;
}
relative_model_dir="${model_dir#"${root}/"}"
[[ -d "${model_dir}/best" ]] || { echo "Missing best checkpoint" >&2; exit 1; }
case "${action_mode}" in
    observed|zero|sign_flip|shuffle) ;;
    *) echo "Invalid ACTION_MODE: ${action_mode}" >&2; exit 2 ;;
esac
[[ "${visualizations}" =~ ^[0-9]+$ ]] || {
    echo "Invalid visualization count" >&2; exit 2;
}
[[ "${evaluation_data}" != /* && -d "${root}/${evaluation_data}" ]] || {
    echo "Evaluation data must be an existing project-relative directory" >&2; exit 2;
}
[[ "${normalization_data}" != /* && -d "${root}/${normalization_data}" ]] || {
    echo "Normalization data must be an existing project-relative directory" >&2; exit 2;
}
[[ -z "${evaluation_label}" || "${evaluation_label}" =~ ^[a-zA-Z0-9_-]+$ ]] || {
    echo "Invalid evaluation label" >&2; exit 2;
}
docker image inspect fluid-control-physicsnemo:2.2.2 >/dev/null

suffix=""
if [[ -n "${evaluation_label}" ]]; then suffix="_${evaluation_label}"; fi
if [[ "${action_mode}" != observed ]]; then suffix+="_${action_mode}"; fi

docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONPATH=/workspace/src --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace fluid-control-physicsnemo:2.2.2 \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/evaluate_tandem_fno.py \
        --data "${evaluation_data}" \
        --normalization-data "${normalization_data}" \
        --config conf/tandem_fno_total_drag.yaml \
        --checkpoint-dir "/workspace/${relative_model_dir}/best" \
        --output "/workspace/${relative_model_dir}/heldout_evaluation${suffix}.json" \
        --visualization-dir "/workspace/${relative_model_dir}/heldout_figures${suffix}" \
        --split test --horizons 1 10 50 100 --segment-stride 25 \
        --action-mode "${action_mode}" \
        --segment-metrics-output "/workspace/${relative_model_dir}/heldout_segments${suffix}.json" \
        --evaluation-batch-size 4 \
        --visualizations-per-horizon "${visualizations}"
