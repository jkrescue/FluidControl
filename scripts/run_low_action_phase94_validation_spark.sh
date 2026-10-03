#!/usr/bin/env bash
set -euo pipefail

root=/workspace/fluid_control
label="${1:?label required}"
model_dir="${2:?project-relative model directory required}"
normalization_data="${3:?project-relative training data directory required}"
data=data/curated/tandem_cylinders_low_action_phase94_validation_v1
result_root=artifacts/distributed_runs/control_gap_low_action_phase94_validation_v1_worker78/evaluations
image_id=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e

cd "${root}"
[[ "${label}" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "${model_dir}" != /* && -d "${model_dir}/best" ]]
[[ "${normalization_data}" != /* && -s "${normalization_data}/normalization.json" ]]
[[ -s "${data}/manifest.json" ]]
[[ "$(find "${data}/validation" -maxdepth 1 -type f -name '*.h5' | wc -l)" -eq 2 ]]
[[ ! -d "${data}/train" || -z "$(find "${data}/train" -type f -name '*.h5' -print -quit)" ]]
[[ ! -d "${data}/test" || -z "$(find "${data}/test" -type f -name '*.h5' -print -quit)" ]]
jq -e '.profile == "low_action_phase94_validation_v1"
  and .trajectory_counts == {"train":0,"validation":2,"test":0}
  and .max_abs_omega == 0.75' "${data}/manifest.json" >/dev/null

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 20 * 1024 * 1024 )) || {
  echo 'Refusing validation with less than 20 GiB MemAvailable' >&2
  exit 75
}
actual_image_id="$(docker image inspect --format '{{.Id}}' "${image_id}")"
[[ "${actual_image_id}" == "${image_id}" ]] || {
  echo "Pinned PhysicsNeMo image ID mismatch: ${actual_image_id}" >&2
  exit 75
}

if pgrep -af '[t]rain_tandem_fno.py|[t]rain_tandem_fno_rollout.py' >/dev/null; then
  echo 'Refusing validation while a PhysicsNeMo FNO training process is active' >&2
  exit 75
fi

output_dir="${result_root}/${label}"
[[ ! -e "${output_dir}" ]] || { echo "Refusing existing output: ${output_dir}" >&2; exit 1; }
mkdir -p "${output_dir}"
normalization_sha256="$(sha256sum "${normalization_data}/normalization.json" | awk '{print $1}')"
checkpoint_sha256="$(find "${model_dir}/best" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
jq -n \
  --arg label "${label}" \
  --arg image_id "${image_id}" \
  --arg model_dir "${model_dir}" \
  --arg checkpoint_tree_sha256 "${checkpoint_sha256}" \
  --arg normalization_data "${normalization_data}" \
  --arg normalization_sha256 "${normalization_sha256}" \
  --arg split validation \
  '{label:$label,image_id:$image_id,model_dir:$model_dir,
    checkpoint_tree_sha256:$checkpoint_tree_sha256,
    normalization_data:$normalization_data,
    normalization_sha256:$normalization_sha256,split:$split,
    frozen_test_status:"NOT_ACCESSED",model_selection_use:"FORBIDDEN"}' \
  > "${output_dir}/evaluation_provenance.json"

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
  --shm-size 2g --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
  --env PYTHONPATH=/workspace/src --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
  "${image_id}" \
  python -u scripts/spark_gpu_guard.py \
    --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py \
    --data "${data}" --normalization-data "${normalization_data}" \
    --config conf/tandem_fno_total_drag.yaml \
    --checkpoint-dir "/workspace/${model_dir}/best" \
    --output "/workspace/${output_dir}/evaluation.json" \
    --visualization-dir "/workspace/${output_dir}/figures" \
    --split validation --horizons 1 10 50 100 --segment-stride 25 \
    --action-mode observed \
    --segment-metrics-output "/workspace/${output_dir}/segments.json" \
    --evaluation-batch-size 4 --visualizations-per-horizon 0
