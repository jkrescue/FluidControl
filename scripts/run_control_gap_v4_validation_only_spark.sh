#!/usr/bin/env bash
# Formal v4/v3 validation-only comparison. This script never opens the frozen split.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"

image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
v4_data="data/curated/tandem_cylinders_control_gap_v4"
v3_data="data/curated/tandem_cylinders_gate_b_aug_v3"
candidate="artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch"
parent="artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch"
output="artifacts/tandem_cylinders/control_gap_v4_validation_only_20261003"
lock="${XDG_RUNTIME_DIR:-/tmp}/fluid-control-v4-validation-only.lock"

preflight="${output}/preflight.json"
candidate_report="${output}/v4_candidate_validation.json"
parent_report="${output}/v3_parent_on_v4_validation.json"
decision="${output}/decision.json"
running_marker="${output}/VALIDATION_RUNNING"
failed_marker="${output}/VALIDATION_FAILED"
complete_marker="${output}/VALIDATION_COMPLETE"

assert_no_training() {
    if pgrep -f '[t]rain_tandem_fno_rollout.py' >/dev/null; then
        echo "Refusing validation while an FNO rollout training process is active" >&2
        exit 75
    fi
}

on_error() {
    code=$?
    trap - ERR
    rm -f "${running_marker}"
    touch "${failed_marker}"
    echo "V4_VALIDATION_ONLY_FAILED exit=${code} at $(date -Is)" >&2
    exit "${code}"
}

exec 9>"${lock}"
flock -n 9 || { echo "Another v4 validation-only evaluation holds ${lock}" >&2; exit 75; }

[[ "$(uname -m)" == aarch64 ]] || { echo "ARM64 DGX Spark required" >&2; exit 2; }
[[ -s "${v4_data}/manifest.json" && -s "${v4_data}/normalization.json" ]] || {
    echo "Finalized v4 manifest or normalization is missing" >&2; exit 2;
}
[[ -s "${candidate}/training_history.json" && -d "${candidate}/best" ]] || {
    echo "Completed v4 candidate is missing" >&2; exit 75;
}
[[ -s "${parent}/resolved_config.yaml" && -d "${parent}/best" ]] || {
    echo "Immutable v3 H20 parent is missing" >&2; exit 2;
}
[[ ! -e "${complete_marker}" && ! -e "${candidate_report}" \
    && ! -e "${parent_report}" && ! -e "${decision}" ]] || {
    echo "Refusing to overwrite an existing formal validation result" >&2; exit 2;
}
actual_image="$(docker image inspect "${image}" --format '{{.Id}}')"
[[ "${actual_image}" == "${expected_image}" ]] || {
    echo "PhysicsNeMo image ID mismatch: ${actual_image}" >&2; exit 2;
}
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge $((20 * 1024 * 1024)) ]] || {
    echo "Host MemAvailable is below the 20 GiB safety floor" >&2; exit 75;
}
assert_no_training

mkdir -p "${output}"
trap on_error ERR
rm -f "${failed_marker}"
touch "${running_marker}"

python3 scripts/audit_control_gap_v4_validation.py preflight \
    --v4-data "${v4_data}" --v3-data "${v3_data}" \
    --candidate "${candidate}" --parent "${parent}" \
    --expected-epochs 5 --output "${preflight}"

run_evaluation() {
    checkpoint=$1
    report=$2
    normalization=$3
    assert_no_training
    docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
        --shm-size 2g --user "$(id -u):$(id -g)" \
        --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
        --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
        --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
        "${image}" \
        python -u scripts/spark_gpu_guard.py \
          --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
          python -u scripts/evaluate_tandem_fno.py \
            --data "${v4_data}" --normalization-data "${normalization}" \
            --config conf/tandem_fno_total_drag.yaml \
            --checkpoint-dir "/workspace/${checkpoint}/best" \
            --output "/workspace/${report}" \
            --split validation --horizons 1 10 50 100 \
            --segment-stride 25 --action-mode observed \
            --evaluation-batch-size 4 --visualizations-per-horizon 0
}

run_evaluation "${candidate}" "${candidate_report}" "${v4_data}"
run_evaluation "${parent}" "${parent_report}" "${v3_data}"

python3 scripts/audit_control_gap_v4_validation.py compare \
    --candidate-report "${candidate_report}" --parent-report "${parent_report}" \
    --preflight-report "${preflight}" --expected-data "${v4_data}" \
    --candidate-normalization "${v4_data}" --parent-normalization "${v3_data}" \
    --output "${decision}"

rm -f "${running_marker}"
touch "${complete_marker}"
trap - ERR
echo "V4_VALIDATION_ONLY_COMPLETE ${decision} $(date -Is)"
