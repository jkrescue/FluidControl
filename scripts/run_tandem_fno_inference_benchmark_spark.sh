#!/usr/bin/env bash
# Guarded, reproducible batch-one FNO forward timing on a real CFD segment.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
model="artifacts/distributed_runs/gateb_multistep_20261002/formal/tandem_fno_total_drag_rollout_seed20261003"
case_path="data/curated/tandem_cylinders_expanded_independent_v2/test/expanded_test_00.h5"
data_root="data/curated/tandem_cylinders_expanded_independent_v2"
output="artifacts/monitor/fno_inference_benchmark_seed20261003.json"

[[ -d "${model}/best" && -s "${case_path}" && -s "${data_root}/normalization.json" ]] || {
    echo "Missing validated FNO checkpoint or real CFD benchmark input" >&2; exit 1;
}
[[ ! -e "${output}" ]] || {
    echo "Refusing to overwrite completed inference benchmark" >&2; exit 1;
}
docker image inspect fluid-control-physicsnemo:2.2.2 >/dev/null

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONPATH=/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace fluid-control-physicsnemo:2.2.2 \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/benchmark_tandem_fno_inference.py \
        --case "${case_path}" --normalization-data "${data_root}" \
        --checkpoint-dir "${model}/best" --output "${output}" --steps 100
