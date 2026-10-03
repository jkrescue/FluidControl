#!/usr/bin/env bash
set -euo pipefail

root=/workspace/fluid_control
image_id=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
run_id="${1:-tandem_fno_h20_resource_20261003}"
warmup="${2:-1}"
iterations="${3:-2}"
[[ "${run_id}" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "${warmup}" =~ ^[1-9][0-9]*$ && "${iterations}" =~ ^[1-9][0-9]*$ ]]
output="artifacts/benchmarks/${run_id}"
cd "${root}"
[[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; exit 1; }
[[ "$(find data/curated/tandem_cylinders_control_gap_v4/train -name '*.h5' -type f | wc -l)" -eq 28 ]]
[[ "$(docker image inspect --format '{{.Id}}' "${image_id}")" == "${image_id}" ]]
if pgrep -af '[t]rain_tandem_fno.py|[t]rain_tandem_fno_rollout.py' >/dev/null; then
  echo 'Refusing benchmark while FNO training is active' >&2
  exit 75
fi
mkdir -p "${output}"
exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 100g \
  --shm-size 2g --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
  --env "BENCHMARK_IMAGE_ID=${image_id}" \
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
  "${image_id}" python -u scripts/spark_gpu_guard.py \
    --min-free-gib 20 --allocator-fraction 0.65 --margin-gib 4 -- \
  python -u scripts/benchmark_tandem_fno_h20_resource.py \
    --data data/curated/tandem_cylinders_control_gap_v4 \
    --output-json "${output}/benchmark.json" \
    --output-csv "${output}/benchmark.csv" \
    --warmup "${warmup}" --iterations "${iterations}"
