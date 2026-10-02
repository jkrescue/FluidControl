#!/usr/bin/env bash
# Run the fail-closed Stage-C CEM screen after the multistep Gate-B audit passes.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="${PHYSICSNEMO_IMAGE:-fluid-control-physicsnemo:2.2.2}"
config="${CONFIG:-conf/tandem_cem_stage_c.yaml}"

[[ "${config}" != /* && -f "${config}" ]] || {
    echo "CONFIG must be an existing project-relative file" >&2
    exit 2
}
docker image inspect "${image}" >/dev/null

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/screen_tandem_cem_mpc.py --config "${config}"
