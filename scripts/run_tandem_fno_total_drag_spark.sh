#!/usr/bin/env bash
# Train the seven-output PhysicsNeMo FNO on GPU0 in an isolated container.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo:2.2.2"
data="data/curated/tandem_cylinders_expanded_independent_v2"
output="${OUTPUT_DIR:-artifacts/tandem_fno_total_drag_spark}"
batch_size="${BATCH_SIZE:-4}"
smoke="${SMOKE:-false}"
fraction="0.20"

[[ "${output}" == /* ]] || output="${root}/${output}"
[[ "${output}" == "${root}/"* ]] || { echo "Output must be inside project" >&2; exit 1; }
[[ -s "${data}/manifest.json" && -s "${data}/normalization.json" ]] || {
    echo "Curated independent-v2 data are not ready" >&2; exit 1;
}
python3 - "${data}/normalization.json" <<'PY'
import json
import sys
from pathlib import Path

stats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = ("all_force_channels", "all_force_mean", "all_force_std")
missing = [key for key in required if key not in stats]
if missing:
    raise SystemExit(f"normalization is missing {missing}; run add_all_force_normalization.py")
if stats["all_force_channels"] != ["front_cd", "front_cl", "rear_cd", "rear_cl"]:
    raise SystemExit("unexpected all-force channel order")
PY
docker image inspect "${image}" >/dev/null

if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    [[ "${ALLOW_RESUME:-false}" == "true" ]] || {
        echo "Refusing non-empty output without ALLOW_RESUME=true: ${output}" >&2
        exit 1
    }
fi
mkdir -p "${output}"

overrides=(
    "data.root=/workspace/${data}"
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
      python -u scripts/train_tandem_fno.py \
      --config-name tandem_fno_total_drag "${overrides[@]}" "$@"
