#!/usr/bin/env bash
# Validation-only v4 window-mean Cd audit. Never opens the frozen test split.
set -eEuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"

image="fluid-control-physicsnemo:2.2.2"
expected_image="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data="data/curated/tandem_cylinders_control_gap_v4"
candidate="artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch"
checkpoint="${candidate}/best"
output="artifacts/tandem_cylinders/control_gap_v4_window_mean_cd_20261003.json"
lock="${XDG_RUNTIME_DIR:-/tmp}/fluid-control-window-mean-cd-v4.lock"

assert_no_training() {
    if pgrep -f '[t]rain_tandem_fno(_rollout)?\.py' >/dev/null; then
        echo "Refusing evaluation while FNO training is active" >&2
        exit 75
    fi
}

exec 9>"${lock}"
flock -n 9 || { echo "Another window-mean evaluation holds ${lock}" >&2; exit 75; }

[[ "$(uname -m)" == aarch64 ]] || { echo "ARM64 DGX Spark required" >&2; exit 2; }
[[ -s "${data}/manifest.json" && -s "${data}/normalization.json" ]] || {
    echo "Finalized v4 data contract is missing" >&2; exit 2;
}
[[ -s "${candidate}/training_history.json" && -d "${checkpoint}" ]] || {
    echo "Completed v4 candidate is missing" >&2; exit 75;
}
[[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 2; }

python3 - "${candidate}/training_history.json" <<'PY'
import json
import sys
from pathlib import Path

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(history, list) or [row.get("epoch") for row in history] != list(range(1, 6)):
    raise SystemExit("v4 candidate has not completed exactly epochs 1..5")
PY

models=("${checkpoint}"/FNO.*.mdlus)
[[ ${#models[@]} -eq 1 && -f "${models[0]}" ]] || {
    echo "Expected exactly one official PhysicsNeMo model checkpoint" >&2; exit 2;
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

checkpoint_sha="$(sha256sum "${models[0]}" | awk '{print $1}')"
normalization_sha="$(sha256sum "${data}/normalization.json" | awk '{print $1}')"
mkdir -p "$(dirname "${output}")"

docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env WINDOW_MEAN_HOST_TRAINING_GUARD=passed \
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" --workdir /workspace \
    "${image}" \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/evaluate_tandem_window_mean_cd.py \
        --data "${data}" --normalization-data "${data}" \
        --checkpoint-dir "${checkpoint}" \
        --expected-checkpoint-sha256 "${checkpoint_sha}" \
        --expected-normalization-sha256 "${normalization_sha}" \
        --output "${output}" --batch-size 4

echo "V4_WINDOW_MEAN_CD_COMPLETE ${output} $(date -Is)"
