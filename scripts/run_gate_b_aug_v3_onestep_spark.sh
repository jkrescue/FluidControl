#!/usr/bin/env bash
# Train a seven-output PhysicsNeMo FNO from scratch on the audited 26-case split.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
data_root="data/curated/tandem_cylinders_gate_b_aug_v3"
output="${OUTPUT_DIR:-artifacts/tandem_fno_gate_b_aug_v3_30epoch}"
audit="artifacts/tandem_cylinders/gate_b_aug_v3_split_integrity.json"
epochs="${EPOCHS:-30}"
batch_size="${BATCH_SIZE:-64}"
seed="${SEED:-20261002}"

[[ "${epochs}" =~ ^[1-9][0-9]*$ && "${batch_size}" =~ ^[1-9][0-9]*$ \
   && "${seed}" =~ ^[1-9][0-9]*$ ]] || {
    echo "EPOCHS, BATCH_SIZE and SEED must be positive integers" >&2; exit 2;
}
[[ "${output}" != /* && "${output}" == artifacts/* ]] || {
    echo "OUTPUT_DIR must be project-relative under artifacts/" >&2; exit 2;
}
[[ -s "${data_root}/manifest.json" && -s "${audit}" ]] || {
    echo "Missing complete audited v3 Curator dataset" >&2; exit 1;
}
python3 - "${audit}" "${data_root}/manifest.json" <<'PY'
import json,sys
audit=json.load(open(sys.argv[1],encoding="utf-8"))
manifest=json.load(open(sys.argv[2],encoding="utf-8"))
expected={"train":26,"validation":4,"test":5}
if audit.get("status")!="SPLIT_INTEGRITY_OK" or audit.get("split_counts")!=expected:
    raise SystemExit("Invalid split-integrity audit")
if manifest.get("trajectory_counts")!=expected or manifest.get("profile")!="gate_b_aug_v3":
    raise SystemExit("Invalid Curator v3 manifest")
PY
[[ ! -e "${output}/training_history.json" ]] || {
    echo "Refusing to overwrite completed one-step FNO training" >&2; exit 1;
}
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 150 )) || {
    echo "Only ${free_gib} GiB disk free; 150 GiB required" >&2; exit 75;
}
docker image inspect fluid-control-physicsnemo:2.2.2 >/dev/null
mkdir -p "${output}"

exec docker run --rm --network none --gpus 'device=0' --cpus 8 --memory 64g \
    --shm-size 2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace fluid-control-physicsnemo:2.2.2 \
    python -u scripts/spark_gpu_guard.py \
      --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 -- \
      python -u scripts/train_tandem_fno.py \
        --config-name tandem_fno_total_drag \
        "data.root=${data_root}" "output_dir=${output}" \
        "training.epochs=${epochs}" "training.batch_size=${batch_size}" \
        "training.seed=${seed}" \
        "training.gpu_memory_fraction=0.20" "data.num_streams=1"
