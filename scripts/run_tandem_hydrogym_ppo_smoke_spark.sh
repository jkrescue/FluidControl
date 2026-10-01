#!/usr/bin/env bash
# CPU-only software smoke: official HydroGym FlowEnv + SB3 PPO + PhysicsNeMo FNO.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
output="${OUTPUT_DIR:-artifacts/hydrogym/tandem_ppo_smoke_5epoch_spark}"
log="${output}.log"
[[ "${output}" == artifacts/hydrogym/* && "${output}" != *..* ]] || { echo "Output must stay under artifacts/hydrogym" >&2; exit 2; }
[[ ! -e "${output}" ]] || { echo "Output already exists: ${output}" >&2; exit 2; }
[[ "$(git -C .tools/hydrogym rev-parse HEAD)" == "4ab9854dea3d84e38a59c25e0f5835a00cf8225f" ]] || exit 1
docker image inspect "${image}" >/dev/null
free_gib="$(df -BG --output=avail "${root}" | tail -n 1 | tr -dc '0-9')"
(( free_gib >= 150 )) || { echo "Insufficient disk free: ${free_gib} GiB" >&2; exit 75; }
mkdir -p artifacts/hydrogym

docker run --rm --network none --cpus 4 --memory 16g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/train_tandem_hydrogym_ppo_smoke.py \
        --data data/curated/tandem_cylinders_expanded_independent_v2 \
        --config conf/tandem_fno_expanded.yaml \
        --checkpoint-dir artifacts/tandem_fno_expanded_spark_5epoch/best \
        --output "${output}" --timesteps 32 --episode-steps 16 \
    2>&1 | tee "${log}"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="TANDEM_HYDROGYM_PPO_SMOKE_OK"; assert d["scientific_status"]=="surrogate_only_software_smoke_not_cfd_control"; print("TANDEM_HYDROGYM_PPO_AUDIT_OK")' "${output}/audit.json"
