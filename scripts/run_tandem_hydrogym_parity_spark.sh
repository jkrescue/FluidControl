#!/usr/bin/env bash
# Real-HDF5 one-step parity and OpenFOAM probe checks; no GPU is required.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${root}"
image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
data="data/curated/tandem_cylinders_expanded_independent_v2"
source_commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
[[ "$(git -C .tools/hydrogym rev-parse HEAD)" == "${source_commit}" ]] || exit 1
docker image inspect "${image}" >/dev/null
[[ -s artifacts/tandem_fno_expanded_spark_5epoch/best/FNO.0.5.mdlus ]] || exit 1
mkdir -p artifacts/tandem_cylinders

docker run --rm --network none --cpus 4 --memory 16g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    --env OMP_NUM_THREADS=2 \
    --mount "type=bind,src=${root},dst=/workspace" \
    --workdir /workspace "${image}" \
    python -u scripts/validate_tandem_hydrogym_adapter.py \
        --data "${data}" --config conf/tandem_fno_expanded.yaml \
        --checkpoint-dir artifacts/tandem_fno_expanded_spark_5epoch/best \
        --output artifacts/tandem_cylinders/expanded_hydrogym_adapter_parity.json

python3 scripts/validate_tandem_probe_mapping.py \
    --data "${data}" --cases-root cfd/tandem_cylinders/cases \
    --split validation --case expanded_validation_00 \
    --output artifacts/tandem_cylinders/expanded_validation_00_probe_mapping.json
python3 scripts/validate_tandem_probe_mapping.py \
    --data "${data}" --cases-root cfd/tandem_cylinders/cases \
    --split test --case expanded_test_04 \
    --output artifacts/tandem_cylinders/expanded_test_04_probe_mapping.json
echo TANDEM_HYDROGYM_ADAPTER_SMOKE_OK
