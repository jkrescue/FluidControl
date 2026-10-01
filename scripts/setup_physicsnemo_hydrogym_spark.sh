#!/usr/bin/env bash
# Build the isolated PhysicsNeMo + HydroGym-core image for the tandem adapter.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_dir="${root}/.tools/hydrogym"
commit="4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
cd "${root}"

[[ -d "${source_dir}/.git" ]] || {
    echo "Missing pinned HydroGym source; run scripts/setup_hydrogym_firedrake.sh first" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${commit}" ]] || {
    echo "HydroGym source commit differs from pinned revision" >&2
    exit 1
}
[[ -z "$(git -C "${source_dir}" status --porcelain)" ]] || {
    echo "HydroGym source worktree is dirty" >&2
    exit 1
}
docker image inspect fluid-control-physicsnemo:2.2.2 >/dev/null
docker build --progress=plain \
    --file docker/physicsnemo/Dockerfile.hydrogym \
    --tag "${image}" docker/physicsnemo
docker run --rm --network none --cpus 2 --memory 6g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src \
    --mount "type=bind,src=${root},dst=/workspace,readonly" \
    --workdir /workspace "${image}" python -c '
import hydrogym
import gymnasium
import stable_baselines3
import physicsnemo
import torch
assert hydrogym.FlowEnv.__name__ == "FlowEnv"
assert gymnasium.__version__ == "1.2.3"
assert stable_baselines3.__version__ == "2.7.1"
assert physicsnemo.__version__ == "2.2.2"
print("HYDROGYM_PHYSICSNEMO_STACK_OK", torch.__version__)
'
