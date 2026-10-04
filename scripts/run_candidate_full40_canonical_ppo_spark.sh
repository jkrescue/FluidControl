#!/usr/bin/env bash
# Pinned container entry for the candidate-aware canonical PPO wrapper.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$mode" in
  --dry-run|--execute) ;;
  *) echo "first argument must be --dry-run or --execute" >&2; exit 2 ;;
esac
if [[ "$mode" == "--dry-run" ]]; then
  for argument in "$@"; do
    [[ "$argument" != "--execute" ]] || {
      echo "--execute cannot be smuggled through the dry-run entry" >&2; exit 2;
    }
  done
fi

image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
runtime_id="sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$runtime_id" ]] || {
  echo "pinned PhysicsNeMo+HydroGym image ID mismatch" >&2; exit 2;
}

common=(--rm --network none --pids-limit 1024 --cap-drop ALL
  --security-opt no-new-privileges --user "$(id -u):$(id -g)"
  --env USER="$(id -un)" --env LOGNAME="$(id -un)" --env HOME=/tmp
  --env CANDIDATE_PPO_RUNTIME_IMAGE_ID="$runtime_id"
  --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace)
command=(python -u scripts/run_candidate_full40_canonical_ppo.py "$@")

if [[ "$mode" == "--dry-run" ]]; then
  docker run "${common[@]}" --cpus 2 --memory 4g \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    "$image" "${command[@]}"
else
  command+=(--execute)
  python3 scripts/spark_ppo_gpu_guard.py --min-free-gib 20 \
    --allocator-fraction 0.20 --margin-gib 4 --poll-seconds 2 -- \
    docker run "${common[@]}" --gpus device=0 --cpus 8 --memory 64g \
    --shm-size 2g --env CANDIDATE_PPO_GPU_GUARD_ACTIVE=1 \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    "$image" "${command[@]}"
fi
