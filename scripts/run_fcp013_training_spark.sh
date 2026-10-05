#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/fluid_control
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
output="$repo/artifacts/fcp013_independent_force_fno_training_20261005"
approval="$repo/docs/FC_P013_TRAINING_EXECUTION_APPROVAL_20261005.json"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
parent="$repo/artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
container=fcp013-independent-force-training-20261005
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
check() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]]; }
check "$source_root/scripts/train_fcp013_independent_force_fno.py" f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7
check "$source_root/tests/test_train_fcp013_independent_force_fno.py" 8efcb89705387539a4478bb04f884213ffabd57c3190c144c0625e106e293707
check "$source_root/src/fluid_control/dual_fno.py" 23380355b812d94025071f62feb9dbf51e593244debff6f2b38b4215c644c254
check "$source_root/scripts/train_fcp011_decoder_scope.py" 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5
check "$approval" 1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120
check "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
check "$parent/FNO.0.0.mdlus" dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31
check "$parent/checkpoint.0.0.pt" 4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e
[[ ! -e "$output" ]]
mkdir "$output"
cp "$approval" "$output/execution_approval.json"
cp "$0" "$output/immutable_launcher.sh"
git -C "$source_root" rev-parse HEAD 2>/dev/null || true
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker run --rm --name "$container" --stop-timeout 30 --user 1000:1000 \
  --gpus device=0 --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 2048 --memory 90g --shm-size 8g \
  --tmpfs /tmp:rw,size=8g --tmpfs /home/USER:rw,size=1g \
  -e HOME=/home/USER -e XDG_CACHE_HOME=/tmp/xdg \
  -e PYTHONPATH=/workspace/project/src:/workspace/project/scripts \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$source_root:/workspace/project:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/base:/workspace/base:ro" \
  -v "$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train:/workspace/base/train:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train8:/workspace/train8:ro" \
  -v "$repo/data/curated/tandem_cylinders_dynamic_train8_v1/train:/workspace/train8/train:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train16:/workspace/train16:ro" \
  -v "$repo/data/curated/tandem_cylinders_directppo_train16_v1/train:/workspace/train16/train:ro" \
  -v "$parent:/workspace/parent:ro" -v "$config:/workspace/config.yaml:ro" \
  -v "$output:/workspace/output:rw" -w /workspace/project "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .45 \
    --margin-gib 4 --poll-seconds 2 -- timeout 14400 \
    python -u scripts/train_fcp013_independent_force_fno.py \
      --config /workspace/config.yaml --parent /workspace/parent \
      --frozen-trainer /workspace/project/scripts/train_fcp011_decoder_scope.py \
      --output /workspace/output/candidate |& tee "$output/run.log"
