#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/fluid_control
source_root="$repo/artifacts/fcp013_probe_source_218eada_immutable"
output="$repo/artifacts/fcp013_independent_force_fno_resource_probe_v2_20261005"
approval="$repo/docs/FC_P013_RESOURCE_PROBE_RETRY_APPROVAL_20261005.json"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
parent="$repo/artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
container=fcp013-independent-force-resource-probe-v2-20261005
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e

check() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]]; }
check "$source_root/scripts/train_fcp013_independent_force_fno.py" 0e004a1520533fae85eb0fac455189e37b711ff0bfa0114b497dcf414e1ada00
check "$source_root/tests/test_train_fcp013_independent_force_fno.py" 2a4ecb75759d6cf47bc9921f4c33a023a168c2ae6cce958bc6918ac67f6e3899
check "$source_root/scripts/train_fcp011_decoder_scope.py" 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5
check "$approval" d461bd7a2f51de2e7de3514cd155d9cffb4bea9ab6ba2bf5035bd55c6e1cd38c
check "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
check "$parent/FNO.0.0.mdlus" dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31
check "$parent/checkpoint.0.0.pt" 4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e
[[ ! -e "$output" ]]
mkdir "$output"
cp "$approval" "$output/execution_approval.json"
cp "$0" "$output/immutable_launcher.sh"
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
    --margin-gib 4 --poll-seconds 2 -- timeout 300 \
    python -u scripts/train_fcp013_independent_force_fno.py \
      --config /workspace/config.yaml --parent /workspace/parent \
      --frozen-trainer /workspace/project/scripts/train_fcp011_decoder_scope.py \
      --output /workspace/output/probe --resource-probe |& tee "$output/run.log"
