#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/fluid_control
source_root="$repo/artifacts/fcp012_gradient_source_67cdc65_immutable"
output="$repo/artifacts/fcp012_decoder_gradient_diagnostic_20261005"
approval="$repo/docs/FC_P012_EXECUTION_APPROVAL_20261005.json"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
p009="$repo/artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
decoder="$repo/artifacts/fcp011_decoder_tail_training_worker_20261005/final"
image=fluid-control-physicsnemo:2.2.2
container=fcp012-decoder-gradient-diagnostic-20261005

check() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]]; }
check "$source_root/scripts/diagnose_fcp012_decoder_gradient_alignment.py" 978d28da0fd42d087a22c3a5620ee15c32c1edc63bf65ef07e5602803ff924c0
check "$source_root/tests/test_diagnose_fcp012_decoder_gradient_alignment.py" 1f571caaf5838aeffc4c2fbd535997ad360eb7885404de7b2740707e163cdb81
check "$source_root/scripts/train_fcp011_decoder_scope.py" 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5
check "$approval" 4bbf7165f0ef23af3b50c119da0d8317f512ec033ebbef8f34adc70e06516dbc
check "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
check "$p009/FNO.0.0.mdlus" dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31
check "$p009/checkpoint.0.0.pt" 4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e
check "$decoder/FNO.0.1.mdlus" 5102e83e00276994ad85b88c59721d52915400a037fc29f321d593fa69b800d8
check "$decoder/checkpoint.0.1.pt" c3c8c92ea792ed7332da75af269143b10ea0d0541b84273b7424e2922d9ae6b0
[[ ! -e "$output" ]]
mkdir "$output"
cp "$approval" "$output/execution_approval.json"
cp "$0" "$output/immutable_launcher.sh"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

python3 - "$output/launch_receipt.json" <<'PY'
import json, pathlib
p=pathlib.Path(__import__('sys').argv[1])
p.write_text(json.dumps({
 "status":"FC_P012_GRADIENT_DIAGNOSTIC_LAUNCH_VERIFIED",
 "source_commit":"67cdc65645e5742bbf469d22ebab558a135c537a",
 "diagnostic_sha256":"978d28da0fd42d087a22c3a5620ee15c32c1edc63bf65ef07e5602803ff924c0",
 "test_sha256":"1f571caaf5838aeffc4c2fbd535997ad360eb7885404de7b2740707e163cdb81",
 "trainer_sha256":"9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5",
 "execution_approval_sha256":"4bbf7165f0ef23af3b50c119da0d8317f512ec033ebbef8f34adc70e06516dbc",
 "image_id":"sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e",
 "allocator_fraction":0.45,"min_mem_available_gib":20,"timeout_seconds":900,
 "numerical_residual_classification":"NOT_CLASSIFIED_PENDING_INDEPENDENT_REVIEW",
 "validation_or_frozen_mounted":False,"optimizer_or_checkpoint_authorized":False
},indent=2)+"\n")
PY

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
  -v "$p009:/workspace/p009:ro" -v "$decoder:/workspace/decoder:ro" \
  -v "$config:/workspace/config.yaml:ro" -v "$output:/workspace/output:rw" \
  -w /workspace/project "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .45 \
    --margin-gib 4 --poll-seconds 2 -- timeout 900 \
    python -u scripts/diagnose_fcp012_decoder_gradient_alignment.py \
      --config /workspace/config.yaml --p009-parent /workspace/p009 \
      --decoder-tail /workspace/decoder \
      --trainer /workspace/project/scripts/train_fcp011_decoder_scope.py \
      --output /workspace/output/diagnostic |& tee "$output/run.log"

python3 - "$output" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); result=r/'diagnostic/result.json'
d=json.loads(result.read_text())
if d.get('status')!='FC_P012_TRAIN_ONLY_DECODER_GRADIENT_DIAGNOSTIC_COMPLETE' or len(d.get('rows',[]))!=12:
 raise SystemExit('incomplete FC-P012 result')
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
(r/'completion_receipt.json').write_text(json.dumps({
 'status':'FC_P012_GRADIENT_DIAGNOSTIC_EXECUTION_COMPLETE_NOT_ADMISSION',
 'result_sha256':sha(result),'launch_receipt_sha256':sha(r/'launch_receipt.json'),
 'execution_approval_sha256':'4bbf7165f0ef23af3b50c119da0d8317f512ec033ebbef8f34adc70e06516dbc',
 'row_count':12,'numerical_residual_classification':'NOT_CLASSIFIED_PENDING_INDEPENDENT_REVIEW',
 'optimizer_steps':0,'candidate_saved':False,'validation_accessed':False,
 'frozen_test_accessed':False,'ppo_executed':False
},indent=2)+'\n')
PY
