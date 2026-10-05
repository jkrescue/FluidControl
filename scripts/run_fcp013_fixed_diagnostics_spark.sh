#!/usr/bin/env bash
# Project orchestration of the original read-only six-window comparison.
set -euo pipefail
repo=/workspace/fluid_control
mode=${1:---dry-run}
[[ "$mode" == --dry-run || "$mode" == --execute ]]
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
chain="$repo/artifacts/p013_posteval_chain_23a4ec020ed6_immutable"
candidate="$repo/artifacts/fcp013_independent_force_fno_training_r2_20261005"
output="$candidate/fixed_six_diagnostics"
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
container=fcp013-fixed-six-diagnostics-r2-20261005
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
parent="$repo/artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]]
[[ "$(sha "$chain/receipt.json")" == a426f82bf1877ed1a930e763a62e6f2739b91185356fdf19bd435735fa7507ef ]]
[[ "$(sha "$source_root/scripts/train_fcp011_decoder_scope.py")" == 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5 ]]
[[ "$(sha "$config")" == 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9 ]]
python3 - "$chain" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); receipt=json.loads((root/'receipt.json').read_text())
name='scripts/evaluate_fcp013_fixed_train_windows.py'
assert hashlib.sha256((root/name).read_bytes()).hexdigest()==receipt['sha256'][name]
PY
if [[ "$mode" == --dry-run ]]; then
  echo FIXED_SIX_SOURCE_VERIFIED_NO_GPU_NO_DIAGNOSTICS
  exit 0
fi
! systemctl --user is-active --quiet fluid-control-fcp013-training-r2-20261005.service
! docker inspect fcp013-independent-force-training-r2-20261005 >/dev/null 2>&1
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=50*1024*1024 && f>=30*1024*1024)}' /proc/meminfo
python3 - "$candidate" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((root/'completion_receipt.json').read_text())
assert receipt['status']=='FC_P013_TRAINING_COMPLETE_NOT_ADMISSION' and receipt['execution_attempt']==2
assert receipt['candidate_audit_sha256']==sha(root/'candidate_audit.json')
audit=json.loads((root/'candidate_audit.json').read_text())
assert audit['status']=='FC_P013_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION'
assert audit['dual_manifest_sha256']==receipt['dual_manifest_sha256']
for name,expected in audit['sha256'].items():
 p=root/name
 assert p.resolve().is_relative_to(root.resolve()) and sha(p)==expected
PY
[[ ! -e "$output" ]]
mkdir "$output"
cp "$0" "$output/immutable_launcher.sh"
manifest_sha=$(sha "$candidate/candidate/dual_model_manifest.json")
result_sha=$(sha "$candidate/candidate/result.json")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker run --rm --name "$container" --user 1000:1000 --gpus device=0 \
  --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 40g --pids-limit 2048 --shm-size 4g --tmpfs /tmp:rw,size=4g \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
  -e PYTHONPATH=/workspace/diagnostic:/workspace/project/src:/workspace/project/scripts \
  -v "$source_root:/workspace/project:ro" -v "$chain/scripts:/workspace/diagnostic:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/base:/workspace/base:ro" \
  -v "$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train:/workspace/base/train:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train8:/workspace/train8:ro" \
  -v "$repo/data/curated/tandem_cylinders_dynamic_train8_v1/train:/workspace/train8/train:ro" \
  -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train16:/workspace/train16:ro" \
  -v "$repo/data/curated/tandem_cylinders_directppo_train16_v1/train:/workspace/train16/train:ro" \
  -v "$parent:/workspace/parent:ro" -v "$config:/workspace/config.yaml:ro" \
  -v "$candidate/candidate:/workspace/dual:ro" -v "$output:/workspace/output:rw" \
  -w /workspace/project "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 --poll-seconds 2 -- \
  timeout 1200 python -u /workspace/diagnostic/evaluate_fcp013_fixed_train_windows.py \
  --config /workspace/config.yaml --parent /workspace/parent \
  --frozen-trainer /workspace/project/scripts/train_fcp011_decoder_scope.py \
  --dual-manifest /workspace/dual/dual_model_manifest.json --expected-manifest-sha256 "$manifest_sha" \
  --training-result /workspace/dual/result.json --expected-training-result-sha256 "$result_sha" \
  --output /workspace/output/result.json |& tee "$output/run.log"
