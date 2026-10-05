#!/usr/bin/env bash
# Bounded train-only diagnostic. Default mode is read-only CPU preflight.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
mode="${1:---dry-run}"
case "$mode" in
  --dry-run|--execute) ;;
  *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2 ;;
esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="578368a3f75edcfc0808c3d8bc79ba972ecba4fb"
diagnostic_sha="d9bcc3e77eeb4da432a2341c7b2aab5413049dd813434d400f40a380a43a9426"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
d015_dependency_sha="530bdf99a13454e7ee050d73ad55bd3c79a9417ece19c8f77496fe424c76f405"

train8="$repo/data/curated/tandem_cylinders_dynamic_train8_v1"
base="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
pair_manifest="$repo/artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
candidate="$repo/artifacts/fcp003c_fixed_feature_force_readout_v2_20261005"
parent="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best"
config="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
container="fcp003c-fixed-feature-force-readout-v2-20261005"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() { [[ -f "$1" && "$(sha "$1")" == "$2" ]]; }

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ "$(git rev-parse "$source_commit^{commit}")" == "$source_commit" ]]
[[ "$(git show "$source_commit:scripts/diagnose_fcp003c_fixed_feature_force_readout.py" | sha256sum | awk '{print $1}')" == "$diagnostic_sha" ]]
[[ "$(git show "$source_commit:scripts/spark_gpu_guard.py" | sha256sum | awk '{print $1}')" == "$guard_sha" ]]
[[ "$(git show "$source_commit:scripts/diagnose_fcp003c_train_vs_validation_true_state_h1.py" | sha256sum | awk '{print $1}')" == "$d015_dependency_sha" ]]
require_sha "$train8/manifest.json" a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35
require_sha "$base/manifest.json" 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2
require_sha "$base/normalization.json" f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1
require_sha "$pair_manifest" b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c
require_sha "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
require_sha "$parent/FNO.0.2.mdlus" f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4
require_sha "$parent/checkpoint.0.2.pt" a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a
[[ ! -e "$candidate" ]] || { echo "exclusive output already exists: $candidate" >&2; exit 2; }

if [[ "$mode" == --dry-run ]]; then
  echo "FCP003C_FIXED_FEATURE_FORCE_READOUT_V2_DRY_RUN_READY_NO_GPU"
  exit 0
fi
[[ "${FCP003C_FIXED_FEATURE_READOUT_V2_TOKEN:-}" == "EXECUTE_REVIEWED_FCP003C_FIXED_FEATURE_READOUT_V2" ]]

scratch="$(mktemp -d)"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  find "$scratch" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT INT TERM
mkdir -p "$scratch/source"
git archive "$source_commit" src scripts | tar -x -C "$scratch/source"
mkdir "$candidate"
cp "$0" "$candidate/immutable_launcher.sh"
printf '%s  %s\n' "$diagnostic_sha" scripts/diagnose_fcp003c_fixed_feature_force_readout.py >"$candidate/source.sha256"

set +e
timeout --signal=TERM --kill-after=30s 10m docker run --rm --name "$container" \
  --gpus device=0 --network none --cpus 8 --memory 90g --shm-size 2g \
  --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
  -e PYTHONPATH=/workspace/src:/workspace/scripts \
  -v "$scratch/source/src:/workspace/src:ro" \
  -v "$scratch/source/scripts:/workspace/scripts:ro" \
  -v "$train8/train:/workspace/train8/train:ro" \
  -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" \
  -v "$base/train:/workspace/base/train:ro" \
  -v "$base/manifest.json:/workspace/base/manifest.json:ro" \
  -v "$base/normalization.json:/workspace/base/normalization.json:ro" \
  -v "$pair_manifest:/workspace/evidence/dynamic_pair_manifest.json:ro" \
  -v "$config:/workspace/config/resolved_config.yaml:ro" \
  -v "$parent:/workspace/checkpoint:ro" \
  -v "$candidate:/workspace/artifact:rw" \
  -w /workspace "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 2 --poll-seconds 2 -- \
  python -u scripts/diagnose_fcp003c_fixed_feature_force_readout.py \
    --train8 /workspace/train8 --base /workspace/base \
    --pair-manifest /workspace/evidence/dynamic_pair_manifest.json \
    --normalization /workspace/base/normalization.json \
    --config /workspace/config/resolved_config.yaml \
    --checkpoint-dir /workspace/checkpoint \
    --output-dir /workspace/artifact/result_bundle --chunk-size 10 \
  2>&1 | tee "$candidate/run.log"
status=${PIPESTATUS[0]}
set -e
[[ $status -eq 0 ]] || exit "$status"
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]

python3 - "$candidate" "$source_commit" "$diagnostic_sha" "$image_id" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
root=pathlib.Path(sys.argv[1]); source_commit,diagnostic_sha,image_id=sys.argv[2:]
result_path=root/'result_bundle/result.json'; cache_path=root/'result_bundle/fixed_features.npz'
result=json.loads(result_path.read_text())
assert result['status']=='FCP003C_FIXED_FEATURE_FORCE_READOUT_DIAGNOSTIC_COMPLETE'
assert result['implementation_sha256']==diagnostic_sha
assert result['optimizer_steps']==0 and result['candidate_saved'] is False
assert result['validation_accessed'] is result['frozen_test_accessed'] is result['ppo_executed'] is False
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
assert result['cache']['sha256']==sha(cache_path)
payload={'status':'FCP003C_FIXED_FEATURE_FORCE_READOUT_V2_EXECUTION_COMPLETE_NOT_ADMISSION','source_commit':source_commit,'diagnostic_sha256':diagnostic_sha,'image_id':image_id,'result_sha256':sha(result_path),'cache_sha256':sha(cache_path),'optimizer_steps':0,'candidate_saved':False,'validation_accessed':False,'frozen_test_accessed':False,'ppo_executed':False}
target=root/'completion_receipt.json'
with tempfile.NamedTemporaryFile('w',dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.link(tmp,target); tmp.unlink()
PY
echo "FCP003C_FIXED_FEATURE_FORCE_READOUT_V2_EXECUTION_COMPLETE_NOT_ADMISSION"
