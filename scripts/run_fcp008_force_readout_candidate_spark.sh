#!/usr/bin/env bash
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
source_commit="5f14dd6f496eb1a0bfc420e5bb49bf85078436ff"
builder_sha="649864f9d3efae2cd8f5efcc6cf93be54b1f64fefa942f5f4ce2f038a31bf585"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
approval="$repo/docs/FC_P008_EXECUTION_APPROVAL_20261005.json"
approval_sha="ace85fc828c0aac7943970f25c76754152bed46d488957c3bd5a003312321c11"

base="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$repo/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$repo/data/curated/tandem_cylinders_directppo_train16_v1"
mapping="$repo/artifacts/fcp003c_full_train_source_phase_mapping_20261005/mapping.json"
config="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
parent="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best"
output="$repo/artifacts/fcp008_force_readout_candidate_20261005"
container="fcp008-force-readout-candidate-20261005"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() { [[ -f "$1" && "$(sha "$1")" == "$2" ]]; }

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ "$(git rev-parse "$source_commit^{commit}")" == "$source_commit" ]]
[[ "$(git show "$source_commit:scripts/build_fcp008_force_readout_candidate.py" | sha256sum | awk '{print $1}')" == "$builder_sha" ]]
[[ "$(git show "$source_commit:scripts/spark_gpu_guard.py" | sha256sum | awk '{print $1}')" == "$guard_sha" ]]
require_sha "$approval" "$approval_sha"
require_sha "$base/manifest.json" 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2
require_sha "$base/splits/train.json" 1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89
require_sha "$base/normalization.json" f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1
require_sha "$train8/manifest.json" a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35
require_sha "$train16/manifest.json" 7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b
require_sha "$mapping" 57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92
require_sha "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
require_sha "$parent/FNO.0.2.mdlus" f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4
require_sha "$parent/checkpoint.0.2.pt" a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a
[[ ! -e "$output" ]] || { echo "exclusive output already exists: $output" >&2; exit 2; }
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]

if [[ "$mode" == --dry-run ]]; then
  echo "FC_P008_FORCE_READOUT_CANDIDATE_DRY_RUN_READY_NO_GPU"
  exit 0
fi
[[ "${FC_P008_EXECUTION_TOKEN:-}" == "EXECUTE_REVIEWED_FC_P008_FORCE_READOUT" ]]

mkdir "$output"
mkdir "$output/source_snapshot" "$output/launch_evidence"
git archive "$source_commit" src scripts | tar -x -C "$output/source_snapshot"
(cd "$output/source_snapshot" && find src scripts -type f -print0 | sort -z | xargs -0 sha256sum) >"$output/source_snapshot.sha256"
(cd "$output/source_snapshot" && sha256sum -c ../source_snapshot.sha256 >/dev/null)
cp "$approval" "$output/launch_evidence/fc_p008_execution_approval.json"
cp "$0" "$output/immutable_launcher.sh"

python3 - "$output" "$source_commit" "$builder_sha" "$guard_sha" "$image_id" "$approval_sha" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); source_commit,builder,guard,image_id,approval_sha=sys.argv[2:]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
payload={
 'status':'FC_P008_FORCE_READOUT_LAUNCH_VERIFIED','source_commit':source_commit,
 'implementation_sha256':builder,'guard_sha256':guard,'runtime_image_id':image_id,
 'execution_approval_sha256':approval_sha,
 'source_snapshot_manifest_sha256':sha(root/'source_snapshot.sha256'),
 'parent_model_sha256':'f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4',
 'parent_training_state_sha256':'a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a',
 'normalization_sha256':'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1',
 'resolved_config_sha256':'07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9',
 'source_phase_mapping_sha256':'57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92',
 'base_manifest_sha256':'5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2',
 'base_train_split_sha256':'1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89',
 'train8_manifest_sha256':'a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35',
 'train16_manifest_sha256':'7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b',
 'allocator_fraction':0.15,'minimum_mem_available_gib':20,
 'validation_accessed':False,'frozen_test_accessed':False,'ppo_executed':False}
path=root/'launch_receipt.json'
with tempfile.NamedTemporaryFile('w',dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.link(tmp,path); tmp.unlink()
PY

cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
set +e
timeout --signal=TERM --kill-after=30s 20m docker run --rm --name "$container" \
  --gpus device=0 --network none --cpus 8 --memory 90g --shm-size 2g \
  --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
  -e PYTHONPATH=/workspace/src:/workspace/scripts \
  -v "$output/source_snapshot/src:/workspace/src:ro" \
  -v "$output/source_snapshot/scripts:/workspace/scripts:ro" \
  -v "$base/train:/workspace/base/train:ro" \
  -v "$base/manifest.json:/workspace/base/manifest.json:ro" \
  -v "$base/splits:/workspace/base/splits:ro" \
  -v "$base/normalization.json:/workspace/base/normalization.json:ro" \
  -v "$train8/train:/workspace/train8/train:ro" \
  -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" \
  -v "$train16/train:/workspace/train16/train:ro" \
  -v "$train16/manifest.json:/workspace/train16/manifest.json:ro" \
  -v "$mapping:/workspace/evidence/source_phase_mapping.json:ro" \
  -v "$config:/workspace/config/resolved_config.yaml:ro" \
  -v "$parent:/workspace/parent:ro" \
  -v "$output/launch_evidence/fc_p008_execution_approval.json:/workspace/evidence/execution_approval.json:ro" \
  -v "$output:/workspace/output:rw" \
  -w /workspace "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 2 --poll-seconds 2 -- \
  python -u scripts/build_fcp008_force_readout_candidate.py \
    --base /workspace/base --train8 /workspace/train8 --train16 /workspace/train16 \
    --normalization /workspace/base/normalization.json \
    --config /workspace/config/resolved_config.yaml --checkpoint-dir /workspace/parent \
    --source-phase-mapping /workspace/evidence/source_phase_mapping.json \
    --approval /workspace/evidence/execution_approval.json \
    --output /workspace/output/candidate_build --chunk-size 10 \
  2>&1 | tee "$output/run.log"
status=${PIPESTATUS[0]}
set -e
[[ $status -eq 0 ]] || exit "$status"
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]

python3 - "$output" "$builder_sha" "$approval_sha" "$image_id" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); builder_sha,approval_sha,image_id=sys.argv[2:]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
result_path=root/'candidate_build/result.json'; result=json.loads(result_path.read_text())
assert result['status']=='FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE_COMPLETE_NOT_ADMISSION'
assert result['implementation_sha256']==builder_sha
assert result['execution_approval_sha256']==approval_sha
assert result['parent_checkpoint_epoch']==2 and result['calibration_generation']==1 and result['candidate_checkpoint_epoch']==0
assert result['optimizer_steps']==0 and result['architecture_changed'] is False
assert result['calibration_fit_performed'] is True and result['no_optimizer_training'] is True
assert result['fresh_official_load_checkpoint_return_epoch']==0
assert result['validation_accessed'] is result['frozen_test_accessed'] is result['ppo_executed'] is False
required=['candidate_build/result.json','candidate_build/source_inventory.json','candidate_build/train_features.npz','candidate_build/candidate/FNO.0.0.mdlus','candidate_build/candidate/checkpoint.0.0.pt','launch_receipt.json','source_snapshot.sha256','run.log']
hashes={name:sha(root/name) for name in required}
assert hashes['candidate_build/candidate/FNO.0.0.mdlus']==result['candidate_model_sha256']
assert hashes['candidate_build/candidate/checkpoint.0.0.pt']==result['candidate_state_sha256']
payload={'status':'FC_P008_FORCE_READOUT_CANDIDATE_VERIFIED','runtime_image_id':image_id,'implementation_sha256':builder_sha,'execution_approval_sha256':approval_sha,'candidate_checkpoint_epoch':0,'parent_checkpoint_epoch':2,'calibration_generation':1,'sha256':hashes,'validation_accessed':False,'frozen_test_accessed':False,'ppo_executed':False,'formal_evaluation_authorized':False}
path=root/'completion_receipt.json'
with tempfile.NamedTemporaryFile('w',dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.link(tmp,path); tmp.unlink()
PY
echo "FC_P008_FORCE_READOUT_CANDIDATE_VERIFIED"
