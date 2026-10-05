#!/usr/bin/env bash
# Guarded cache-only FC-P009 launcher. Dry-run is the default.
set -euo pipefail

repo="${FCP009_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$repo"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute) ;; *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2;; esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="02c0c99f7b8b48657269595993b1f3dba2a58bf6"
builder_sha="a2c83846714b3d5575840dcd89c0c8e955fc40f217dad011f3e74d277807bfdf"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"

base="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$repo/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$repo/data/curated/tandem_cylinders_directppo_train16_v1"
mapping="$repo/artifacts/fcp003c_full_train_source_phase_mapping_20261005/mapping.json"
config="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
parent="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best"
p008_cache="$repo/artifacts/fcp008_force_readout_candidate_20261005/candidate_build/train_features.npz"
approval="${FCP009_APPROVAL:-$repo/docs/FC_P009_EXECUTION_APPROVAL_20261005.json}"
approval_sha="${FCP009_APPROVAL_SHA256:-}"
output="$repo/artifacts/fcp009_free_ar_force_readout_cache_20261005"
container="fcp009-free-ar-cache-20261005"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() { [[ -f "$1" && "$(sha "$1")" == "$2" ]]; }

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ "$(git rev-parse "$source_commit^{commit}")" == "$source_commit" ]]
[[ "$(git show "$source_commit:scripts/build_fcp009_free_ar_force_readout_candidate.py" | sha256sum | awk '{print $1}')" == "$builder_sha" ]]
[[ "$(git show "$source_commit:scripts/spark_gpu_guard.py" | sha256sum | awk '{print $1}')" == "$guard_sha" ]]
require_sha "$base/manifest.json" 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2
require_sha "$base/splits/train.json" 1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89
for root in "$base" "$train8" "$train16"; do
  require_sha "$root/normalization.json" f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1
done
require_sha "$train8/manifest.json" a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35
require_sha "$train16/manifest.json" 7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b
require_sha "$mapping" 57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92
require_sha "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
require_sha "$parent/FNO.0.2.mdlus" f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4
require_sha "$parent/checkpoint.0.2.pt" a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a
require_sha "$p008_cache" 22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff
[[ ! -e "$output" ]] || { echo "exclusive output already exists: $output" >&2; exit 2; }
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]

if [[ "$mode" == --dry-run ]]; then
  echo "FC_P009_FREE_AR_CACHE_DRY_RUN_READY_NO_GPU"
  echo "execution_approval_present=$([[ -f "$approval" ]] && echo true || echo false)"
  exit 0
fi

[[ "${FCP009_EXECUTION_TOKEN:-}" == EXECUTE_REVIEWED_FC_P009_FREE_AR_CACHE ]]
[[ "$approval_sha" =~ ^[0-9a-f]{64}$ ]] && require_sha "$approval" "$approval_sha"
python3 - "$approval" "$builder_sha" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
required={"status":"FC_P009_FREE_AR_CACHE_EXECUTION_APPROVED","implementation_sha256":sys.argv[2],"parent_model_sha256":"f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4","alpha":0.0}
if any(value.get(k)!=v for k,v in required.items()): raise SystemExit("FC-P009 approval differs")
PY

mkdir "$output"
mkdir "$output/source_snapshot" "$output/launch_evidence"
git archive "$source_commit" -- src scripts conf | tar -x -C "$output/source_snapshot"
(cd "$output/source_snapshot" && find src scripts conf -type f -print0 | sort -z | xargs -0 sha256sum) >"$output/source_snapshot.sha256"
(cd "$output/source_snapshot" && sha256sum -c ../source_snapshot.sha256 >/dev/null)
cp "$approval" "$output/launch_evidence/fc_p009_execution_approval.json"
cp "$0" "$output/immutable_launcher.sh"

python3 - "$output" "$source_commit" "$builder_sha" "$guard_sha" "$image_id" "$approval_sha" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); commit,builder,guard,image_id,approval=sys.argv[2:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"status":"FC_P009_FREE_AR_CACHE_LAUNCH_VERIFIED","source_commit":commit,"implementation_sha256":builder,"guard_sha256":guard,"runtime_image_id":image_id,"execution_approval_sha256":approval,"source_snapshot_manifest_sha256":sha(root/"source_snapshot.sha256"),"parent_model_sha256":"f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4","parent_training_state_sha256":"a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a","p008_h1_cache_sha256":"22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff","batch_size":4,"workers":1,"allocator_fraction":0.15,"minimum_mem_available_gib":20,"timeout_minutes":25,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False}
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as s:
 t=pathlib.Path(s.name);json.dump(payload,s,indent=2,sort_keys=True);s.write("\n");s.flush();os.fsync(s.fileno())
os.link(t,root/"launch_receipt.json");t.unlink()
PY

cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
set +e
timeout --signal=TERM --kill-after=30s 25m docker run --rm --name "$container" \
  --gpus device=0 --network none --cpus 8 --memory 90g --shm-size 2g \
  --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
  -e PYTHONPATH=/workspace/src:/workspace/scripts \
  -v "$output/source_snapshot/src:/workspace/src:ro" \
  -v "$output/source_snapshot/scripts:/workspace/scripts:ro" \
  -v "$output/source_snapshot/conf:/workspace/conf:ro" \
  -v "$base/train:/workspace/base/train:ro" -v "$base/manifest.json:/workspace/base/manifest.json:ro" \
  -v "$base/splits/train.json:/workspace/base/splits/train.json:ro" -v "$base/normalization.json:/workspace/base/normalization.json:ro" \
  -v "$train8/train:/workspace/train8/train:ro" -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" -v "$train8/normalization.json:/workspace/train8/normalization.json:ro" \
  -v "$train16/train:/workspace/train16/train:ro" -v "$train16/manifest.json:/workspace/train16/manifest.json:ro" -v "$train16/normalization.json:/workspace/train16/normalization.json:ro" \
  -v "$mapping:/workspace/evidence/source_phase_mapping.json:ro" \
  -v "$config:/workspace/config/resolved_config.yaml:ro" -v "$parent:/workspace/parent:ro" \
  -v "$p008_cache:/workspace/evidence/p008_train_features.npz:ro" \
  -v "$output/launch_evidence/fc_p009_execution_approval.json:/workspace/evidence/execution_approval.json:ro" \
  -v "$output:/workspace/output:rw" -w /workspace "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 2 --poll-seconds 2 -- \
  python -u scripts/build_fcp009_free_ar_force_readout_candidate.py \
    --base /workspace/base --train8 /workspace/train8 --train16 /workspace/train16 \
    --normalization /workspace/base/normalization.json --config /workspace/config/resolved_config.yaml \
    --checkpoint-dir /workspace/parent --source-phase-mapping /workspace/evidence/source_phase_mapping.json \
    --p008-cache /workspace/evidence/p008_train_features.npz --approval /workspace/evidence/execution_approval.json \
    --output /workspace/output/cache_build --batch-size 4 --workers 1 \
  2>&1 | tee "$output/run.log"
status=${PIPESTATUS[0]}; set -e
[[ $status -eq 0 ]] || exit "$status"
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]

python3 - "$output" "$builder_sha" "$approval_sha" "$image_id" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]);builder,approval,image=sys.argv[2:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((root/"cache_build/result.json").read_text())
assert r["status"]=="FC_P009_TRAIN_ONLY_FREE_AR_FEATURE_CACHE_COMPLETE_NOT_ADMISSION"
assert r["input_sha256"]["implementation"]==builder and r["input_sha256"]["execution_approval"]==approval
assert r["parent_checkpoint_epoch"]==2 and r["alpha"]==0.0 and r["alpha_selection_performed"] is False
assert r["optimizer_steps"]==0 and r["architecture_changed"] is False and r["candidate_saved"] is False
assert r["validation_accessed"] is r["frozen_test_accessed"] is r["ppo_executed"] is False
assert r["parent_tensor_sha256_before"]==r["parent_tensor_sha256_after"]
names=["cache_build/result.json","cache_build/source_inventory.json","cache_build/window_inventory.json","cache_build/train_free_ar_features.npz","launch_receipt.json","source_snapshot.sha256","run.log"]
hashes={n:sha(root/n) for n in names}
assert hashes["cache_build/train_free_ar_features.npz"]==r["feature_cache_sha256"]
payload={"status":"FC_P009_FREE_AR_FEATURE_CACHE_VERIFIED_NOT_ADMISSION","runtime_image_id":image,"implementation_sha256":builder,"execution_approval_sha256":approval,"sha256":hashes,"candidate_saved":False,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False,"formal_evaluation_authorized":False}
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as s:
 t=pathlib.Path(s.name);json.dump(payload,s,indent=2,sort_keys=True);s.write("\n");s.flush();os.fsync(s.fileno())
os.link(t,root/"completion_receipt.json");t.unlink()
PY
echo "FC_P009_FREE_AR_FEATURE_CACHE_VERIFIED_NOT_ADMISSION"
