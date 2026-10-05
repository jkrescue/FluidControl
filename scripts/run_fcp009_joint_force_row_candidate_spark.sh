#!/usr/bin/env bash
# Guarded FC-P009 cache-fit candidate build. Dry-run is the default.
set -euo pipefail
repo="${FCP009_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"; cd "$repo"
mode="${1:---dry-run}"; case "$mode" in --dry-run|--execute) ;; *) exit 2;; esac
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="2c6be38021425ec8a7e18f6f967135155e2d6198"
builder_sha="dbe75763f8c480feefc54cb657f53848c24055e88ecc70a9a0aaaad544d81ecf"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
base="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
norm="$base/normalization.json"
config="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
parent="$repo/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best"
p009_root="$repo/artifacts/fcp009_free_ar_force_readout_cache_20261005"
p009_result="$p009_root/cache_build/result.json"
p009_cache="$p009_root/cache_build/train_free_ar_features.npz"
joint="$p009_root/cpu_joint_50_50_analysis.json"
approval="${FCP009_CANDIDATE_APPROVAL:-$repo/docs/FC_P009_CANDIDATE_EXECUTION_APPROVAL_20261005.json}"
approval_sha="${FCP009_CANDIDATE_APPROVAL_SHA256:-}"
output="$repo/artifacts/fcp009_joint_force_row_candidate_20261005"
container="fcp009-joint-force-row-candidate-20261005"
sha() { sha256sum "$1" | awk '{print $1}'; }; require_sha() { [[ -f "$1" && "$(sha "$1")" == "$2" ]]; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ "$(git rev-parse "$source_commit^{commit}")" == "$source_commit" ]]
[[ "$(git show "$source_commit:scripts/build_fcp009_joint_force_row_candidate.py" | sha256sum | awk '{print $1}')" == "$builder_sha" ]]
[[ "$(git show "$source_commit:scripts/spark_gpu_guard.py" | sha256sum | awk '{print $1}')" == "$guard_sha" ]]
require_sha "$base/manifest.json" 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2
require_sha "$base/splits/train.json" 1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89
require_sha "$norm" f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1
require_sha "$config" 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9
require_sha "$parent/FNO.0.2.mdlus" f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4
require_sha "$parent/checkpoint.0.2.pt" a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a
require_sha "$p009_result" 1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf
require_sha "$p009_cache" fc1b84fdd5a43d2b29e1f068531940bd6c12f320556d767a2540dfb5a238ca84
require_sha "$joint" 931fcd2ddd6901ddfbb3ecdbfe9774d7b1e5fe6d17479c87c44ccaf51ff2b0bc
[[ ! -e "$output" ]] && [[ -z "$(docker ps -aq --filter name="^${container}$")" ]]
if [[ "$mode" == --dry-run ]]; then echo "FC_P009_JOINT_CANDIDATE_DRY_RUN_READY_NO_GPU"; echo "approval_present=$([[ -f "$approval" ]] && echo true || echo false)"; exit 0; fi
[[ "${FCP009_CANDIDATE_TOKEN:-}" == EXECUTE_REVIEWED_FC_P009_JOINT_CANDIDATE ]]
[[ "$approval_sha" =~ ^[0-9a-f]{64}$ ]] && require_sha "$approval" "$approval_sha"
mkdir "$output"; mkdir "$output/source_snapshot" "$output/launch_evidence"
git archive "$source_commit" -- src scripts conf | tar -x -C "$output/source_snapshot"
(cd "$output/source_snapshot" && find src scripts conf -type f -print0 | sort -z | xargs -0 sha256sum) >"$output/source_snapshot.sha256"
(cd "$output/source_snapshot" && sha256sum -c ../source_snapshot.sha256 >/dev/null)
cp "$approval" "$output/launch_evidence/execution_approval.json"; cp "$0" "$output/immutable_launcher.sh"
python3 - "$output" "$source_commit" "$builder_sha" "$guard_sha" "$image_id" "$approval_sha" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
r=pathlib.Path(sys.argv[1]);commit,builder,guard,image,approval=sys.argv[2:];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v={"status":"FC_P009_JOINT_CANDIDATE_LAUNCH_VERIFIED","source_commit":commit,"implementation_sha256":builder,"guard_sha256":guard,"runtime_image_id":image,"execution_approval_sha256":approval,"source_snapshot_manifest_sha256":sha(r/"source_snapshot.sha256"),"parent_model_sha256":"f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4","p009_result_sha256":"1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf","p009_cache_sha256":"fc1b84fdd5a43d2b29e1f068531940bd6c12f320556d767a2540dfb5a238ca84","joint_analysis_sha256":"931fcd2ddd6901ddfbb3ecdbfe9774d7b1e5fe6d17479c87c44ccaf51ff2b0bc","minimum_mem_available_gib":20,"timeout_minutes":5,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False}
with tempfile.NamedTemporaryFile("w",dir=r,delete=False) as s:t=pathlib.Path(s.name);json.dump(v,s,indent=2,sort_keys=True);s.write("\n");s.flush();os.fsync(s.fileno())
os.link(t,r/"launch_receipt.json");t.unlink()
PY
cleanup(){ docker rm -f "$container" >/dev/null 2>&1 || true; }; trap cleanup EXIT INT TERM
set +e
timeout --signal=TERM --kill-after=30s 5m docker run --rm --name "$container" --gpus device=0 --network none --cpus 8 --memory 90g --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e PYTHONPATH=/workspace/src:/workspace/scripts \
 -v "$output/source_snapshot/src:/workspace/src:ro" -v "$output/source_snapshot/scripts:/workspace/scripts:ro" -v "$output/source_snapshot/conf:/workspace/conf:ro" \
 -v "$base/train:/workspace/base/train:ro" -v "$base/manifest.json:/workspace/base/manifest.json:ro" -v "$base/splits/train.json:/workspace/base/splits/train.json:ro" -v "$norm:/workspace/base/normalization.json:ro" \
 -v "$config:/workspace/config/resolved_config.yaml:ro" -v "$parent:/workspace/parent:ro" -v "$p009_result:/workspace/evidence/p009_result.json:ro" -v "$p009_cache:/workspace/evidence/p009_cache.npz:ro" -v "$joint:/workspace/evidence/joint_analysis.json:ro" -v "$output/launch_evidence/execution_approval.json:/workspace/evidence/approval.json:ro" -v "$output:/workspace/output:rw" -w /workspace "$image" \
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 2 --poll-seconds 2 -- python -u scripts/build_fcp009_joint_force_row_candidate.py --base /workspace/base --normalization /workspace/base/normalization.json --config /workspace/config/resolved_config.yaml --checkpoint-dir /workspace/parent --p009-result /workspace/evidence/p009_result.json --p009-cache /workspace/evidence/p009_cache.npz --joint-analysis /workspace/evidence/joint_analysis.json --approval /workspace/evidence/approval.json --output /workspace/output/candidate_build 2>&1 | tee "$output/run.log"
status=${PIPESTATUS[0]};set -e;[[ $status -eq 0 ]]||exit "$status";[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]
python3 - "$output" "$builder_sha" "$approval_sha" "$image_id" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
r=pathlib.Path(sys.argv[1]);builder,approval,image=sys.argv[2:];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();x=json.loads((r/"candidate_build/result.json").read_text())
assert x["status"]=="FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE_COMPLETE_NOT_ADMISSION" and x["candidate_checkpoint_epoch"]==0 and x["parent_checkpoint_epoch"]==2
assert x["input_sha256"]["implementation"]==builder and x["input_sha256"]["execution_approval"]==approval
assert x["optimizer_steps"]==0 and x["validation_accessed"] is x["frozen_test_accessed"] is x["ppo_executed"] is False and x["formal_evaluation_authorized"] is False
assert x["native_h1_sanity"]["batch_size"]==4 and x["native_h1_sanity"]["rollout_horizon"]==1 and x["native_h1_sanity"]["parent_candidate_state_bitwise_equal"] is True
names=["candidate_build/result.json","candidate_build/candidate/FNO.0.0.mdlus","candidate_build/candidate/checkpoint.0.0.pt","launch_receipt.json","source_snapshot.sha256","run.log"];h={n:sha(r/n) for n in names}
assert h[names[1]]==x["candidate_model_sha256"] and h[names[2]]==x["candidate_state_sha256"]
v={"status":"FC_P009_JOINT_FORCE_ROW_CANDIDATE_VERIFIED_NOT_ADMISSION","runtime_image_id":image,"implementation_sha256":builder,"execution_approval_sha256":approval,"sha256":h,"candidate_checkpoint_epoch":0,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False,"formal_evaluation_authorized":False}
with tempfile.NamedTemporaryFile("w",dir=r,delete=False) as s:t=pathlib.Path(s.name);json.dump(v,s,indent=2,sort_keys=True);s.write("\n");s.flush();os.fsync(s.fileno())
os.link(t,r/"completion_receipt.json");t.unlink()
PY
echo FC_P009_JOINT_FORCE_ROW_CANDIDATE_VERIFIED_NOT_ADMISSION
