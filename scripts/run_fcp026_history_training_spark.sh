#!/usr/bin/env bash
# Staged only: requires a separately approved immutable source tree/receipt.
set -euo pipefail
repo=/workspace/fluid_control
mode=${1:---dry-run}
[[ "$mode" == --dry-run || "$mode" == --execute ]]
arm=${2:?specify matched history arm 1 or 4}
[[ "$arm" == 1 || "$arm" == 4 ]]
frozen="$repo/artifacts/fcp026_history_training_source_20261005_immutable"
output="$repo/artifacts/fcp026_history_training_k${arm}_20261005"
approval="$repo/docs/FC_P026_K${arm}_TRAINING_EXECUTION_APPROVAL_20261005.json"
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
helper="$repo/artifacts/fcp014_train_objective_source_20261005_immutable/scripts"
gradient="$repo/artifacts/fcp019_gradient_alignment_source_20261005_immutable/scripts/probe_fcp019_gradient_alignment.py"
accumulation="$repo/artifacts/fcp015_window_accumulation_source_20261005_immutable/scripts/train_fcp015_window_accumulation.py"
comparison="$repo/artifacts/fcp020_symmetric_statistics_source_20261005_immutable/scripts/probe_fcp020_symmetric_statistics.py"
candidate="$repo/artifacts/fcp018_reduced_rate_training_20261005"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
view="$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view"
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
container=fcp026-history-training-k${arm}-20261005
token=${FCP026_TRAINING_APPROVAL_SHA256:?exact arm-specific Lead-approved receipt SHA required}
# Shared across K1/K4: matched arms never overlap in unified memory.
exec 9>"$repo/artifacts/fcp026_history_training_execution.lock"
flock -n 9
[[ "$(sha256sum "$approval" | awk '{print $1}')" == "$token" ]]
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]]
python3 - "$repo" "$frozen" "$approval" "$0" "$arm" <<'PY'
import hashlib,json,pathlib,sys
repo,frozen,approval,launcher=map(pathlib.Path,sys.argv[1:5]);arm=int(sys.argv[5])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a=json.loads(approval.read_text())
assert a['status']=='FC_P026_APPROVED_MATCHED_HISTORY_TRAINING_NOT_ADMISSION' and a['history_k']==arm
assert a['launcher_sha256']==sha(launcher)
assert a['source_sha256'] and a['dependency_sha256']
assert {'scripts/p026_state_history.py','scripts/p026_history_objective.py','scripts/probe_fcp026_history_resource.py','scripts/train_fcp026_history.py','scripts/p026_history_inference.py'} <= set(a['source_sha256'])
for root,mapping in [(frozen,a['source_sha256']),(repo,a['dependency_sha256'])]:
 for name,digest in mapping.items():
  p=root/name
  assert p.resolve().is_relative_to(root.resolve()) and sha(p)==digest
import importlib.util
spec=importlib.util.spec_from_file_location('protocol_trainer',frozen/'scripts/train_fcp026_history.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
assert a['effective_protocol']==trainer.protocol(arm)
assert a['effective_protocol_sha256']==trainer.canonical_sha(trainer.protocol(arm))
receipt=repo/a['inventory_receipt']
assert receipt.resolve().is_relative_to(repo.resolve()) and sha(receipt)==a['inventory_receipt_sha256']
pre=json.loads(receipt.read_text())
assert pre['status']=='FC_P026_CPU_TRAIN_INVENTORY_ORDER_VERIFIED_NOT_TRAINING_APPROVAL'
assert pre['inventory']==trainer.protocol(arm)['inventory'] and pre['sampler_order_sha256']==trainer.ORDER_SHA
assert pre['trainer_sha256']==sha(frozen/'scripts/train_fcp026_history.py')
assert pre['protocols'][str(arm)]['canonical_sha256']==a['effective_protocol_sha256']
PY
if [[ "$mode" == --dry-run ]]; then
 echo FC_P026_TRAINING_IDENTITIES_VERIFIED_NO_GPU_EXECUTION
 exit 0
fi
[[ ! -e "$output" ]]
! docker inspect "$container" >/dev/null 2>&1
gpu_pids=$(timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_pids" ]]
python3 - "$repo" "$candidate/candidate_audit.json" <<'PY'
import hashlib,json,os,pathlib,stat,sys
repo,audit=map(pathlib.Path,sys.argv[1:])
assert hashlib.sha256(audit.read_bytes()).hexdigest()=='03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9'
mapping=json.loads(audit.read_text())['train_hdf_sha256'];assert len(mapping)==44
paths={repo/name for name in mapping}
for directory in {p.parent for p in paths}:
 assert set(directory.glob('*.h5'))=={p for p in paths if p.parent==directory}
for name,digest in mapping.items():
 p=repo/name
 assert p.resolve().is_relative_to((repo/'data/curated').resolve()) and '/train/' in name
 assert not any(x.is_symlink() for x in [p,*p.parents])
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
  assert hashlib.file_digest(f,'sha256').hexdigest()==digest
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print(json.dumps(dict(event='exact44_train_hash_and_bounded_clean_cache_advice',files=44,data_modified=False)),flush=True)
PY
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=50*1024*1024 && f>=30*1024*1024)}' /proc/meminfo
mkdir "$output"
cp "$0" "$output/immutable_launcher.sh"
cp "$approval" "$output/execution_approval.json"
watch_pid=
owned_capture() {
 local cid
 cid=$(timeout 5 docker inspect "$container" --format '{{.Id}}' 2>/dev/null) || { timeout 5 docker info >/dev/null 2>&1; return $?; }
 timeout 5 docker inspect "$cid" | python3 -c 'import json,pathlib,sys; c=json.load(sys.stdin)[0]; assert c["Image"]==sys.argv[1]; assert c["Config"]["Labels"].get("fc.p026.approval")==sys.argv[2]; assert any(m["Source"]==sys.argv[3] and m["Destination"]=="/workspace/output" for m in c["Mounts"]); p=pathlib.Path(sys.argv[3])/"runtime_container.json"; p.write_text(json.dumps(c,indent=2)) if not p.exists() else None; print(c["Id"])' "$image" "$token" "$output"
}
owned_stop() {
 local cid
 cid=$(owned_capture) || return 1
 [[ -n "$cid" ]] || return 0
 timeout -k 5 20 docker rm -f "$cid" >> "$output/cleanup.log" 2>&1
}
cleanup() {
 [[ -z "$watch_pid" ]] || kill "$watch_pid" 2>/dev/null || true
 # Repeat exact-owned discovery to cover daemon create/start races.
 for attempt in 1 2 3; do owned_stop || true; sleep 1; done
}
trap cleanup EXIT
trap 'exit 143' TERM INT
(
 trap 'exit 0' TERM
 trap 'code=$?; if (( code != 0 )); then touch "$output/watcher_failure"; owned_stop || true; fi' EXIT
 begin=$SECONDS
 while sleep 2; do
  read -r available free < <(awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {print a,f}' /proc/meminfo)
  [[ "$available" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]]
  printf '{"timestamp":%s,"mem_available_kib":%s,"mem_free_kib":%s}\n' "$(date +%s)" "$available" "$free" >> "$output/resource_watch.jsonl"
  owned_capture >/dev/null
  if (( available<20*1024*1024 || free<20*1024*1024 || SECONDS-begin>14500 )); then touch "$output/resource_violation"; fi
  if [[ -e "$output/resource_violation" ]]; then owned_stop; fi
 done
) &
watch_pid=$!
(
set +e
timeout -k 20 14540 docker run --rm --name "$container" --label "fc.p026.approval=$token" \
 --user 1000:1000 --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
 --memory 12g --pids-limit 2048 --shm-size 4g --tmpfs /tmp:rw,size=4g \
 -e WARP_CACHE_PATH=/tmp/warp -e XDG_CACHE_HOME=/tmp/cache -e LOCAL_CACHE=/tmp/physicsnemo-cache -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/workspace/probe:/workspace/project/src:/workspace/project/scripts \
 -v "$source_root:/workspace/project:ro" -v "$frozen/scripts:/workspace/probe:ro" -v "$helper:/workspace/helpers:ro" \
 -v "$gradient:/workspace/gradient_helper.py:ro" \
 -v "$accumulation:/workspace/accumulation_helper.py:ro" -v "$comparison:/workspace/comparison_helper.py:ro" \
 -v "$view/base:/workspace/base:ro" -v "$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train:/workspace/base/train:ro" \
 -v "$view/train8:/workspace/train8:ro" -v "$repo/data/curated/tandem_cylinders_dynamic_train8_v1/train:/workspace/train8/train:ro" \
 -v "$view/train16:/workspace/train16:ro" -v "$repo/data/curated/tandem_cylinders_directppo_train16_v1/train:/workspace/train16/train:ro" \
 -v "$candidate/candidate:/workspace/candidate:ro" -v "$candidate/candidate_audit.json:/workspace/candidate_audit.json:ro" \
 -v "$config:/workspace/config.yaml:ro" -v "$output:/workspace/output:rw" \
 -w /workspace/project "$image" \
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .06 --margin-gib 4 --poll-seconds 2 -- \
 timeout -k 20 14400 python -u /workspace/probe/train_fcp026_history.py \
 --source-root /workspace/project --diagnostic-script /workspace/helpers/diagnose_fcp014_train_objective.py \
 --gradient-helper /workspace/gradient_helper.py --history-module /workspace/probe/p026_state_history.py \
 --history-objective /workspace/probe/p026_history_objective.py --config /workspace/config.yaml \
 --history-inference-module /workspace/probe/p026_history_inference.py --resource-helper /workspace/probe/probe_fcp026_history_resource.py \
 --accumulation-helper /workspace/accumulation_helper.py --comparison-helper /workspace/comparison_helper.py \
 --flow-parent /workspace/candidate/flow --aerodynamic-parent /workspace/candidate/aerodynamic \
 --parent-manifest /workspace/candidate/dual_model_manifest.json --candidate-audit /workspace/candidate_audit.json \
 --history-k "$arm" --output /workspace/output/candidate --execute 2>&1 | tee "$output/run.log"
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > "$output/container_exit_code"
exit "$code"
) &
run_pid=$!
while kill -0 "$run_pid" 2>/dev/null; do
 if ! kill -0 "$watch_pid" 2>/dev/null; then
  touch "$output/watcher_failure"
  # Keep supervising the real pipeline until it exits or its14540s timeout.
  # Killing only its subshell can orphan an in-flight daemon create/start.
  # Retry owned discovery on EVERY iteration, including late-created containers.
  owned_stop || true
 fi
 sleep 2
done
set +e
wait "$run_pid"
code=$?
set -e
kill -0 "$watch_pid" 2>/dev/null || { touch "$output/watcher_failure"; exit 70; }
kill "$watch_pid"
wait "$watch_pid" || { touch "$output/watcher_failure"; exit 70; }
watch_pid=
[[ ! -e "$output/resource_violation" && ! -e "$output/watcher_failure" ]] || exit 70
[[ -s "$output/runtime_container.json" && -s "$output/container_exit_code" ]] || exit 71
exit "$code"
