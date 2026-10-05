#!/usr/bin/env bash
# Project-owned fixed eight-window accumulation training orchestration.
set -euo pipefail
repo=/workspace/fluid_control
mode=${1:---dry-run}
[[ "$mode" == --dry-run || "$mode" == --execute ]]
frozen="$repo/artifacts/fcp015_window_accumulation_source_20261005_immutable"
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
candidate="$repo/artifacts/fcp013_independent_force_fno_training_r2_20261005"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
parent="$repo/artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
output="$repo/artifacts/fcp015_window_accumulation_training_20261005"
approval="$repo/docs/FC_P015_EXECUTION_APPROVAL_20261005.json"
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
container=fcp015-window-accumulation-training-20261005
exec 9>"$repo/artifacts/fcp015_window_accumulation_execution.lock"
flock -n 9
sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]]
[[ "$(sha "$approval")" == "${FCP015_APPROVAL_SHA256:?set exact Lead-approved approval SHA}" ]]
python3 - "$repo" "$frozen" "$approval" "$0" <<'PY'
import hashlib,json,pathlib,sys
repo,frozen,approval,launcher=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(approval.read_text())
assert a['status']=='FC_P015_APPROVED_FIXED_ACCUMULATION_TRAINING_NOT_ADMISSION'
assert a['launcher_sha256']==sha(launcher)
for name,expected in a['source_sha256'].items():
 p=frozen/name
 assert p.resolve().is_relative_to(frozen.resolve()) and sha(p)==expected
for name,expected in a['dependency_sha256'].items():
 p=repo/name
 assert p.resolve().is_relative_to(repo.resolve()) and sha(p)==expected
PY
if [[ "$mode" == --dry-run ]]; then
 echo FC_P015_LAUNCH_IDENTITIES_VERIFIED_NO_GPU_NO_TRAINING
 exit 0
fi
[[ "$(systemctl --user show fluid-control-fcp013-posteval-r2-20261005.service -p SubState --value)" == exited ]]
[[ "$(systemctl --user show fluid-control-fcp013-posteval-r2-20261005.service -p ExecMainStatus --value)" == 0 ]]
[[ "$(systemctl --user show fluid-control-fcp013-posteval-r2-20261005.service -p InvocationID --value)" == 7235b2f06282435a89b84964e384c60f ]]
[[ -f "$candidate/posteval_fc_p013/receipt.json" ]]
gpu_pids=$(timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_pids" ]]
! docker inspect "$container" >/dev/null 2>&1
[[ ! -e "$output" ]]
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=80*1024*1024 && f>=30*1024*1024)}' /proc/meminfo
python3 - "$candidate" "$repo" "$approval" <<'PY'
import hashlib,json,pathlib,sys
root,repo,approval=map(pathlib.Path,sys.argv[1:])
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
a=json.loads(approval.read_text())
assert sha(root/'posteval_fc_p013/receipt.json')==a['formal_receipt_sha256']
assert sha(repo/'artifacts/fcp014_train_objective_diagnostic_20261005/result.json')=='5550140b818a3d9028073b913cf994da99d9d3e917ea24896e4a9d477b8e95a7'
assert sha(root/'candidate_audit.json')=='1c280b291ae7ded1f8e63ccc46ba40a7e085e7d7b6fa69f0dddff18ffac704a7'
assert sha(root/'completion_receipt.json')=='3c53a7fb94d5ad5f29bed6522317389f02842d078d157d63e20b90b6948e2d90'
assert sha(root/'fixed_six_diagnostics/result.json')=='8e0255c955c1b26fdff240a0854fc0a92d3bd247cc38ed6c268fc1d397cec873'
audit=json.loads((root/'candidate_audit.json').read_text())
for name,expected in audit['sha256'].items():
 p=root/name
 assert p.resolve().is_relative_to(root.resolve()) and sha(p)==expected
mapping=audit['train_hdf_sha256']; assert len(mapping)==44
paths={repo/name for name in mapping}
for directory in {p.parent for p in paths}:
 assert set(directory.glob('*.h5'))=={p for p in paths if p.parent==directory}
for name,expected in mapping.items():
 p=repo/name
 assert p.resolve().is_relative_to((repo/'data/curated').resolve()) and '/train/' in name and sha(p)==expected
print(json.dumps({'event':'train_hdf_bytes_verified','files':44,'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()}),flush=True)
PY
mkdir "$output"
cp "$0" "$output/immutable_launcher.sh"
cp "$approval" "$output/execution_approval.json"
watch_pid=
cleanup() {
 [[ -z "$watch_pid" ]] || kill "$watch_pid" 2>/dev/null || true
 timeout -k 5 15 docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
(
 watcher_failure() {
  code=$?
  if (( code != 0 )); then
   touch "$output/resource_watcher_unexpected_exit" || true
   timeout -k 5 30 docker stop --timeout 20 "$container" >> "$output/resource_watch.jsonl" 2>&1 || timeout -k 5 10 docker kill "$container" >> "$output/resource_watch.jsonl" 2>&1 || true
  fi
 }
 trap watcher_failure EXIT
 trap 'exit 0' TERM
 started=$(date +%s)
 while sleep 2; do
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  physical_free=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
  printf '{"timestamp":%s,"mem_available_kib":%s,"mem_free_kib":%s}\n' "$(date +%s)" "$available" "$physical_free" >> "$output/resource_watch.jsonl"
  updated=$(stat -c %Y "$output/run.log" 2>/dev/null || echo "$started")
  if (( available < 20*1024*1024 || physical_free < 20*1024*1024 || $(date +%s)-started > 14460 || $(date +%s)-updated > 300 )); then
   touch "$output/resource_or_deadline_violation"
  fi
  if [[ -f "$output/resource_or_deadline_violation" ]]; then
   timeout -k 5 30 docker stop --timeout 20 "$container" >> "$output/resource_watch.jsonl" 2>&1 || timeout -k 5 10 docker kill "$container" >> "$output/resource_watch.jsonl" 2>&1 || true
  fi
 done
) &
watch_pid=$!
set +e
(
timeout -k 20 14520 docker run --rm --name "$container" --user 1000:1000 --gpus device=0 \
 --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
 --memory 90g --pids-limit 2048 --shm-size 8g --tmpfs /tmp:rw,size=8g \
 -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/workspace/diagnostic:/workspace/project/src:/workspace/project/scripts \
 -v "$source_root:/workspace/project:ro" -v "$frozen/scripts:/workspace/diagnostic:ro" \
 -v "$repo/artifacts/fcp014_train_objective_source_20261005_immutable/scripts/diagnose_fcp014_train_objective.py:/workspace/p014_diagnostic.py:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/base:/workspace/base:ro" \
 -v "$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train:/workspace/base/train:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train8:/workspace/train8:ro" \
 -v "$repo/data/curated/tandem_cylinders_dynamic_train8_v1/train:/workspace/train8/train:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train16:/workspace/train16:ro" \
 -v "$repo/data/curated/tandem_cylinders_directppo_train16_v1/train:/workspace/train16/train:ro" \
 -v "$parent:/workspace/parent:ro" -v "$config:/workspace/config.yaml:ro" \
 -v "$output:/workspace/output:rw" \
 -w /workspace/project "$image" \
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .45 --margin-gib 4 --poll-seconds 2 -- \
 timeout -k 20 14400 python -u /workspace/diagnostic/train_fcp015_window_accumulation.py \
 --source-root /workspace/project --diagnostic-script /workspace/p014_diagnostic.py \
 --config /workspace/config.yaml --parent /workspace/parent \
 --output /workspace/output/candidate --execute 2>&1 | tee "$output/run.log"
) &
run_pid=$!
while kill -0 "$run_pid" 2>/dev/null; do
 if ! kill -0 "$watch_pid" 2>/dev/null; then
  touch "$output/resource_watcher_unexpected_exit"
  timeout -k 5 30 docker stop --timeout 20 "$container" >> "$output/resource_watch.jsonl" 2>&1 || timeout -k 5 10 docker kill "$container" >> "$output/resource_watch.jsonl" 2>&1 || true
 fi
 sleep 1
done
wait "$run_pid"
run_status=$?
set -e
kill -0 "$watch_pid" 2>/dev/null || { touch "$output/resource_watcher_unexpected_exit"; exit 75; }
kill "$watch_pid"
wait "$watch_pid" || watcher_status=$?
watch_pid=
[[ "${watcher_status:-0}" == 0 ]]
[[ ! -f "$output/resource_or_deadline_violation" ]]
[[ ! -f "$output/resource_watcher_unexpected_exit" ]]
[[ "$run_status" == 0 ]]
