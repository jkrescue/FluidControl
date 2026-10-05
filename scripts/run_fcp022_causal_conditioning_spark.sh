#!/usr/bin/env bash
# Project-owned matched six-window input comparison; no candidate/admission.
set -euo pipefail
repo=/workspace/fluid_control
mode=${1:---dry-run}
[[ "$mode" == --dry-run || "$mode" == --execute ]]
frozen="$repo/artifacts/fcp022_causal_conditioning_source_20261005_immutable"
comparison_helper="$repo/artifacts/fcp020_symmetric_statistics_source_20261005_immutable/scripts/probe_fcp020_symmetric_statistics.py"
gradient_helper="$repo/artifacts/fcp019_gradient_alignment_source_20261005_immutable/scripts/probe_fcp019_gradient_alignment.py"
helper="$repo/artifacts/fcp014_train_objective_source_20261005_immutable/scripts"
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
candidate="$repo/artifacts/fcp018_reduced_rate_training_20261005"
raw_view="$repo/artifacts/fcp021_raw_source_view_20261005_immutable"
causal_audit="$repo/artifacts/causal_force_input_audit_20261005/timestamp_audit.json"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
output="$repo/artifacts/fcp022_causal_conditioning_20261005"
approval="$repo/docs/FC_P022_EXECUTION_APPROVAL_20261005.json"
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
container=fcp022-causal-conditioning-20261005
exec 9>"$repo/artifacts/fcp022_causal_conditioning_execution.lock"
flock -n 9
sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]]
[[ "$(sha "$approval")" == "${FCP022_APPROVAL_SHA256:?set exact Lead-approved approval SHA}" ]]
python3 - "$repo" "$frozen" "$approval" "$0" <<'PY'
import hashlib,json,pathlib,sys
repo,frozen,approval,launcher=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(approval.read_text())
assert a['status']=='FC_P022_APPROVED_CAUSAL_INPUT_COMPARISON_NOT_ADMISSION'
assert a['launcher_sha256']==sha(launcher)
for name,expected in a['source_sha256'].items():
 p=frozen/name
 assert p.resolve().is_relative_to(frozen.resolve()) and sha(p)==expected
for name,expected in a['dependency_sha256'].items():
 p=repo/name
 assert p.resolve().is_relative_to(repo.resolve()) and sha(p)==expected
PY
if [[ "$mode" == --dry-run ]]; then
 echo FC_P022_LAUNCH_IDENTITIES_VERIFIED_NO_GPU_NO_TRAINING
 exit 0
fi
python3 - <<'PY'
import subprocess
expected=dict(ActiveState='active',SubState='exited',Result='success',ExecMainCode='1',
              ExecMainStatus='0',MainPID='0',InvocationID='abb778c24a494eaa881eaa2669c5b57f')
raw=subprocess.check_output(['systemctl','--user','show','fluid-control-fcp021-causal-resource-r2-20261005.service',
                            *[arg for key in expected for arg in ('-p',key)]],text=True,timeout=10)
actual=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
assert all(actual.get(key)==value for key,value in expected.items()), actual
PY
[[ -f "$candidate/posteval_fc_p018/receipt.json" ]]
gpu_pids=$(timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_pids" ]]
! docker inspect "$container" >/dev/null 2>&1
[[ ! -e "$output" ]]
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=50*1024*1024 && f>=30*1024*1024)}' /proc/meminfo
python3 - "$candidate" "$repo" "$approval" <<'PY'
import hashlib,json,pathlib,sys,os,stat
root,repo,approval=map(pathlib.Path,sys.argv[1:])
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
a=json.loads(approval.read_text())
failed=repo/'artifacts/fcp021_causal_resource_probe_20261005'
assert sha(failed/'result.json.failure.json')=='30c1d17c7692a85ff60f62e11eb5700d1fdda1a49163e543253f44cf828317b5'
assert not (failed/'result.json').exists()
assert sha(repo/'artifacts/fcp021_causal_resource_probe_r2_20261005/result.json')=='975fc40bbb88d5d0ab3d239ee0ce994bc0635aabcad9b7d74568bf73f1a8ce7a'
assert sha(root/'posteval_fc_p018/receipt.json')==a['formal_receipt_sha256']
assert sha(repo/'artifacts/fcp020_symmetric_statistics_20261005/result.json')=='a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042'
assert sha(root/'candidate_audit.json')=='03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9'
assert sha(root/'completion_receipt.json')=='bce5fb4688e6fc784c309a4e63979a06603270c8b1e5b336572a1092306205bf'
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

raw_audit=repo/'artifacts/causal_force_input_audit_20261005/timestamp_audit.json'
assert sha(raw_audit)=='72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2'
raw_sources=json.loads(raw_audit.read_text())['raw_source_sha256']
view=repo/'artifacts/fcp021_raw_source_view_20261005_immutable'
assert len(raw_sources)==274 and view.is_dir() and not view.is_symlink()
assert not any(p.is_symlink() for p in view.rglob('*'))
assert {str(p.relative_to(view)) for p in view.rglob('*') if p.is_file()}==set(raw_sources)
for name,digest in raw_sources.items():
 p=view/name
 assert p.resolve().is_relative_to(view.resolve()) and sha(p)==digest
print(json.dumps({'event':'causal_raw_sources_verified','files':274}),flush=True)
# Advise eviction only for clean pages of the exact verified training HDF files
# just read by this process. No global cache operation or data modification.
assert hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED')
def memory():
 return {line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemFree:','MemAvailable:'))}
before=memory(); advised_bytes=0
for name,expected in mapping.items():
 path=repo/name
 assert path.resolve().is_relative_to((repo/'data/curated').resolve())
 assert not any(p.is_symlink() for p in [path,*path.parents])
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as handle:
  info=os.fstat(handle.fileno()); assert stat.S_ISREG(info.st_mode)
  assert hashlib.file_digest(handle,'sha256').hexdigest()==expected
  os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  advised_bytes+=info.st_size
print(json.dumps({'event':'verified_train_read_cache_release_advised','files':len(mapping),'file_bytes':advised_bytes,'before_kib':before,'after_kib':memory(),'data_modified':False,'reclamation_guaranteed':False}),flush=True)
PY
# Keep conservative startup headroom for two sequential sixteen-update arms.
# Recheck physical free memory after hashing, not only before dataset reads.
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=50*1024*1024 && f>=30*1024*1024)}' /proc/meminfo
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
  if (( available < 20*1024*1024 || physical_free < 20*1024*1024 || $(date +%s)-started > 1900 )); then
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
timeout -k 20 1940 docker run --rm --name "$container" --user 1000:1000 --gpus device=0 \
 --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
 --memory 12g --pids-limit 2048 --shm-size 4g --tmpfs /tmp:rw,size=4g \
 -e WARP_CACHE_PATH=/tmp/warp -e XDG_CACHE_HOME=/tmp/cache -e LOCAL_CACHE=/tmp/physicsnemo-cache -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/workspace/probe:/workspace/project/src:/workspace/project/scripts \
 -v "$source_root:/workspace/project:ro" -v "$frozen/scripts:/workspace/probe:ro" -v "$helper:/workspace/helpers:ro" \
 -v "$gradient_helper:/workspace/gradient_helper.py:ro" \
 -v "$comparison_helper:/workspace/comparison_helper.py:ro" \
 -v "$raw_view:/workspace/raw_sources:ro" -v "$causal_audit:/workspace/causal_audit.json:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/base:/workspace/base:ro" \
 -v "$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train:/workspace/base/train:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train8:/workspace/train8:ro" \
 -v "$repo/data/curated/tandem_cylinders_dynamic_train8_v1/train:/workspace/train8/train:ro" \
 -v "$repo/artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/train16:/workspace/train16:ro" \
 -v "$repo/data/curated/tandem_cylinders_directppo_train16_v1/train:/workspace/train16/train:ro" \
 -v "$candidate/candidate:/workspace/candidate:ro" \
 -v "$candidate/candidate_audit.json:/workspace/candidate_audit.json:ro" -v "$config:/workspace/config.yaml:ro" \
 -v "$output:/workspace/output:rw" \
 -w /workspace/project "$image" \
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .06 --margin-gib 4 --poll-seconds 2 -- \
 timeout -k 20 1800 python -u /workspace/probe/probe_fcp022_causal_conditioning.py \
 --source-root /workspace/project --config /workspace/config.yaml \
 --diagnostic-script /workspace/helpers/diagnose_fcp014_train_objective.py \
 --gradient-helper /workspace/gradient_helper.py \
 --comparison-helper /workspace/comparison_helper.py \
 --causal-module /workspace/probe/p021_causal_force.py --causal-audit /workspace/causal_audit.json --raw-source-view /workspace/raw_sources \
 --candidate /workspace/candidate --candidate-audit /workspace/candidate_audit.json \
 --output /workspace/output/result.json --execute 2>&1 | tee "$output/run.log"
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
