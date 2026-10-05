#!/usr/bin/env bash
# Project-owned two-arm finite-update diagnostic; no saved candidate or admission.
set -euo pipefail
repo=/workspace/fluid_control
mode=${1:---dry-run}
[[ "$mode" == --dry-run || "$mode" == --execute ]]
frozen="$repo/artifacts/fcp020_symmetric_statistics_source_20261005_immutable"
gradient_helper="$repo/artifacts/fcp019_gradient_alignment_source_20261005_immutable/scripts/probe_fcp019_gradient_alignment.py"
helper="$repo/artifacts/fcp014_train_objective_source_20261005_immutable/scripts"
source_root="$repo/artifacts/fcp013_training_source_1634c05_immutable"
candidate="$repo/artifacts/fcp018_reduced_rate_training_20261005"
config="$repo/artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
output="$repo/artifacts/fcp020_symmetric_statistics_20261005"
approval="$repo/docs/FC_P020_EXECUTION_APPROVAL_20261005.json"
image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
container=fcp020-symmetric-statistics-20261005
exec 9>"$repo/artifacts/fcp020_symmetric_statistics_execution.lock"
flock -n 9
sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]]
[[ "$(sha "$approval")" == "${FCP020_APPROVAL_SHA256:?set exact Lead-approved approval SHA}" ]]
python3 - "$repo" "$frozen" "$approval" "$0" <<'PY'
import hashlib,json,pathlib,sys
repo,frozen,approval,launcher=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(approval.read_text())
assert a['status']=='FC_P020_APPROVED_SYMMETRIC_STATISTICS_DIAGNOSTIC_NOT_ADMISSION'
assert a['launcher_sha256']==sha(launcher)
for name,expected in a['source_sha256'].items():
 p=frozen/name
 assert p.resolve().is_relative_to(frozen.resolve()) and sha(p)==expected
for name,expected in a['dependency_sha256'].items():
 p=repo/name
 assert p.resolve().is_relative_to(repo.resolve()) and sha(p)==expected
PY
if [[ "$mode" == --dry-run ]]; then
 echo FC_P020_LAUNCH_IDENTITIES_VERIFIED_NO_GPU_NO_PROBE
 exit 0
fi
python3 - <<'PY'
import subprocess
expected=dict(ActiveState='active',SubState='exited',Result='success',ExecMainCode='1',
              ExecMainStatus='0',MainPID='0',InvocationID='275365360254440aba18ed96aac58630')
raw=subprocess.check_output(['systemctl','--user','show','fluid-control-fcp019-gradient-alignment-20261005.service',
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
import hashlib,json,pathlib,sys
root,repo,approval=map(pathlib.Path,sys.argv[1:])
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
a=json.loads(approval.read_text())
assert sha(root/'posteval_fc_p018/receipt.json')==a['formal_receipt_sha256']
assert sha(repo/'artifacts/fcp019_gradient_alignment_20261005/result.json')=='1bd66e3cbf7c1200ff0af96d23bd62803d59422129eec5c5dddf7615fbcb183f'
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
PY
# Keep conservative startup headroom for optimizer state and sequential arm work.
# Recheck physical free memory after hashing, not only before dataset reads.
awk '/^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} END {exit !(a>=50*1024*1024 && f>=32*1024*1024)}' /proc/meminfo
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
 --memory 40g --pids-limit 2048 --shm-size 4g --tmpfs /tmp:rw,size=4g \
 -e XDG_CACHE_HOME=/tmp/cache -e LOCAL_CACHE=/tmp/physicsnemo-cache -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/workspace/probe:/workspace/project/src:/workspace/project/scripts \
 -v "$source_root:/workspace/project:ro" -v "$frozen/scripts:/workspace/probe:ro" -v "$helper:/workspace/helpers:ro" \
 -v "$gradient_helper:/workspace/gradient_helper.py:ro" \
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
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 --poll-seconds 2 -- \
 timeout -k 20 1800 python -u /workspace/probe/probe_fcp020_symmetric_statistics.py \
 --source-root /workspace/project --config /workspace/config.yaml \
 --diagnostic-script /workspace/helpers/diagnose_fcp014_train_objective.py \
 --gradient-helper /workspace/gradient_helper.py \
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
