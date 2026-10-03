#!/usr/bin/env bash
# Run fixed, non-selecting diagnostics only after the H50 full workflow exits.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
unit=fluid-control-dynamic-train8-h50-e5-4epoch-20261004.service
training="$root/artifacts/tandem_fno_dynamic_train8_h50_spark_h50_e5_dynamic4_20261004"
parent="$root/artifacts/tandem_fno_full40_free_ar_h50_epoch5_frozen_20261003"
output="$root/artifacts/fno_force_window_parent_vs_dynamic_h50_20261004"
image=fluid-control-physicsnemo:2.2.2
expected_image=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
[[ "$mode" == --dry-run || "$mode" == --execute ]] || exit 2
[[ -d "$parent" && -d "$training" ]] || { echo 'missing fixed parent/training'; exit 2; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$expected_image" ]] || exit 2
[[ ! -e "$output" ]] || { echo 'refusing existing diagnostic output'; exit 2; }
if [[ "$mode" == --dry-run ]]; then
  echo "FORCE_WINDOW_QUEUE_READY waiting_for=$unit parent=$parent candidate=$training/best output=$output"
  exit 0
fi
while systemctl --user is-active --quiet "$unit"; do sleep 30; done
unit_result="$(systemctl --user show "$unit" -p Result --value 2>/dev/null || true)"
[[ -z "$unit_result" || "$unit_result" == success ]] || {
  echo 'training workflow failed; diagnostic queue does not silently recover or select another candidate'; exit 3;
}
[[ -f "$training/validation10/diagnostic.json" && -f "$training/dynamic6/diagnostic.json" ]] || {
  echo 'full post-training diagnostics are not complete'; exit 3;
}
python3 - "$training" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
history=json.loads((p/'training_history.json').read_text())
if [r['epoch'] for r in history] != [1,2,3,4]: raise SystemExit('not all four epochs completed')
static=json.loads((p/'validation10/diagnostic.json').read_text())
dynamic=json.loads((p/'dynamic6/diagnostic.json').read_text())
if static.get('status')!='DEV30_VALIDATION_DIAGNOSTIC_COMPLETE': raise SystemExit('static diagnostic incomplete')
if dynamic.get('status') not in ('DYNAMIC6_FNO_DIAGNOSTIC_PASS','DYNAMIC6_FNO_DIAGNOSTIC_FAIL'):
    raise SystemExit('dynamic diagnostic incomplete')
PY
mkdir "$output"
mkdir "$output/candidate_checkpoint"
cp --reflink=auto "$training/best"/FNO.0.*.mdlus "$training/best"/checkpoint.0.*.pt "$output/candidate_checkpoint/"
chmod 444 "$output/candidate_checkpoint"/*
python3 - "$root" "$output" "$parent" "$training" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,out,parent,training=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
receipt={'scope':'fixed validation-only diagnostic, no checkpoint selection or PPO authorization',
         'training':str(training),'parent':str(parent),'sha256':{}}
for p in sorted((out/'candidate_checkpoint').iterdir()):
    if sha(p)!=sha(training/'best'/p.name): raise SystemExit('candidate copy SHA differs')
    receipt['sha256'][str(p.relative_to(out))]=sha(p)
for name in ('scripts/diagnose_fno_force_window.py','scripts/run_fno_force_window_after_training.sh',
             'scripts/evaluate_tandem_fno.py','scripts/train_tandem_fno.py'):
    receipt['sha256'][name]=sha(root/name)
(out/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
for label in parent candidate; do
  checkpoint="$parent"
  [[ "$label" != candidate ]] || checkpoint="$output/candidate_checkpoint"
  model_sha="$(python3 - "$checkpoint" <<'PY'
import hashlib,sys
from pathlib import Path
models=list(Path(sys.argv[1]).glob('FNO.0.*.mdlus'))
if len(models)!=1: raise SystemExit('exactly one immutable model required')
print(hashlib.sha256(models[0].read_bytes()).hexdigest())
PY
)"
  docker run --rm --network none --gpus device=0 --cpus 4 --memory 40g \
    --shm-size 1g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1 \
    --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=2 \
    --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly" \
    --mount "type=bind,src=$root/src,dst=/workspace/src,readonly" \
    --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly" \
    --mount "type=bind,src=$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1,dst=/workspace/base,readonly" \
    --mount "type=bind,src=$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1,dst=/workspace/dynamic,readonly" \
    --mount "type=bind,src=$checkpoint,dst=/workspace/checkpoint,readonly" \
    --mount "type=bind,src=$output,dst=/workspace/output" --workdir /workspace \
    "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
    python -u scripts/diagnose_fno_force_window.py --data /workspace/dynamic \
    --normalization-data /workspace/base --config /workspace/conf/tandem_fno_full40_h20.yaml \
    --checkpoint-dir /workspace/checkpoint --expected-model-sha "$model_sha" \
    --output "/workspace/output/${label}.json" 2>&1 | tee "$output/${label}.log"
done
