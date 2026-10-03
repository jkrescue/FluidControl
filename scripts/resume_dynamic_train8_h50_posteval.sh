#!/usr/bin/env bash
# Resume only the immutable H50 validation suites after the legacy mount-name failure.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
token="${DYNAMIC_H50_POSTEVAL_APPROVAL_TOKEN:-}"
run="$root/artifacts/tandem_fno_dynamic_train8_h50_spark_h50_e5_dynamic4_20261004"
best="$run/best"
base="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
dynamic="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
output="$run/posteval_resume_v1"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
[[ "$mode" == --dry-run || "$mode" == --execute ]] || { echo "mode must be --dry-run or --execute" >&2; exit 2; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image ID differs" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing existing resume output" >&2; exit 2; }

read -r epoch model_sha < <(python3 - "$base" "$dynamic" "$best" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
base,dynamic,best=map(Path,sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(base/'manifest.json')!='5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2': raise SystemExit('dev30 manifest differs')
if sha(base/'normalization.json')!='f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1': raise SystemExit('normalization differs')
if (base/'frozen_test').exists() or len(list((base/'train').glob('*.h5')))!=20 or len(list((base/'validation').glob('*.h5')))!=10: raise SystemExit('dev30 split differs')
dm=json.loads((dynamic/'manifest.json').read_text())
if sha(dynamic/'manifest.json')!='bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae' or dm.get('training_access')!='FORBIDDEN': raise SystemExit('dynamic6 identity differs')
models=list(best.glob('FNO.0.*.mdlus'))
if len(models)!=1: raise SystemExit('best must contain one model')
m=re.fullmatch(r'FNO\.0\.(\d+)\.mdlus',models[0].name)
if not m or m.group(1)!='4': raise SystemExit('immutable selected epoch must be 4')
print(m.group(1),sha(models[0]))
PY
)
echo "DYNAMIC_H50_POSTEVAL_RESUME_READY epoch=$epoch model_sha=$model_sha output=$output"
[[ "$mode" == --execute ]] || exit 0
[[ "$token" == EXECUTE_REVIEWED_DYNAMIC_H50_POSTEVAL_RESUME ]] || { echo "explicit posteval token required" >&2; exit 2; }

mkdir "$output"
mkdir "$output/immutable_checkpoint" "$output/validation10" "$output/dynamic6"
cp --reflink=auto "$best/FNO.0.$epoch.mdlus" "$output/immutable_checkpoint/"
chmod 0444 "$output/immutable_checkpoint/FNO.0.$epoch.mdlus"
[[ "$(sha256sum "$output/immutable_checkpoint/FNO.0.$epoch.mdlus" | awk '{print $1}')" == "$model_sha" ]] || { echo "checkpoint copy differs" >&2; exit 2; }

base_cmd=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 90g
  --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1
  --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$base,dst=/workspace/devdata,readonly"
  --mount "type=bind,src=$output/immutable_checkpoint,dst=/workspace/checkpoint,readonly"
  --mount "type=bind,src=$output,dst=/workspace/output"
  --workdir /workspace)

"${base_cmd[@]}" "$image" python -u scripts/spark_gpu_guard.py \
  --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py --data /workspace/devdata --normalization-data /workspace/devdata \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint \
  --split validation --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4 \
  --action-mode observed --visualizations-per-horizon 0 \
  --output /workspace/output/validation10/evaluation.json \
  --segment-metrics-output /workspace/output/validation10/segments.json \
  2>&1 | tee "$output/validation10/evaluate.log"
python3 scripts/audit_dev30_validation_diagnostic.py \
  --report "$output/validation10/evaluation.json" --segments "$output/validation10/segments.json" \
  --data "$base" --checkpoint-dir "$output/immutable_checkpoint" \
  --candidate-kind dev30_free_ar_development --output "$output/validation10/diagnostic.json"

"${base_cmd[@]}" --mount "type=bind,src=$dynamic,dst=/workspace/dynamic,readonly" \
  "$image" python -u scripts/spark_gpu_guard.py \
  --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/devdata \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint \
  --split validation --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 8 \
  --action-mode observed --visualizations-per-horizon 0 \
  --output /workspace/output/dynamic6/evaluation.json \
  --segment-metrics-output /workspace/output/dynamic6/segments.json \
  2>&1 | tee "$output/dynamic6/evaluate.log"
python3 cfd/tandem_cylinders/audit_full40_dynamic6_fno.py \
  --data "$dynamic" --checkpoint "$output/immutable_checkpoint" \
  --checkpoint-epoch "$epoch" --expected-model-sha "$model_sha" \
  --report "$output/dynamic6/evaluation.json" --segments "$output/dynamic6/segments.json" \
  --physical-qc artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json \
  --output "$output/dynamic6/diagnostic.json"

python3 - "$output" "$epoch" "$model_sha" "$base" "$dynamic" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
from pathlib import Path
out=Path(sys.argv[1]); epoch=int(sys.argv[2]); model_sha=sys.argv[3]; base=Path(sys.argv[4]); dynamic=Path(sys.argv[5])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=['validation10/evaluation.json','validation10/segments.json','validation10/diagnostic.json','dynamic6/evaluation.json','dynamic6/segments.json','dynamic6/diagnostic.json']
logs=['validation10/evaluate.log','dynamic6/evaluate.log']
for name in files+logs:
 if not (out/name).is_file(): raise SystemExit(f'missing posteval output: {name}')
payload={'status':'DYNAMIC_FNO_POSTEVAL_RESUME_COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_epoch':epoch,'model_sha256':model_sha,'dev30_manifest_sha256':sha(base/'manifest.json'),'normalization_sha256':sha(base/'normalization.json'),'dynamic6_manifest_sha256':sha(dynamic/'manifest.json'),'sha256':{name:sha(out/name) for name in files},'execution_log_sha256':{name:sha(out/name) for name in logs},'evaluation_data_mount':'/workspace/devdata','frozen_test_accessed':False,'training_performed':False}
fd,tmp=tempfile.mkstemp(dir=out); os.close(fd); p=Path(tmp); p.write_text(json.dumps(payload,indent=2)+'\n'); os.link(p,out/'receipt.json'); p.unlink()
print(json.dumps(payload,indent=2))
PY
