#!/usr/bin/env bash
# Dry-run-first train8-augmented official PhysicsNeMo FNO fine-tuning.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
parent="${2:-}"
run_id="${3:-}"
horizon="${4:-}"
token="${DYNAMIC_TRAIN8_FNO_APPROVAL_TOKEN:-}"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
base="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
predecl="$root/artifacts/tandem_cylinders/dynamic_train8_fno_finetune_predeclared_20261003.json"
case "$horizon" in
  h20) config="tandem_fno_dynamic_train8_h20"; fraction=.25 ;;
  h50) config="tandem_fno_dynamic_train8_h50"; fraction=.45 ;;
  h100) config="tandem_fno_dynamic_train8_h100"; fraction=.45 ;;
  *) echo "explicit parent horizon must be h20, h50, or h100" >&2; exit 2 ;;
esac

[[ -n "$parent" && -d "$parent" ]] || { echo "explicit completed parent directory required" >&2; exit 2; }
[[ "$(basename "$parent")" != best ]] || { echo "mutable best parent directory is forbidden" >&2; exit 2; }
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || { echo "valid run-id required" >&2; exit 2; }
case "$mode" in --dry-run|--probe|--execute) ;; *) echo "mode must be --dry-run, --probe, or --execute" >&2; exit 2 ;; esac
[[ -f "$train8/manifest.json" ]] || { echo "train8 final manifest is not ready" >&2; exit 3; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] \
  || { echo "PhysicsNeMo image ID mismatch" >&2; exit 2; }

python3 - "$base" "$train8" "$predecl" "$parent" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
base, extra, predecl, parent = map(Path, sys.argv[1:])
p=json.loads(predecl.read_text())
if p.get("status") != "DYNAMIC_TRAIN8_FNO_FINETUNE_PREDECLARED_NOT_EXECUTED" or p["data"]["frozen_test_access"] is not False:
    raise SystemExit("training predeclaration differs")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()
if sha(base/"manifest.json") != "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2":
    raise SystemExit("immutable dev30 manifest differs")
if sha(base/"normalization.json") != "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1":
    raise SystemExit("immutable dev30 normalization differs")
if (base/"frozen_test").exists() or len(list((base/"train").glob("*.h5"))) != 20 or len(list((base/"validation").glob("*.h5"))) != 10:
    raise SystemExit("immutable dev30 split contract differs")
m=json.loads((extra/"manifest.json").read_text())
if m.get("status") != "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED" or m.get("trajectory_counts") != {"train":8,"validation":0,"frozen_test":0}:
    raise SystemExit("train8 dataset is not finalized train-only")
if (extra/"normalization.json").read_bytes() != (base/"normalization.json").read_bytes():
    raise SystemExit("normalization differs from train20")
models=sorted(parent.glob("FNO.0.*.mdlus")); states=sorted(parent.glob("checkpoint.0.*.pt"))
if len(models) != 1 or len(states) != 1:
    raise SystemExit("parent must contain exactly one model/checkpoint pair")
epoch=lambda x: re.fullmatch(r"(?:FNO|checkpoint)\.0\.(\d+)\.(?:mdlus|pt)",x.name).group(1)
if epoch(models[0]) != epoch(states[0]):
    raise SystemExit("parent epochs differ")
receipt=parent/"parent_receipt.json"
if receipt.exists():
    payload=json.loads(receipt.read_text())
    recorded=payload.get("sha256", {})
    immutable=payload.get("immutable_parent", {})
    if not recorded and immutable:
        recorded={
            immutable.get("model", ""): immutable.get("model_sha256"),
            immutable.get("state", ""): immutable.get("state_sha256"),
        }
    for item in (models[0], states[0]):
        if recorded.get(item.name) != sha(item):
            raise SystemExit("parent receipt SHA differs")
PY

output="$root/artifacts/tandem_fno_dynamic_train8_${horizon}_${run_id}"
[[ ! -e "$output" ]] || { echo "refusing existing output" >&2; exit 2; }
if [[ "$mode" == "--dry-run" ]]; then
  echo "DYNAMIC_TRAIN8_FNO_FINETUNE_DRY_RUN_READY parent=$parent horizon=$horizon fraction=$fraction output=$output"
  exit 0
fi
[[ "$token" == "EXECUTE_REVIEWED_DYNAMIC_TRAIN8_FNO_FINETUNE" ]] \
  || { echo "explicit training token required" >&2; exit 2; }

mkdir "$output"
parent_copy="$output/immutable_parent"
mkdir "$parent_copy"
cp --reflink=auto "$parent"/FNO.0.*.mdlus "$parent"/checkpoint.0.*.pt "$parent_copy"/
python3 - "$parent" "$parent_copy" "$output/parent_receipt.json" <<'PY'
import hashlib,json,os,sys,tempfile
from pathlib import Path
source,target,receipt=map(Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
rows={}
for copied in sorted(target.iterdir()):
 original=source/copied.name
 if sha(original)!=sha(copied): raise SystemExit('immutable parent copy SHA differs')
 copied.chmod(0o444); rows[copied.name]=sha(copied)
payload={'status':'IMMUTABLE_PARENT_COPIED','source':str(source.resolve()),'sha256':rows}
receipt.write_text(json.dumps(payload,indent=2)+'\n')
PY

base_cmd=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 90g \
  --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1 \
  --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" \
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly" \
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly" \
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly" \
  --mount "type=bind,src=$base,dst=/workspace/base,readonly" \
  --mount "type=bind,src=$train8,dst=/workspace/train8,readonly" \
  --mount "type=bind,src=$parent_copy,dst=/workspace/parent,readonly" \
  --mount "type=bind,src=$output,dst=/workspace/output" \
  --workdir /workspace)
train_cmd=("${base_cmd[@]}" "$image" python -u scripts/spark_gpu_guard.py \
  --min-free-gib 20 --allocator-fraction "$fraction" --margin-gib 4 -- \
  python -u scripts/train_tandem_fno_rollout.py --config-name "$config" \
  data.prefetch_factor=0 hydra.run.dir=/tmp/hydra hydra.output_subdir=null)
if [[ "$mode" == --probe ]]; then
  train_cmd+=(training.epochs=1 training.max_train_batches=1 training.max_validation_batches=1)
fi
"${train_cmd[@]}" 2>&1 | tee "$output/train.log"
[[ "$mode" == --execute ]] || exit 0

# Training has released the GPU. Independently evaluate the unchanged
# validation10 and dynamic6 validation-only suites; frozen data is unavailable.
best="$output/best"
read -r epoch model_sha < <(python3 - "$best" <<'PY'
import hashlib,re,sys
from pathlib import Path
p=Path(sys.argv[1]); models=list(p.glob("FNO.0.*.mdlus"))
if len(models)!=1: raise SystemExit("best must contain exactly one model")
m=re.fullmatch(r"FNO\.0\.(\d+)\.mdlus",models[0].name)
if not m: raise SystemExit("unexpected best model name")
h=hashlib.sha256(models[0].read_bytes()).hexdigest()
print(m.group(1),h)
PY
)
mkdir "$output/validation10" "$output/dynamic6"
"${base_cmd[@]}" --mount "type=bind,src=$best,dst=/workspace/checkpoint,readonly" \
  "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py --data /workspace/base --normalization-data /workspace/base \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint \
  --split validation --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4 \
  --action-mode observed --visualizations-per-horizon 0 --output /workspace/output/validation10/evaluation.json \
  --segment-metrics-output /workspace/output/validation10/segments.json 2>&1 | tee "$output/validation10/evaluate.log"
python3 scripts/audit_dev30_validation_diagnostic.py \
  --report "$output/validation10/evaluation.json" --segments "$output/validation10/segments.json" \
  --data "$base" --checkpoint-dir "$best" --candidate-kind dev30_free_ar_development \
  --output "$output/validation10/diagnostic.json"

dynamic="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
"${base_cmd[@]}" --mount "type=bind,src=$dynamic,dst=/workspace/dynamic,readonly" \
  --mount "type=bind,src=$best,dst=/workspace/checkpoint,readonly" \
  "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/base \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint \
  --split validation --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 8 \
  --action-mode observed --visualizations-per-horizon 0 --output /workspace/output/dynamic6/evaluation.json \
  --segment-metrics-output /workspace/output/dynamic6/segments.json 2>&1 | tee "$output/dynamic6/evaluate.log"
python3 cfd/tandem_cylinders/audit_full40_dynamic6_fno.py \
  --data "$dynamic" --checkpoint "$best" --checkpoint-epoch "$epoch" --expected-model-sha "$model_sha" \
  --report "$output/dynamic6/evaluation.json" --segments "$output/dynamic6/segments.json" \
  --physical-qc artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json \
  --output "$output/dynamic6/diagnostic.json"
