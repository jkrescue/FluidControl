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
base="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
predecl="$root/artifacts/tandem_cylinders/dynamic_train8_fno_finetune_predeclared_20261003.json"
case "$horizon" in
  h20) config="tandem_fno_dynamic_train8_h20" ;;
  h50) config="tandem_fno_dynamic_train8_h50" ;;
  *) echo "explicit parent horizon must be h20 or h50" >&2; exit 2 ;;
esac

[[ -n "$parent" && -d "$parent" ]] || { echo "explicit completed parent directory required" >&2; exit 2; }
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || { echo "valid run-id required" >&2; exit 2; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] \
  || { echo "PhysicsNeMo image ID mismatch" >&2; exit 2; }

python3 - "$base" "$train8" "$predecl" "$parent" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
base, extra, predecl, parent = map(Path, sys.argv[1:])
p=json.loads(predecl.read_text())
if p.get("status") != "DYNAMIC_TRAIN8_FNO_FINETUNE_PREDECLARED_NOT_EXECUTED" or p["data"]["frozen_test_access"] is not False:
    raise SystemExit("training predeclaration differs")
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
PY

output="$root/artifacts/tandem_fno_dynamic_train8_h20_${run_id}"
[[ ! -e "$output" ]] || { echo "refusing existing output" >&2; exit 2; }
if [[ "$mode" == "--dry-run" ]]; then
  echo "DYNAMIC_TRAIN8_FNO_FINETUNE_DRY_RUN_READY parent=$parent horizon=$horizon output=$output"
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "mode must be --dry-run or --execute" >&2; exit 2; }
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

exec docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g \
  --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly" \
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly" \
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly" \
  --mount "type=bind,src=$base,dst=/workspace/base,readonly" \
  --mount "type=bind,src=$train8,dst=/workspace/train8,readonly" \
  --mount "type=bind,src=$parent_copy,dst=/workspace/parent,readonly" \
  --mount "type=bind,src=$output,dst=/workspace/output" \
  --workdir /workspace "$image" python -u scripts/spark_gpu_guard.py \
  --min-free-gib 20 --allocator-fraction 0.25 --margin-gib 4 -- \
  python -u scripts/train_tandem_fno_rollout.py --config-name "$config" \
  hydra.run.dir=/tmp/hydra hydra.output_subdir=null
