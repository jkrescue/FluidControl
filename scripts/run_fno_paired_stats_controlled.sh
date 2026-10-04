#!/usr/bin/env bash
# Strict single-factor lambda=0 versus lambda=10 paired-stat FNO training.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
branch="${2:-lambda0}"
case "$mode" in --dry-run|--probe|--execute) ;; *) echo "invalid mode" >&2; exit 2 ;; esac
case "$branch" in lambda0) pair_weight=0.0 ;; lambda10) pair_weight=10.0 ;; *) echo "branch must be lambda0 or lambda10" >&2; exit 2 ;; esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
dev30="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$root/data/curated/tandem_cylinders_directppo_train16_v1"
pair_artifact="$root/artifacts/train20_paired_stat_datapipe_v1"
pair_manifest="$pair_artifact/manifest.json"
pair_probe="$pair_artifact/cpu_probe_v3.json"
scale_audit="$root/artifacts/tandem_fno_paired_stats_parent_scale_audit_20261004/parent_scale_audit.json"
parent_candidate="$root/artifacts/tandem_fno_control_train16_h100_20261004"
parent="$parent_candidate/best"
development_gate="$parent_candidate/posteval_complete_v2/development_gate.json"
pair_manifest_sha="15bfa7a47e3195afad59884f96fd4e305c8ba0001dd6eb043be2412b2b9ce2b7"
pair_probe_sha="8453a2836a14fb271ad9fdfe3205915d15b6414ade916434a5713e068a185ec4"
scale_audit_sha="8feda2a4ea86c9dabf9d7e8d4d8c8bf4a1a967123ab0dd790b9b7893aa059296"
parent_model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
parent_state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"

sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image differs" >&2; exit 2; }
[[ "$(sha "$pair_manifest")" == "$pair_manifest_sha" ]] || { echo "pair manifest differs" >&2; exit 2; }
[[ "$(sha "$pair_probe")" == "$pair_probe_sha" ]] || { echo "pair probe differs" >&2; exit 2; }
[[ "$(sha "$scale_audit")" == "$scale_audit_sha" ]] || { echo "parent scale audit differs" >&2; exit 2; }
[[ "$(sha "$parent/FNO.0.2.mdlus")" == "$parent_model_sha" ]] || { echo "parent model differs" >&2; exit 2; }
[[ "$(sha "$parent/checkpoint.0.2.pt")" == "$parent_state_sha" ]] || { echo "parent state differs" >&2; exit 2; }
for path in "$dev30" "$train8" "$train16"; do [[ -d "$path" ]] || { echo "missing data: $path" >&2; exit 2; }; done
python3 - "$pair_probe" "$development_gate" "$scale_audit" <<'PY'
import json, pathlib, sys
probe=json.loads(pathlib.Path(sys.argv[1]).read_text())
if probe.get("status") != "TRAIN20_MATCHED_PAIR_OFFICIAL_DATAPIPE_CPU_PROBE_PASS": raise SystemExit("pair probe status differs")
if probe.get("pair_count") != 16 or probe.get("validation_or_frozen_accessed") is not False: raise SystemExit("pair probe contract differs")
gate=json.loads(pathlib.Path(sys.argv[2]).read_text())
if gate.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL" or gate.get("ppo_authorized") is not False: raise SystemExit("parent is not the recorded failed development candidate")
audit=json.loads(pathlib.Path(sys.argv[3]).read_text())
if audit.get("status") != "TRAIN20_PAIRED_STAT_PARENT_SCALE_AUDIT_COMPLETE" or audit.get("training_executed") is not False or audit.get("validation_or_frozen_accessed") is not False: raise SystemExit("parent scale audit contract differs")
if abs(float(audit.get("normalized_mse",-1))-0.005348736377337855) > 1e-15: raise SystemExit("parent scale differs")
PY

if [[ "$mode" == "--dry-run" ]]; then
  printf 'PAIRED_STATS_CONTROLLED_READY_NO_GPU branch=%s lambda=%s\n' "$branch" "$pair_weight"
  exit 0
fi
[[ "${PAIRED_STATS_TRAIN_TOKEN:-}" == "EXECUTE_REVIEWED_PAIRED_STATS" ]] || { echo "reviewed token required" >&2; exit 2; }

suffix=""
[[ "$mode" == "--probe" ]] && suffix="_probe_v3"
output="$root/artifacts/tandem_fno_paired_stats_${branch}${suffix}_20261004"
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 2; }
commit="$(git rev-parse HEAD)"
tree="$(git rev-parse "$commit^{tree}")"
mkdir -p "$output/source_snapshot" "$output/immutable_parent"
git archive "$commit" src scripts conf | tar -x -C "$output/source_snapshot"
cp -a "$parent/FNO.0.2.mdlus" "$parent/checkpoint.0.2.pt" "$output/immutable_parent/"
chmod -R a-w "$output/source_snapshot" "$output/immutable_parent"
python3 - "$output/launch_receipt.json" "$branch" "$pair_weight" "$mode" "$commit" "$tree" "$pair_manifest" "$pair_probe" "$scale_audit" "$output" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); branch,weight,mode,commit,tree=sys.argv[2:7]
manifest,probe,scale,output=map(pathlib.Path,sys.argv[7:11])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
snapshot=output/"source_snapshot"
files=[p for base in (snapshot/"src",snapshot/"scripts",snapshot/"conf") for p in base.rglob("*") if p.is_file()]
payload={"status":"PAIRED_STATS_CONTROLLED_LAUNCH_STAGED","branch":branch,
 "paired_stat_loss_weight":float(weight),"mode":mode,"git_commit":commit,"git_tree":tree,
 "pair_manifest_sha256":sha(manifest),"pair_probe_sha256":sha(probe),
 "parent_scale_audit_sha256":sha(scale),
 "parent_model_sha256":sha(output/"immutable_parent/FNO.0.2.mdlus"),
 "parent_state_sha256":sha(output/"immutable_parent/checkpoint.0.2.pt"),
 "source_snapshot_sha256":{str(p.relative_to(snapshot)):sha(p) for p in sorted(files)},
 "single_factor":"paired_stat_loss_weight_0_vs_10","validation_or_frozen_used_to_choose_lambda":False,
 "ppo_auto_launch":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.link(tmp,target); tmp.unlink()
PY

extra=()
if [[ "$mode" == "--probe" ]]; then
  extra=(training.epochs=1 training.max_train_batches=1 training.max_validation_batches=1 training.paired_batches_per_epoch=1 training.max_paired_eval_batches=1)
fi
source="$output/source_snapshot"
command=(docker run --rm --name "paired-stats-${branch}-${mode#--}-20261004" --gpus device=0
 --cpus 8 --memory 90g --shm-size 2g --pids-limit 1024 --cap-drop ALL
 --security-opt no-new-privileges --user "$(id -u):$(id -g)"
 -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e HOME=/tmp
 -e PYTHONPATH=/workspace/src:/workspace/scripts --tmpfs /tmp:rw,nosuid,nodev,size=4g
 -v "$source/src:/workspace/src:ro" -v "$source/scripts:/workspace/scripts:ro"
 -v "$source/conf:/workspace/conf:ro" -v "$dev30:/workspace/base:ro"
 -v "$train8:/workspace/train8:ro" -v "$train16:/workspace/train16:ro"
 -v "$pair_manifest:/workspace/pair_manifest.json:ro"
 -v "$output/immutable_parent:/workspace/parent:ro" -v "$output:/workspace/output:rw"
 -w /workspace "$image" python scripts/train_tandem_fno_paired_stats.py
 hydra.run.dir=/tmp/hydra hydra.output_subdir=null
 "training.paired_stat_loss_weight=$pair_weight" "${extra[@]}")
python3 "$source/scripts/spark_gpu_guard.py" --min-free-gib 20 --allocator-fraction 0.45 --margin-gib 2 --poll-seconds 2 -- "${command[@]}" 2>&1 | tee "$output/train.log"
status="PAIRED_STATS_CONTROLLED_PROBE_PASS"
[[ "$mode" == "--execute" ]] && status="PAIRED_STATS_CONTROLLED_TRAINING_COMPLETE"
python3 - "$output/completion_receipt.json" "$status" "$output" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); status=sys.argv[2]; root=pathlib.Path(sys.argv[3])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
required=[root/"launch_receipt.json",root/"resolved_config.yaml",root/"runtime_metadata.json",root/"training_history.json",root/"train.log"]
if any(not p.is_file() for p in required): raise SystemExit("training outputs incomplete")
history=json.loads((root/"training_history.json").read_text())
if not history or any(not isinstance(row.get("train_only_paired_stat_loss"),(int,float)) for row in history): raise SystemExit("paired training history incomplete")
payload={"status":status,"sha256":{p.name:sha(p) for p in required},"epochs":len(history),"ppo_auto_launched":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.link(tmp,target); tmp.unlink()
PY
