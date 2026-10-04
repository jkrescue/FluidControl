#!/usr/bin/env bash
# Worker-only immutable launcher for Lead-approved FC-P003B.
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in --dry-run|--probe|--execute) ;; *) echo "invalid mode" >&2; exit 2;; esac
root="${FCP003B_ROOT:-/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005}"
source="$root/source_snapshot"
receipt="$root/source_receipt.json"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="44b7c6b172cc479657cc959a32b5ff0b7d516d2a"
approval_sha="b042a8570aec4802f1963392bb90368a2f0428c59d8f7afff76bbfa0417611d8"
manifest_sha="b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
probe_sha="e549121553e4e1fa0fffe5c0f3080dc1f03db615342530a7a7fcb0a0e45ac7aa"
parent_model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
parent_state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"

sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image differs" >&2; exit 2; }
[[ -d "$source" && -f "$receipt" ]] || { echo "immutable source snapshot missing" >&2; exit 2; }
[[ "$(sha "$root/evidence/FC-P003B_APPROVAL.md")" == "$approval_sha" ]] || { echo "approval differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/dynamic_pair_manifest.json")" == "$manifest_sha" ]] || { echo "dynamic pair manifest differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/cpu_probe.json")" == "$probe_sha" ]] || { echo "CPU probe differs" >&2; exit 2; }
[[ "$(sha "$root/immutable_parent/FNO.0.2.mdlus")" == "$parent_model_sha" ]] || { echo "parent model differs" >&2; exit 2; }
[[ "$(sha "$root/immutable_parent/checkpoint.0.2.pt")" == "$parent_state_sha" ]] || { echo "parent state differs" >&2; exit 2; }
for path in "$root/data/dev30" "$root/data/train8" "$root/data/train16"; do
  [[ "$(sha "$path/normalization.json")" == "$normalization_sha" ]] || { echo "normalization differs: $path" >&2; exit 2; }
done
python3 - "$receipt" "$source_commit" <<'PY'
import json,pathlib,sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
if d.get("status") != "FC_P003B_IMMUTABLE_SOURCE_STAGED" or d.get("git_commit") != sys.argv[2]:
    raise SystemExit("source receipt differs")
PY
declare -A expected=(
 [scripts/train_tandem_fno_paired_stats.py]=01c94e104541a9e75e8b85a9049264b18f26c9554bb3f0393d892a385119ac44
 [src/fluid_control/paired_training.py]=5d96142f815df0206451f066a47a2170b1e5b197c5a76ec3e7bc6a7044f32ad5
 [conf/tandem_fno_dynamic_paired_interleaved_h100.yaml]=a8e4033240af24ec6ee0c537059244bda765800c239c292b6b18085a42c9ffc9
 [src/fluid_control/dynamic_pair_stat_datapipe.py]=2414b0a2b2635f87e0842ed12d36ad7a3eb6b1b8759cacdb5b8c8fa9fed5af31
 [scripts/spark_gpu_guard.py]=75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195
)
for relative in "${!expected[@]}"; do
  [[ "$(sha "$source/$relative")" == "${expected[$relative]}" ]] || { echo "source differs: $relative" >&2; exit 2; }
done

if [[ "$mode" == --dry-run ]]; then echo "FC_P003B_WORKER_READY_NO_GPU"; exit 0; fi
[[ "${FCP003B_EXECUTION_TOKEN:-}" == EXECUTE_REVIEWED_FC_P003B ]] || { echo "reviewed token required" >&2; exit 2; }
if [[ "$mode" == --probe ]]; then
  output="$root/probe_v1"; container="fcp003b-dynamic-pairs-probe"
  extra=(training.epochs=1 training.max_train_batches=8 training.max_validation_batches=1 training.expected_regular_batches=8 training.paired_dataset_repetitions=1 training.paired_batches_per_epoch=8 training.max_paired_eval_batches=1)
else
  output="$root/output"; container="fcp003b-dynamic-pairs-full"; extra=()
fi
[[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; exit 2; }
mkdir -p "$output"
python3 - "$output/launch_receipt.json" "$mode" "$source_commit" "$approval_sha" "$manifest_sha" "$probe_sha" <<'PY'
import json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); mode,commit,approval,manifest,probe=sys.argv[2:]
payload={"status":"FC_P003B_WORKER_LAUNCH_STAGED","mode":mode,"git_commit":commit,
 "approval_sha256":approval,"dynamic_pair_manifest_sha256":manifest,"cpu_probe_sha256":probe,
 "single_factor":"paired_supervision_content_static16_vs_dynamic8_repeated_twice",
 "paired_dataset_kind":"dynamic8","unique_pair_count":8,"paired_updates_per_epoch":16,
 "paired_dataset_repetitions":2 if mode=="--execute" else 1,
 "validation_or_frozen_training_accessed":False,"ppo_auto_launch":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.link(tmp,target); tmp.unlink()
PY
# ``device=0`` is a single Docker --gpus argument, not comma-separated shell items.
# shellcheck disable=SC2054
command=(docker run --rm --name "$container" --gpus device=0 --cpus 8 --memory 90g
 --shm-size 2g --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges
 --user "$(id -u):$(id -g)" -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e HOME=/tmp
 -e PYTHONPATH=/workspace/src:/workspace/scripts --tmpfs /tmp:rw,nosuid,nodev,size=4g
 -v "$source/src:/workspace/src:ro" -v "$source/scripts:/workspace/scripts:ro"
 -v "$source/conf:/workspace/conf:ro" -v "$root/data/dev30:/workspace/base:ro"
 -v "$root/data/train8:/workspace/train8:ro" -v "$root/data/train16:/workspace/train16:ro"
 -v "$root/evidence/dynamic_pair_manifest.json:/workspace/dynamic_pair_manifest.json:ro"
 -v "$root/immutable_parent:/workspace/parent:ro" -v "$output:/workspace/output:rw"
 -w /workspace "$image" python scripts/train_tandem_fno_paired_stats.py
 --config-name tandem_fno_dynamic_paired_interleaved_h100
 hydra.run.dir=/tmp/hydra hydra.output_subdir=null output_dir=/workspace/output
 "${extra[@]}")
python3 "$source/scripts/spark_gpu_guard.py" --min-free-gib 20 --allocator-fraction 0.45 --margin-gib 2 --poll-seconds 2 -- "${command[@]}" 2>&1 | tee "$output/train.log"
python3 - "$output/completion_receipt.json" "$output" "$mode" <<'PY'
import collections,hashlib,json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); mode=sys.argv[3]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
required=[root/name for name in ("launch_receipt.json","resolved_config.yaml","runtime_metadata.json","training_history.json","train.log")]
if any(not path.is_file() for path in required): raise SystemExit("FC-P003B outputs incomplete")
history=json.loads((root/"training_history.json").read_text())
expected_epochs=1 if mode=="--probe" else 2; repetitions=1 if mode=="--probe" else 2
indices=list(range(8)) if mode=="--probe" else [i*1367//15 for i in range(16)]
identities={f"b{phase:02d}:{profile}" for phase in (0,2,4,6) for profile in ("multisine","prbs")}
if len(history)!=expected_epochs: raise SystemExit("epoch count differs")
for row in history:
 if row.get("paired_dataset_kind")!="dynamic8" or row.get("paired_batch_indices")!=indices: raise SystemExit("dynamic schedule differs")
 passes=row.get("paired_identity_passes")
 if not isinstance(passes,list) or len(passes)!=repetitions: raise SystemExit("paired pass count differs")
 if any(len(x)!=8 or set(x)!=identities for x in passes): raise SystemExit("paired pass coverage differs")
 if collections.Counter(row.get("paired_identities",[]))!=collections.Counter({key:repetitions for key in identities}): raise SystemExit("paired multiplicity differs")
status="FC_P003B_DYNAMIC_PAIR_PROBE_PASS" if mode=="--probe" else "FC_P003B_DYNAMIC_PAIR_TRAINING_COMPLETE"
payload={"status":status,"training_exit_code":0,"epochs":len(history),"sha256":{p.name:sha(p) for p in required},"frozen_test_accessed":False,"ppo_auto_launched":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.link(tmp,target); tmp.unlink()
PY
