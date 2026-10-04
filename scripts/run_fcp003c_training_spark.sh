#!/usr/bin/env bash
# Isolated FC-P003C launcher. Dry-run is the default; execution requires later evidence.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--preflight|--execute) ;; *) echo "mode must be --dry-run, --preflight, or --execute" >&2; exit 2;; esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
candidate="$root/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005"
dev30="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$root/data/curated/tandem_cylinders_directppo_train16_v1"
pair_manifest="$root/artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
parent="$root/artifacts/tandem_fno_control_train16_h100_20261004/best"
auditor="$root/scripts/audit_fcp003c_candidate.py"
config_name="tandem_fno_dynamic_paired_true_state_step_h100"

sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "pinned image differs" >&2; exit 2; }
[[ ! -e "$candidate" ]] || { echo "unique FC-P003C output already exists: $candidate" >&2; exit 2; }
python3 "$auditor" --repo "$root" >/dev/null

scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
docker run --rm -i --network none --cpus 2 --memory 4g --pids-limit 256 --cap-drop ALL \
  --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly" \
  --mount "type=bind,src=$scratch,dst=/workspace/resolved" -w /workspace "$image" \
  python - "$config_name" /workspace/resolved/resolved_config.yaml <<'PY'
import pathlib,sys
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
initialize_config_dir(version_base=None,config_dir="/workspace/conf")
cfg=compose(config_name=sys.argv[1])
pathlib.Path(sys.argv[2]).write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding="utf-8")
PY
python3 "$auditor" --repo "$root" --resolved-config "$scratch/resolved_config.yaml" >"$scratch/preflight.json"

if [[ "$mode" == --dry-run ]]; then
  echo "FC_P003C_TRAINING_DRY_RUN_PASS_NO_GPU"
  echo "candidate=$candidate"
  echo "resolved_config_sha256=$(sha "$scratch/resolved_config.yaml")"
  echo "execution_blocked_until=mixed_probe_pass_and_separate_lead_approval"
  exit 0
fi

mixed="${FCP003C_MIXED_PROBE_RECEIPT:-$root/artifacts/fcp003c_mixed_loss_technical_probe_20261005/completion_receipt.json}"
mixed_result="${FCP003C_MIXED_PROBE_RESULT:-$root/artifacts/fcp003c_mixed_loss_technical_probe_20261005/container_output/result_bundle/result.json}"
[[ -f "$mixed" && -n "${FCP003C_MIXED_PROBE_RECEIPT_SHA256:-}" && "$(sha "$mixed")" == "$FCP003C_MIXED_PROBE_RECEIPT_SHA256" ]] || { echo "reviewed mixed-probe receipt required" >&2; exit 2; }
[[ -f "$mixed_result" && -n "${FCP003C_MIXED_PROBE_RESULT_SHA256:-}" && "$(sha "$mixed_result")" == "$FCP003C_MIXED_PROBE_RESULT_SHA256" ]] || { echo "reviewed mixed-probe result required" >&2; exit 2; }
python3 - "$mixed" "$mixed_result" <<'PY'
import hashlib,json,pathlib,sys
mixed=json.loads(pathlib.Path(sys.argv[1]).read_text()); result_path=pathlib.Path(sys.argv[2]); result=json.loads(result_path.read_text())
if mixed.get("status") != "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE" or mixed.get("optimizer_steps") != 1 or mixed.get("candidate_saved") is not False or mixed.get("validation_or_frozen_accessed") is not False:
 raise SystemExit("mixed-probe contract differs")
if mixed.get("result_sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest(): raise SystemExit("mixed-probe result binding differs")
if result.get("status") != "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS" or result.get("optimizer_steps") != 1 or result.get("candidate_saved") is not False or result.get("validation_or_frozen_accessed") is not False: raise SystemExit("mixed-probe result contract differs")
parent={"model":"8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240","state":"1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"}
if result.get("parent_file_sha256_before") != parent or result.get("parent_file_sha256_after") != parent or mixed.get("checkpoint_model_sha256") != parent["model"] or mixed.get("checkpoint_state_sha256") != parent["state"]: raise SystemExit("mixed-probe parent binding differs")
if result.get("config_sha256") != "ebcc839f302cf3eb93eff2a23f9d8ba38f1063e09591668dad3d65d31bc91f4f": raise SystemExit("mixed-probe resolved config differs")
PY
if [[ "$mode" == --preflight ]]; then
  echo "FC_P003C_MIXED_PROBE_EVIDENCE_PREFLIGHT_PASS_NO_GPU"
  echo "mixed_completion_sha256=$(sha "$mixed")"
  echo "mixed_result_sha256=$(sha "$mixed_result")"
  echo "full_training_still_requires_separate_lead_approval=true"
  exit 0
fi

[[ "${FCP003C_EXECUTION_TOKEN:-}" == EXECUTE_APPROVED_FC_P003C ]] || { echo "FC-P003C execution token required" >&2; exit 2; }
approval="${FCP003C_FULL_APPROVAL:-}"
[[ -f "$approval" && -n "${FCP003C_FULL_APPROVAL_SHA256:-}" && "$(sha "$approval")" == "$FCP003C_FULL_APPROVAL_SHA256" ]] || { echo "separate full-training approval required" >&2; exit 2; }
python3 - "$approval" <<'PY'
import json,pathlib,sys
approval=json.loads(pathlib.Path(sys.argv[1]).read_text())
if approval.get("status") != "FC_P003C_FULL_TRAINING_APPROVED" or approval.get("gpu_execution_authorized") is not True:
 raise SystemExit("full-training approval contract differs")
PY

mkdir -p "$candidate/source_snapshot" "$candidate/immutable_parent" "$candidate/launch_evidence"
commit="$(git rev-parse HEAD)"; tree="$(git rev-parse "$commit^{tree}")"
git archive "$commit" src scripts conf cfd/tandem_cylinders/audit_full40_dynamic6_fno.py | tar -x -C "$candidate/source_snapshot"
cp -a "$parent/FNO.0.2.mdlus" "$parent/checkpoint.0.2.pt" "$candidate/immutable_parent/"
cp "$scratch/resolved_config.yaml" "$candidate/preflight_resolved_config.yaml"
cp "$mixed" "$candidate/launch_evidence/mixed_probe_completion_receipt.json"
cp "$mixed_result" "$candidate/launch_evidence/mixed_probe_result.json"
cp "$approval" "$candidate/launch_evidence/full_training_approval.json"
chmod -R a-w "$candidate/source_snapshot" "$candidate/immutable_parent"

python3 - "$candidate/launch_receipt.json" "$candidate" "$commit" "$tree" "$image_id" "$mixed" "$mixed_result" "$approval" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target,candidate=map(pathlib.Path,sys.argv[1:3]); commit,tree,image=sys.argv[3:6]; mixed,mixed_result,approval=map(pathlib.Path,sys.argv[6:])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
 return h.hexdigest()
snapshot=candidate/"source_snapshot"
payload={"status":"FC_P003C_LAUNCH_STAGED","mode":"--execute","git_commit":commit,"git_tree":tree,
 "official_image_id":image,"approval_sha256":"e0733bcceb131018089584f1daed5193e3ac6aeff6d933c8ef32ca3b0a8cdea4",
 "full_execution_approval_sha256":sha(approval),"mixed_probe_receipt_sha256":sha(mixed),"mixed_probe_result_sha256":sha(mixed_result),
 "parent_model_sha256":"8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240",
 "parent_state_sha256":"1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0",
 "dynamic_pair_manifest_sha256":"b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c",
 "real_sampling_receipt_sha256":"da078a1c43f03bf86ee71a5010632c05a4868b169dc8f8c7c341c78134873242",
 "baseline_order_receipt_sha256":"fab043a70652475e0b03aa869eac3445eaec1ec6c66a74cf976a0342a3dbca88",
 "resolved_config_sha256":sha(candidate/"preflight_resolved_config.yaml"),
 "source_snapshot_sha256":{str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()},
 "single_factor":"paired_objective_paired_statistics_to_true_state_step_force",
 "paired_dataset_kind":"dynamic8","paired_dataset_repetitions":2,"paired_updates_per_epoch":16,
 "ppo_auto_launch":False,"frozen_test_accessed":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY

source="$candidate/source_snapshot"
command=(docker run --rm --name fcp003c-true-state-step-full-20261005 --gpus device=0 --read-only
 --network none --cpus 8 --memory 90g --shm-size 2g --pids-limit 1024 --cap-drop ALL
 --security-opt no-new-privileges --user "$(id -u):$(id -g)" -e "USER=$(id -un)"
 -e "LOGNAME=$(id -un)" -e HOME=/tmp -e PYTHONPATH=/workspace/src:/workspace/scripts
 --tmpfs /tmp:rw,nosuid,nodev,size=4g -v "$source/src:/workspace/src:ro"
 -v "$source/scripts:/workspace/scripts:ro" -v "$source/conf:/workspace/conf:ro"
 -v "$dev30:/workspace/base:ro" -v "$train8:/workspace/train8:ro" -v "$train16:/workspace/train16:ro"
 -v "$pair_manifest:/workspace/dynamic_pair_manifest.json:ro"
 -v "$candidate/immutable_parent:/workspace/parent:ro" -v "$candidate:/workspace/output:rw"
 -w /workspace "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20
 --allocator-fraction .45 --margin-gib 2 --poll-seconds 2 -- python -u
 scripts/train_tandem_fno_paired_stats.py --config-name "$config_name"
 hydra.run.dir=/tmp/hydra hydra.output_subdir=null output_dir=/workspace/output)
"${command[@]}" 2>&1 | tee "$candidate/train.log"
cmp -s "$candidate/preflight_resolved_config.yaml" "$candidate/resolved_config.yaml" || { echo "runtime resolved config differs from preflight" >&2; exit 3; }

python3 - "$candidate" <<'PY'
import hashlib,json,math,os,pathlib,tempfile,sys
root=pathlib.Path(sys.argv[1])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
 return h.hexdigest()
history=json.loads((root/"training_history.json").read_text())
if len(history)!=2: raise SystemExit("training did not complete two epochs")
selected=min(history,key=lambda row:row["selection_score"]); epoch=int(selected["epoch"])
paths=[root/name for name in ("launch_receipt.json","resolved_config.yaml","runtime_metadata.json","training_data_sources.json","training_history.json","train.log")]
paths += [root/f"best/FNO.0.{epoch}.mdlus",root/f"best/checkpoint.0.{epoch}.pt",root/f"checkpoints/FNO.0.{epoch}.mdlus",root/f"checkpoints/checkpoint.0.{epoch}.pt"]
if any(not path.is_file() for path in paths): raise SystemExit("training outputs incomplete")
payload={"status":"FC_P003C_TRAINING_COMPLETE","epochs":2,"checkpoint_epoch":epoch,
 "sha256":{str(path.relative_to(root)):sha(path) for path in paths},"ppo_auto_launched":False,"frozen_test_accessed":False}
target=root/"completion_receipt.json"
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
python3 "$source/scripts/audit_fcp003c_candidate.py" --repo "$root" --candidate "$candidate" --output "$candidate/lineage.final.json"
echo FC_P003C_TRAINING_COMPLETE
