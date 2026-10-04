#!/usr/bin/env bash
# Default is a real CPU-loader dry-run. GPU execution requires separate approval.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute) ;; *) echo "mode must be --dry-run or --execute" >&2; exit 2;; esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="ff34a1cf402a12832ad8bedc9d4aa9c4971e6758"
wrapper_sha="576790749f731f3cfff06666efe97772ed61c5a11573ea47559aaf00f57675f8"
preflight_script_sha="e45b68810382d56b3126193d8ff159f4cca613649ab40a47a8b1e3a1708f753e"
config_sha="07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
base="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$root/data/curated/tandem_cylinders_directppo_train16_v1"
pair_manifest="$root/artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
parent="$root/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best"
config="$root/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
candidate="$root/artifacts/fcp003c_train_fit_calibration_20261005"
preflight="$root/artifacts/fcp003c_train_fit_calibration_cpu_preflight_20261005.json"
preflight_script="$root/scripts/preflight_fcp003c_train_fit_calibration.py"

sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ "$(git rev-parse "$source_commit^{commit}")" == "$source_commit" ]]
[[ "$(git show "$source_commit:scripts/run_fcp003c_train_fit_calibration.py" | sha256sum | awk '{print $1}')" == "$wrapper_sha" ]]
[[ "$(sha "$preflight_script")" == "$preflight_script_sha" ]]
[[ "$(sha "$config")" == "$config_sha" ]]
[[ "$(sha "$base/manifest.json")" == 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2 ]]
[[ "$(sha "$train8/manifest.json")" == a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35 ]]
[[ "$(sha "$train16/manifest.json")" == 7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b ]]
[[ "$(sha "$pair_manifest")" == b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c ]]
[[ "$(sha "$parent/FNO.0.2.mdlus")" == f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4 ]]
[[ "$(sha "$parent/checkpoint.0.2.pt")" == a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a ]]
[[ ! -e "$candidate" ]] || { echo "exclusive calibration output exists" >&2; exit 2; }

scratch="$(mktemp -d)"
container="fcp003c-train-fit-calibration-20261005"
cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; rm -rf -- "$scratch"; }
trap cleanup EXIT INT TERM
mkdir -p "$scratch/source"
git archive "$source_commit" src scripts | tar -x -C "$scratch/source"
cp "$config" "$scratch/config.yaml"
mounts=(
  -v "$scratch/source/src:/workspace/src:ro" -v "$scratch/source/scripts:/workspace/scripts:ro"
  -v "$scratch/config.yaml:/workspace/config.yaml:ro"
  -v "$preflight_script:/workspace/preflight.py:ro"
  -v "$base/train:/workspace/base/train:ro" -v "$base/manifest.json:/workspace/base/manifest.json:ro" -v "$base/normalization.json:/workspace/base/normalization.json:ro"
  -v "$train8/train:/workspace/train8/train:ro" -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" -v "$train8/normalization.json:/workspace/train8/normalization.json:ro"
  -v "$train16/train:/workspace/train16/train:ro" -v "$train16/manifest.json:/workspace/train16/manifest.json:ro" -v "$train16/normalization.json:/workspace/train16/normalization.json:ro"
  -v "$pair_manifest:/workspace/dynamic_pair_manifest.json:ro"
  -v "$parent/FNO.0.2.mdlus:/workspace/parent/FNO.0.2.mdlus:ro" -v "$parent/checkpoint.0.2.pt:/workspace/parent/checkpoint.0.2.pt:ro"
)
docker run --rm --network none --cpus 2 --memory 8g --shm-size 1g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
  -e PYTHONPATH=/workspace/src:/workspace/scripts "${mounts[@]}" -w /workspace "$image" \
  python /workspace/preflight.py --config /workspace/config.yaml >"$scratch/preflight.raw"
tail -n 1 "$scratch/preflight.raw" >"$scratch/preflight.json"
python3 - "$scratch/preflight.json" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "FCP003C_TRAIN_FIT_CALIBRATION_CPU_PREFLIGHT_PASS"
assert p["validation_accessed"] is p["frozen_test_accessed"] is p["gpu_used"] is False
PY
if [[ ! -e "$preflight" ]]; then cp "$scratch/preflight.json" "$preflight"; else cmp -s "$scratch/preflight.json" "$preflight"; fi
if [[ "$mode" == --dry-run ]]; then
  echo "FCP003C_TRAIN_FIT_CALIBRATION_DRY_RUN_PASS_NO_GPU"
  echo "preflight=$preflight"
  echo "preflight_sha256=$(sha "$preflight")"
  exit 0
fi

[[ "${FCP003C_CALIBRATION_EXECUTION_TOKEN:-}" == EXECUTE_APPROVED_FCP003C_CALIBRATION ]]
approval="${FCP003C_CALIBRATION_APPROVAL:-}"
[[ -f "$approval" && -n "${FCP003C_CALIBRATION_APPROVAL_SHA256:-}" && "$(sha "$approval")" == "$FCP003C_CALIBRATION_APPROVAL_SHA256" ]]
python3 - "$approval" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["gpu_execution_authorized"] is True and p["optimizer_steps"] == 128
assert p["validation_accessed"] is p["frozen_test_accessed"] is False
PY

mkdir "$candidate"
mkdir "$candidate/source_snapshot" "$candidate/immutable_parent" "$candidate/evidence"
cp -a "$scratch/source/." "$candidate/source_snapshot/"
cp "$scratch/config.yaml" "$candidate/resolved_config.input.yaml"
cp "$parent/FNO.0.2.mdlus" "$parent/checkpoint.0.2.pt" "$candidate/immutable_parent/"
cp "$preflight" "$candidate/evidence/cpu_preflight.json"
cp "$approval" "$candidate/evidence/execution_approval.json"
chmod -R a-w "$candidate/source_snapshot" "$candidate/immutable_parent"
run_output="$candidate/calibration"
command=(docker run --rm --name "$container" --gpus device=0 --network none --cpus 8 --memory 90g --shm-size 2g --pids-limit 1024
 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g
 --user "$(id -u):$(id -g)" -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
 -e PYTHONPATH=/workspace/src:/workspace/scripts "${mounts[@]}"
 -v "$candidate:/workspace/output:rw" -w /workspace "$image"
 python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .45 --margin-gib 2 --poll-seconds 2 --
 python -u scripts/run_fcp003c_train_fit_calibration.py --config /workspace/config.yaml --output /workspace/output/calibration)
set +e
timeout --signal=TERM --kill-after=30s 30m "${command[@]}" 2>&1 | tee "$candidate/run.log"
status=${PIPESTATUS[0]}
set -e
[[ $status -eq 0 ]] || exit "$status"
[[ -z "$(docker ps -aq --filter name="^${container}$")" ]]
python3 - "$candidate" "$run_output/result.json" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root,result_path=map(pathlib.Path,sys.argv[1:]); result=json.loads(result_path.read_text())
assert result["status"] == "FCP003C_TRAIN_FIT_CALIBRATION_COMPLETE_NOT_ADMISSION"
assert result["optimizer_steps"] == 128 and result["paired_steps"] == 64
assert result["validation_accessed"] is result["frozen_test_accessed"] is result["ppo_executed"] is False
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
models=list((root/"calibration/final").glob("FNO.0.*.mdlus")); states=list((root/"calibration/final").glob("checkpoint.0.*.pt"))
if len(models)!=1 or len(states)!=1: raise SystemExit("final checkpoint pair differs")
paths=[result_path,root/"calibration/resolved_config.yaml",models[0],states[0],root/"evidence/cpu_preflight.json",root/"evidence/execution_approval.json"]
payload={"status":"FCP003C_TRAIN_FIT_CALIBRATION_EXECUTION_COMPLETE_NOT_ADMISSION","optimizer_steps":128,"paired_steps":64,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False,"sha256":{str(p.relative_to(root)):sha(p) for p in paths}}
target=root/"completion_receipt.json"
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
os.link(tmp,target); tmp.unlink()
PY
echo "FCP003C_TRAIN_FIT_CALIBRATION_EXECUTION_COMPLETE_NOT_ADMISSION"
