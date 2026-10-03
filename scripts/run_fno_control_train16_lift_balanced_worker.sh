#!/usr/bin/env bash
# Worker-only controlled force-weight comparison; never accesses frozen data.
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in
  --dry-run|--probe|--execute) ;;
  *) echo "mode must be --dry-run, --probe, or --execute" >&2; exit 2 ;;
esac

source_root="${CONTROL_SOURCE_ROOT:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/source_snapshot}"
output="${CONTROL_OUTPUT:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/output}"
probe_output="${CONTROL_PROBE_OUTPUT:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/probe_output}"
source_receipt="${CONTROL_SOURCE_RECEIPT:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/source_receipt.json}"
base="${CONTROL_BASE_DATA:-/home/USER/workspace/fluid_control_free_ar_20261003/project/data/curated/tandem_cylinders_matched_start_full40_dev30_v1}"
train8="${CONTROL_TRAIN8_DATA:-/home/USER/workspace/fluid_control_dynamic_h100_20261004/data/tandem_cylinders_dynamic_train8_v1}"
train16="${CONTROL_TRAIN16_DATA:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/data/tandem_cylinders_directppo_train16_v1}"
parent="${CONTROL_PARENT:-/home/USER/workspace/fluid_control_dynamic_h100_20261004/output/best}"
datapipe_probe="${CONTROL_DATAPIPE_PROBE:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/directppo_train16_official_datapipe_probe_20261004.json}"
development_gate="${CONTROL_DEVELOPMENT_GATE:-/home/USER/workspace/fluid_control_dynamic_weights_worker_20261004/dynamic_fno_development_gate_h100_e2_20261004.json}"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
parent_model_sha="ef95ff96582983680800710679258a6705eb2294fab6a43dfa38163a606ed0c8"
parent_state_sha="63e88160faef1db50140c6aee859ef4de50b84d68e0f675af83f3bc5fb35c63a"
dev30_sha="5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
train8_sha="a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
train16_sha="7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"

for path in "$source_root" "$base" "$train8" "$train16" "$parent"; do
  [[ -d "$path" ]] || { echo "required Worker directory absent: $path" >&2; exit 2; }
done
[[ -f "$source_receipt" && -f "$datapipe_probe" ]] || {
  echo "Worker source/DataPipe receipt absent" >&2; exit 2;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "Worker PhysicsNeMo image identity differs" >&2; exit 2;
}

python3 - "$source_root" "$source_receipt" "$datapipe_probe" "$base" "$train8" \
  "$train16" "$parent" "$dev30_sha" "$train8_sha" "$train16_sha" \
  "$normalization_sha" "$parent_model_sha" "$parent_state_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

(source, receipt_path, probe_path, base, train8, train16, parent) = map(
    pathlib.Path, sys.argv[1:8]
)
(dev30_sha, train8_sha, train16_sha, norm_sha, model_sha, state_sha) = sys.argv[8:]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

if [sha(path / "manifest.json") for path in (base, train8, train16)] != [
    dev30_sha, train8_sha, train16_sha
]:
    raise SystemExit("Worker data manifest identity differs")
if [sha(path / "normalization.json") for path in (base, train8, train16)] != [
    norm_sha, norm_sha, norm_sha
]:
    raise SystemExit("Worker normalization identity differs")
if sha(parent / "FNO.0.2.mdlus") != model_sha or sha(
    parent / "checkpoint.0.2.pt"
) != state_sha:
    raise SystemExit("Worker immutable H100-e2 parent differs")
if (base / "frozen_test").exists() or any(
    (path / split).exists()
    for path in (train8, train16)
    for split in ("validation", "test", "frozen_test")
):
    raise SystemExit("Worker training mounts expose a forbidden split")

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("status") != "CONTROL_TRAIN16_H100_LIFT_BALANCED_STAGED"
    or receipt.get("candidate_kind") != "control_train16_h100_lift_balanced"
    or receipt.get("datapipe_probe_sha256") != sha(probe_path)
    or receipt.get("parent", {}).get("sha256")
    != {"FNO.0.2.mdlus": model_sha, "checkpoint.0.2.pt": state_sha}
):
    raise SystemExit("Worker controlled-branch source receipt differs")
actual = {
    str(path.relative_to(source)): sha(path)
    for path in source.rglob("*") if path.is_file()
}
if actual != receipt.get("source_snapshot_sha256"):
    raise SystemExit("Worker source snapshot differs")
required = {
    "scripts/train_tandem_fno.py",
    "scripts/train_tandem_fno_rollout.py",
    "scripts/spark_gpu_guard.py",
    "src/fluid_control/augmented_datapipe.py",
    "src/fluid_control/tandem_datapipe.py",
    "conf/tandem_fno_control_train16_h100.yaml",
    "conf/tandem_fno_control_train16_h100_lift_balanced.yaml",
}
if not required <= set(actual):
    raise SystemExit("Worker source snapshot lacks controlled-branch dependencies")
balanced = (source / "conf/tandem_fno_control_train16_h100_lift_balanced.yaml").read_text()
if "force_channel_weights: [1.0, 1.0, 4.0, 4.0]" not in balanced:
    raise SystemExit("Worker controlled force weights differ")

probe = json.loads(probe_path.read_text(encoding="utf-8"))
if (
    probe.get("status") != "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS"
    or probe.get("training_executed") is not False
    or probe.get("validation_or_frozen_hdf_opened") is not False
    or probe.get("rollout_steps") != 100
    or probe.get("base_windows") != 720
    or probe.get("total_windows") != 1368
):
    raise SystemExit("Worker DataPipe probe contract differs")
PY

common=(--rm --network none --gpus device=0 --cpus 8 --memory 100g --shm-size 2g
  --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=8g --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1
  --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$source_root/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$source_root/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$source_root/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$parent,dst=/workspace/parent,readonly"
  --mount "type=bind,src=$base,dst=/workspace/base,readonly"
  --mount "type=bind,src=$train8,dst=/workspace/train8,readonly"
  --mount "type=bind,src=$train16,dst=/workspace/train16,readonly"
  --workdir /workspace)

if [[ "$mode" == "--dry-run" ]]; then
  printf 'CONTROL_TRAIN16_H100_LIFT_BALANCED_WORKER_PREFLIGHT_PASS_NO_GPU\n'
  printf 'single_change=force_channel_weights:[1,1,4,1]->[1,1,4,4] windows=1368 parent=%s\n' "$parent_model_sha"
  exit 0
fi

if [[ "$mode" == "--execute" ]]; then
  [[ -f "$development_gate" ]] || {
    echo "reviewed H100 development-gate receipt absent" >&2; exit 2;
  }
  python3 - "$development_gate" "$parent_model_sha" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    receipt.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
    or receipt.get("checkpoint_sha256") != sys.argv[2]
    or receipt.get("ppo_authorized") is not False
    or receipt.get("frozen_test_accessed") is not False
):
    raise SystemExit("reviewed H100 development-gate receipt differs")
PY
fi

target="$output"
epochs=2
extra=()
token="${CONTROL_TRAIN16_LIFT_BALANCED_EXECUTE_TOKEN:-}"
expected_token="EXECUTE_REVIEWED_CONTROL_TRAIN16_H100_LIFT_BALANCED_FULL"
if [[ "$mode" == "--probe" ]]; then
  target="$probe_output"
  epochs=1
  extra=(training.max_train_batches=1 training.max_validation_batches=1)
  token="${CONTROL_TRAIN16_LIFT_BALANCED_PROBE_TOKEN:-}"
  expected_token="EXECUTE_REVIEWED_CONTROL_TRAIN16_H100_LIFT_BALANCED_PROBE"
fi
[[ "$token" == "$expected_token" ]] || { echo "reviewed Worker token required" >&2; exit 2; }
[[ ! -e "$target" ]] || { echo "refusing existing Worker output: $target" >&2; exit 2; }
mkdir "$target"
docker run "${common[@]}" --mount "type=bind,src=$target,dst=/workspace/output" \
  "$image" python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 40 \
  --allocator-fraction 0.45 --margin-gib 4 -- \
  python -u /workspace/scripts/train_tandem_fno_rollout.py \
  --config-name tandem_fno_control_train16_h100_lift_balanced \
  output_dir=/workspace/output hydra.run.dir=/tmp/hydra hydra.output_subdir=null \
  training.initial_checkpoint=/workspace/parent training.epochs="$epochs" "${extra[@]}"
