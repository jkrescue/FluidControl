#!/usr/bin/env bash
# Stage and run the conditional train20+train8+train16 H100 fine-tune.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
case "$mode" in
  --dry-run|--probe|--execute) ;;
  *) echo "mode must be --dry-run, --probe, or --execute" >&2; exit 2 ;;
esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
candidate="$root/artifacts/tandem_fno_control_train16_h100_20261004"
probe_output="$root/artifacts/tandem_fno_control_train16_h100_probe_20261004"
parent_source="$root/artifacts/tandem_fno_dynamic_train8_h100_worker_h100_e5_2ep_r2_20261004/best"
base="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$root/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$root/data/curated/tandem_cylinders_directppo_train16_v1"
datapipe_probe="$root/artifacts/tandem_cylinders/directppo_train16_official_datapipe_probe_20261004.json"
parent_model_sha="ef95ff96582983680800710679258a6705eb2294fab6a43dfa38163a606ed0c8"
parent_state_sha="63e88160faef1db50140c6aee859ef4de50b84d68e0f675af83f3bc5fb35c63a"
dev30_sha="5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
train8_sha="a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
train16_sha="7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "pinned PhysicsNeMo image identity differs" >&2; exit 2;
}
[[ "$(sha256sum "$parent_source/FNO.0.2.mdlus" | awk '{print $1}')" == "$parent_model_sha" ]] || {
  echo "immutable H100-e2 parent model differs" >&2; exit 2;
}
[[ "$(sha256sum "$parent_source/checkpoint.0.2.pt" | awk '{print $1}')" == "$parent_state_sha" ]] || {
  echo "immutable H100-e2 parent state differs" >&2; exit 2;
}

python3 - "$base" "$train8" "$train16" "$datapipe_probe" \
  "$dev30_sha" "$train8_sha" "$train16_sha" "$normalization_sha" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

base, train8, train16, probe = map(pathlib.Path, sys.argv[1:5])
dev30_sha, train8_sha, train16_sha, normalization_sha = sys.argv[5:]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if [sha(path / "manifest.json") for path in (base, train8, train16)] != [
    dev30_sha, train8_sha, train16_sha
]:
    raise SystemExit("training manifest identity differs")
if [sha(path / "normalization.json") for path in (base, train8, train16)] != [
    normalization_sha, normalization_sha, normalization_sha
]:
    raise SystemExit("training normalization identity differs")
if (base / "frozen_test").exists() or any(
    (path / split).exists()
    for path in (train8, train16)
    for split in ("validation", "test", "frozen_test")
):
    raise SystemExit("training mounts expose a forbidden split")

report = json.loads(probe.read_text(encoding="utf-8"))
if (
    report.get("status") != "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS"
    or report.get("training_executed") is not False
    or report.get("validation_or_frozen_hdf_opened") is not False
    or report.get("rollout_steps") != 100
    or report.get("base_windows") != 720
    or report.get("total_windows") != 1368
    or len(report.get("additional_sources", [])) != 2
    or len(report.get("samples", [])) != 4
    or {row.get("metadata", {}).get("dataset_index") for row in report["samples"]}
    != {0, 1, 2}
):
    raise SystemExit("official real-HDF DataPipe probe contract differs")
expected_shapes = {
    "state": [1, 3, 128, 256],
    "target_state": [1, 100, 3, 128, 256],
    "target_force": [1, 100, 4],
    "omega": [1, 101, 1],
    "mask": [1, 1, 128, 256],
}
for row in report["samples"]:
    dataset_index = row.get("metadata", {}).get("dataset_index")
    tolerance = 2.0e-6 if dataset_index == 2 else 2.0e-5
    if (
        row.get("metadata", {}).get("split") != "train"
        or row.get("metadata", {}).get("rollout_steps") != 100
        or any(row.get("shapes", {}).get(key) != value for key, value in expected_shapes.items())
        or not math.isfinite(float(row.get("max_abs_omega", math.nan)))
        or not math.isfinite(float(row.get("max_delta_omega", math.nan)))
        or float(row["max_abs_omega"]) > 0.75 + tolerance
        or float(row["max_delta_omega"]) > 0.1 + tolerance
        or not math.isclose(
            float(row.get("action_slew_representation_tolerance", math.nan)),
            tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise SystemExit("official real-HDF DataPipe sample contract differs")
expected = [
    ("/workspace/train8", 408, train8_sha, 8, "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED"),
    ("/workspace/train16", 240, train16_sha, 16, "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED"),
]
for row, (root, windows, manifest, count, status) in zip(
    report["additional_sources"], expected, strict=True
):
    if (
        row.get("root") != root
        or row.get("windows") != windows
        or row.get("stride") != 2
        or row.get("manifest_sha256") != manifest
        or row.get("normalization_sha256") != normalization_sha
        or row.get("trajectory_count") != count
        or row.get("release_status") != status
    ):
        raise SystemExit("official real-HDF DataPipe probe source differs")
implementation = report.get("implementation_sha256", {})
for relative in (
    "src/fluid_control/augmented_datapipe.py",
    "src/fluid_control/tandem_datapipe.py",
):
    if implementation.get(relative) != sha(pathlib.Path(relative)):
        raise SystemExit(f"DataPipe probe implementation differs: {relative}")
if report.get("script_sha256") != sha(pathlib.Path("scripts/probe_directppo_train16_datapipe.py")):
    raise SystemExit("DataPipe probe script differs")
PY

stage_launch() {
  if [[ -e "$candidate/source_receipt.json" ]]; then
    [[ -d "$candidate/source_snapshot" && -d "$candidate/immutable_parent" ]] || {
      echo "partial candidate staging is not reusable" >&2; exit 2;
    }
    python3 - "$candidate" "$datapipe_probe" "$parent_model_sha" "$parent_state_sha" <<'PY'
import hashlib
import json
import pathlib
import sys
candidate, probe = map(pathlib.Path, sys.argv[1:3])
model_sha, state_sha = sys.argv[3:]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = json.loads((candidate / "source_receipt.json").read_text())
parent = json.loads((candidate / "parent_receipt.json").read_text())
expected_parent = {"FNO.0.2.mdlus": model_sha, "checkpoint.0.2.pt": state_sha}
if (
    receipt.get("status") != "CONTROL_TRAIN16_H100_STAGED_NOT_EXECUTED"
    or receipt.get("datapipe_probe_sha256") != sha(probe)
    or receipt.get("parent", {}).get("sha256") != expected_parent
    or parent.get("sha256") != expected_parent
):
    raise SystemExit("stored launch staging receipt differs")
snapshot = candidate / "source_snapshot"
actual = {
    str(path.relative_to(snapshot)): sha(path)
    for path in snapshot.rglob("*") if path.is_file()
}
if actual != receipt.get("source_snapshot_sha256"):
    raise SystemExit("stored source snapshot differs")
for name, digest in expected_parent.items():
    if sha(candidate / "immutable_parent" / name) != digest:
        raise SystemExit("stored immutable parent differs")
PY
    return
  fi
  [[ ! -e "$candidate" ]] || { echo "refusing partial candidate directory" >&2; exit 2; }
  [[ -z "$(git status --porcelain -- scripts src conf)" ]] || {
    echo "scripts/src/conf must be committed and clean before snapshot" >&2; exit 2;
  }
  mkdir "$candidate"
  mkdir "$candidate/source_snapshot" "$candidate/immutable_parent"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf - scripts src conf |
    tar -xf - -C "$candidate/source_snapshot"
  cp "$parent_source/FNO.0.2.mdlus" "$candidate/immutable_parent/"
  cp "$parent_source/checkpoint.0.2.pt" "$candidate/immutable_parent/"
  chmod 0444 "$candidate/immutable_parent"/*
  chmod -R a-w "$candidate/source_snapshot"

  python3 - "$root" "$candidate" "$datapipe_probe" "$image_id" \
    "$dev30_sha" "$train8_sha" "$train16_sha" "$normalization_sha" \
    "$parent_model_sha" "$parent_state_sha" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

(repo, candidate, probe) = map(pathlib.Path, sys.argv[1:4])
(image_id, dev30_sha, train8_sha, train16_sha, normalization_sha,
 parent_model_sha, parent_state_sha) = sys.argv[4:]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def write_exclusive(path, payload):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
).stdout.strip()
tree = subprocess.run(
    ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=True,
    capture_output=True, text=True
).stdout.strip()
snapshot = candidate / "source_snapshot"
snapshot_hashes = {
    str(path.relative_to(snapshot)): sha(path)
    for path in sorted(snapshot.rglob("*"))
    if path.is_file()
}
parent_hashes = {
    "FNO.0.2.mdlus": parent_model_sha,
    "checkpoint.0.2.pt": parent_state_sha,
}
write_exclusive(candidate / "parent_receipt.json", {
    "status": "IMMUTABLE_PARENT_COPIED",
    "source": "artifacts/tandem_fno_dynamic_train8_h100_worker_h100_e5_2ep_r2_20261004/best",
    "sha256": parent_hashes,
})
write_exclusive(candidate / "source_receipt.json", {
    "status": "CONTROL_TRAIN16_H100_STAGED_NOT_EXECUTED",
    "candidate_kind": "control_train16_h100",
    "source_snapshot_commit": commit,
    "source_snapshot_tree": tree,
    "source_snapshot_sha256": snapshot_hashes,
    "physicsnemo_image": "fluid-control-physicsnemo:2.2.2",
    "physicsnemo_image_id": image_id,
    "datapipe_probe_sha256": sha(probe),
    "data": {
        "dev30": {"manifest_sha256": dev30_sha, "normalization_sha256": normalization_sha},
        "train8": {"manifest_sha256": train8_sha, "normalization_sha256": normalization_sha, "file_count": 8},
        "train16": {"manifest_sha256": train16_sha, "normalization_sha256": normalization_sha, "file_count": 16},
        "frozen_test_transferred_or_opened": False,
    },
    "parent": {
        "candidate_kind": "dynamic_train8_h100",
        "checkpoint_epoch": 2,
        "sha256": parent_hashes,
        "optimizer_loaded": False,
        "new_optimizer": "AdamW",
    },
    "planned_training": {
        "rollout_steps": 100, "validation_rollout_steps": 100,
        "batch_size": 2, "epochs": 2, "learning_rate": 1.0e-5,
        "base_windows": 720, "train8_windows": 408, "train16_windows": 240,
    },
})
PY
}

common=(--rm --network none --gpus device=0 --cpus 8 --memory 90g --shm-size 2g --pids-limit 512
  --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=8g --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$candidate/source_snapshot/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$candidate/source_snapshot/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$candidate/source_snapshot/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$candidate/immutable_parent,dst=/workspace/parent,readonly"
  --mount "type=bind,src=$base,dst=/workspace/base,readonly"
  --mount "type=bind,src=$train8,dst=/workspace/train8,readonly"
  --mount "type=bind,src=$train16,dst=/workspace/train16,readonly"
  --workdir /workspace)

if [[ "$mode" == "--dry-run" ]]; then
  printf 'CONTROL_TRAIN16_H100_PREFLIGHT_PASS_NO_GPU_NO_TRAINING_NO_PPO\n'
  printf 'candidate=%s windows=720+408+240=1368 parent_model_sha=%s\n' \
    "${candidate#$root/}" "$parent_model_sha"
  printf 'image_id=%s allocator_fraction=.45 minimum_free_gib=20 episode_horizon=100\n' "$image_id"
  exit 0
fi

stage_launch
if [[ "$mode" == "--probe" ]]; then
  [[ "${CONTROL_TRAIN16_PROBE_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_CONTROL_TRAIN16_H100_PROBE" ]] || {
    echo "reviewed probe approval token is required" >&2; exit 2;
  }
  [[ ! -e "$probe_output" ]] || { echo "refusing existing probe output" >&2; exit 2; }
  mkdir "$probe_output"
  docker run "${common[@]}" \
    --mount "type=bind,src=$probe_output,dst=/workspace/output" \
    "$image" \
    python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20 \
      --allocator-fraction 0.45 --margin-gib 4 -- \
    python -u /workspace/scripts/train_tandem_fno_rollout.py \
      --config-name tandem_fno_control_train16_h100 output_dir=/workspace/output \
      hydra.run.dir=/tmp/hydra hydra.output_subdir=null \
      training.initial_checkpoint=/workspace/parent training.epochs=1 \
      training.max_train_batches=1 training.max_validation_batches=1
  exit 0
fi

[[ "${CONTROL_TRAIN16_EXECUTE_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_CONTROL_TRAIN16_H100_FULL" ]] || {
  echo "reviewed full-training approval token is required" >&2; exit 2;
}
development_relative="${CURRENT_FNO_DEVELOPMENT_GATE:-}"
[[ "$development_relative" == artifacts/tandem_cylinders/dynamic_fno_development_gate_* && "$development_relative" != *..* ]] || {
  echo "CURRENT_FNO_DEVELOPMENT_GATE must identify the reviewed current-candidate result" >&2; exit 2;
}
python3 - "$root/$development_relative" "$parent_model_sha" <<'PY'
import json
import pathlib
import sys
result = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    result.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
    or result.get("checkpoint_sha256") != sys.argv[2]
    or result.get("frozen_test_accessed") is not False
    or result.get("ppo_authorized") is not False
):
    raise SystemExit("current H100 development result does not justify conditional training")
PY
[[ ! -e "$candidate/training_history.json" && ! -e "$candidate/checkpoints" ]] || {
  echo "refusing to resume or overwrite the fixed two-epoch candidate" >&2; exit 2;
}
docker run "${common[@]}" \
  --mount "type=bind,src=$candidate,dst=/workspace/output" \
  "$image" \
  python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20 \
    --allocator-fraction 0.45 --margin-gib 4 -- \
  python -u /workspace/scripts/train_tandem_fno_rollout.py \
    --config-name tandem_fno_control_train16_h100 output_dir=/workspace/output \
    hydra.run.dir=/tmp/hydra hydra.output_subdir=null \
    training.initial_checkpoint=/workspace/parent
