#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--dry-run" && "${1:-}" != "--execute" ]]; then
  echo "usage: $0 --dry-run|--execute" >&2
  exit 64
fi
mode="$1"
repo="/workspace/fluid_control"
output="$repo/artifacts/true_state_paired_force_backward_probe_20261005"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
script_sha="d9552f7f96b3c4d73109fcf81c9557c7aaf23492c863f2fbcc9e89b4e2f28331"
helper_sha="d382c7c886509fb12b85e7f79080dcf1b85d9c2315c15717e97310459dbfee30"
adapter_sha="2414b0a2b2635f87e0842ed12d36ad7a3eb6b1b8759cacdb5b8c8fa9fed5af31"
trainer_sha="9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
manifest_sha="b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
action_manifest_sha="a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
config_sha="1ea6e162f1cf4994f1dd628b1ee9b06bded99c382e346839697df3a74b7ccfbb"
action_hdf_sha="400b3b5f8ea381da59c7806aef19f7cd2b50da303f717ad93cebf505c5949e9f"
zero_hdf_sha="243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01"
cpu_probe_sha="e549121553e4e1fa0fffe5c0f3080dc1f03db615342530a7a7fcb0a0e45ac7aa"
action="$repo/data/curated/tandem_cylinders_dynamic_train8_v1"
zero="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
manifest="$repo/artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
parent="$repo/artifacts/tandem_fno_control_train16_h100_20261004/best"
cpu_probe="$repo/artifacts/fc_p003_dynamic8_pair_candidate_20261005/cpu_probe.json"

check_sha() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || { echo "SHA mismatch: $1" >&2; exit 65; }; }
check_sha "$repo/scripts/probe_true_state_paired_force_backward.py" "$script_sha"
check_sha "$repo/src/fluid_control/paired_step_force.py" "$helper_sha"
check_sha "$repo/src/fluid_control/dynamic_pair_stat_datapipe.py" "$adapter_sha"
check_sha "$repo/scripts/train_tandem_fno.py" "$trainer_sha"
check_sha "$repo/scripts/spark_gpu_guard.py" "$guard_sha"
check_sha "$manifest" "$manifest_sha"
check_sha "$action/manifest.json" "$action_manifest_sha"
check_sha "$action/normalization.json" "$normalization_sha"
cmp -s "$action/normalization.json" "$zero/normalization.json" || { echo "normalization differs" >&2; exit 65; }
check_sha "$action/train/dynamic_train8_b00_multisine.h5" "$action_hdf_sha"
check_sha "$zero/train/matched_start_acquisition_train_b00_zero.h5" "$zero_hdf_sha"
check_sha "$parent/FNO.0.2.mdlus" "$model_sha"
check_sha "$parent/checkpoint.0.2.pt" "$state_sha"
check_sha "$cpu_probe" "$cpu_probe_sha"
check_sha "$repo/conf/tandem_fno_control_train16_h100.yaml" "$config_sha"
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image differs" >&2; exit 65; }
[[ ! -e "$output" ]] || { echo "exclusive output exists: $output" >&2; exit 65; }
if [[ "$mode" == "--dry-run" ]]; then
  printf '{"status":"TRUE_STATE_PAIRED_FORCE_BACKWARD_PROBE_DRY_RUN_READY","pair_id":"b00:multisine","horizon":100,"chunk_size":10,"optimizer_steps":0,"validation_or_frozen_mounted":false}\n'
  exit 0
fi
[[ "${TRUE_STATE_FORCE_PROBE_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_TRUE_STATE_FORCE_BACKWARD_PROBE" ]] || { echo "approval token missing" >&2; exit 77; }
mkdir "$output"
mkdir -p "$output/source_snapshot/scripts" "$output/results"
cp -a "$repo/src" "$repo/conf" "$output/source_snapshot/"
cp -a "$repo/scripts/probe_true_state_paired_force_backward.py" \
  "$repo/scripts/train_tandem_fno.py" "$output/source_snapshot/scripts/"
check_sha "$output/source_snapshot/scripts/probe_true_state_paired_force_backward.py" "$script_sha"
check_sha "$output/source_snapshot/src/fluid_control/paired_step_force.py" "$helper_sha"
check_sha "$output/source_snapshot/src/fluid_control/dynamic_pair_stat_datapipe.py" "$adapter_sha"
(cd "$output/source_snapshot" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source_snapshot.sha256"
cp -a "$0" "$output/immutable_launcher.sh"
cp -a "$repo/scripts/spark_gpu_guard.py" "$output/immutable_guard.py"
python3 - "$output" "$image_id" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root, image = Path(sys.argv[1]), sys.argv[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
value = {"status": "TRUE_STATE_PAIRED_FORCE_BACKWARD_PROBE_LAUNCH_BOUND",
         "image_id": image, "validation_or_frozen_mounted": False,
         "optimizer_steps": 0,
         "immutable_launcher_sha256": sha(root / "immutable_launcher.sh"),
         "immutable_guard_sha256": sha(root / "immutable_guard.py"),
         "source_snapshot_manifest_sha256": sha(root / "source_snapshot.sha256")}
tmp = root / f"launch_receipt.json.tmp.{os.getpid()}"
tmp.write_text(json.dumps(value, indent=2) + "\n")
os.link(tmp, root / "launch_receipt.json"); tmp.unlink()
PY
/home/USER/env_isaaclab/bin/python "$repo/scripts/spark_gpu_guard.py" \
  --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4 --poll-seconds 2 -- \
docker run --rm --gpus device=0 --network none --read-only --shm-size 1g \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g --tmpfs /root/.cache:rw,noexec,nosuid,size=1g \
  --memory 90g --cpus 8 \
  -e PYTHONPATH=/workspace/project/src:/workspace/project/scripts \
  -v "$output/source_snapshot:/workspace/project:ro" \
  -v "$action/train:/workspace/action/train:ro" \
  -v "$action/normalization.json:/workspace/action/normalization.json:ro" \
  -v "$zero/train:/workspace/zero/train:ro" \
  -v "$zero/normalization.json:/workspace/zero/normalization.json:ro" \
  -v "$manifest:/workspace/pair_manifest.json:ro" \
  -v "$parent:/workspace/parent:ro" \
  -v "$output/source_snapshot.sha256:/workspace/source_snapshot.sha256:ro" \
  -v "$output/launch_receipt.json:/workspace/launch_receipt.json:ro" \
  -v "$output/results:/workspace/output:rw" \
  "$image" python /workspace/project/scripts/probe_true_state_paired_force_backward.py \
    --repo /workspace/project --action-root /workspace/action --zero-root /workspace/zero \
    --pair-manifest /workspace/pair_manifest.json --checkpoint /workspace/parent \
    --output /workspace/output/result.json --pair-id b00:multisine \
    --image-id "$image_id" \
    --source-manifest /workspace/source_snapshot.sha256 \
    --launch-receipt /workspace/launch_receipt.json \
    --chunk-size 10 --equivalence-steps 20 --total-steps 100 \
    --gpu-memory-fraction 0.20 --min-mem-available-gib 20
python3 - "$output" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
result = json.loads((root / "results/result.json").read_text())
if result.get("status") != "TRUE_STATE_PAIRED_FORCE_BACKWARD_TECHNICAL_PROBE_PASS":
    raise SystemExit("technical probe result status differs")
value = {"status": "TRUE_STATE_PAIRED_FORCE_BACKWARD_TECHNICAL_PROBE_COMPLETE",
         "scientific_result": False, "optimizer_steps": 0,
         "validation_or_frozen_accessed": False,
         "image_id": result["image_id"], "input_sha256": result["input_sha256"],
         "result_sha256": sha(root / "results/result.json"),
         "launch_receipt_sha256": sha(root / "launch_receipt.json"),
         "source_snapshot_manifest_sha256": sha(root / "source_snapshot.sha256"),
         "immutable_launcher_sha256": sha(root / "immutable_launcher.sh"),
         "immutable_guard_sha256": sha(root / "immutable_guard.py")}
tmp = root / f"completion_receipt.json.tmp.{os.getpid()}"
tmp.write_text(json.dumps(value, indent=2) + "\n")
os.link(tmp, root / "completion_receipt.json"); tmp.unlink()
PY
