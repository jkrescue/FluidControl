#!/usr/bin/env bash
set -euo pipefail

work="/home/USER/workspace/fluid_control_fcp003c_gradient_debug_20261005"
source_root="$work/source_snapshot"
source_manifest="$work/source_snapshot.sha256"
output="$work/output"
guard_python="/home/USER/.venvs/physicsnemo-94dbdf82-cu13/bin/python"
data_root="/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005"
base="$data_root/data/dev30"
train8="$data_root/data/train8"
train16="$data_root/data/train16"
pair_manifest="$data_root/evidence/dynamic_pair_manifest.json"
sampling="$data_root/evidence/real_sampling_contract_v2.json"
parent="$data_root/immutable_parent"
config="$source_root/config.yaml"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"

check_sha() {
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || {
    echo "SHA mismatch: $1" >&2
    exit 65
  }
}
check_sha "$source_root/scripts/diagnose_fcp003c_gradient_balance.py" \
  "ec10b754dbf83e7cddbed36047a4d2042507d14e309e2c13a48f21e3bd983457"
check_sha "$source_manifest" \
  "05680e6056f6df75552e58dda8f1a781def3a412ed5d956b3eefbe700af2dc29"
check_sha "$work/immutable_guard.py" \
  "75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
check_sha "$work/evidence/approval.md" \
  "73efc3508170fd650f04919904c60d080e395d1ffb0da8e9c8f7fa09f754023e"
check_sha "$config" \
  "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
check_sha "$sampling" \
  "da078a1c43f03bf86ee71a5010632c05a4868b169dc8f8c7c341c78134873242"
check_sha "$pair_manifest" \
  "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
check_sha "$base/manifest.json" \
  "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
check_sha "$base/splits/train.json" \
  "1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89"
check_sha "$train8/manifest.json" \
  "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
check_sha "$train16/manifest.json" \
  "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
for root in "$base" "$train8" "$train16"; do
  check_sha "$root/normalization.json" \
    "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
done
check_sha "$parent/FNO.0.2.mdlus" \
  "8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
check_sha "$parent/checkpoint.0.2.pt" \
  "1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
python3 - "$base" "$train8" "$train16" <<'PY'
import hashlib, json, sys
from pathlib import Path
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()
base, train8, train16 = map(Path, sys.argv[1:])
split = json.loads((base / "splits/train.json").read_text())["hdf5_sha256"]
contracts = [
    (base, {f"{name}.h5": value for name, value in split.items()}),
    (train8, json.loads((train8 / "manifest.json").read_text())["hdf_sha256"]),
    (train16, json.loads((train16 / "manifest.json").read_text())["hdf_sha256"]),
]
for root, declared in contracts:
    if {path.name for path in (root / "train").glob("*.h5")} != set(declared):
        raise SystemExit(f"train HDF set differs: {root}")
    for name, expected in declared.items():
        if digest(root / "train" / name) != expected:
            raise SystemExit(f"train HDF SHA differs: {root / 'train' / name}")
print("TRAIN_ONLY_HDF_SHA_PREFLIGHT_PASS")
PY
(cd "$source_root" && sha256sum -c "$source_manifest")
[[ -x "$guard_python" ]]
"$guard_python" -c 'import torch; assert torch.cuda.is_available()'
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ ! -e "$output" ]]
[[ "${FCP003C_FIRST_POSITION_DEBUG_APPROVAL_TOKEN:-}" == \
  "EXECUTE_REVIEWED_FCP003C_FIRST_POSITION_DEBUG" ]]

mkdir "$output"
mkdir "$output/results"
cp -a "$0" "$output/immutable_launcher.sh"
cp -a "$work/immutable_guard.py" "$output/immutable_guard.py"
python3 - "$output" "$image_id" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root, image = Path(sys.argv[1]), sys.argv[2]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "status": "FC_P003C_FIRST_POSITION_DEBUG_LAUNCH_BOUND",
    "debug_only": True,
    "max_positions": 1,
    "relative_tolerance": 2e-5,
    "optimizer_created_or_stepped": False,
    "candidate_saved": False,
    "validation_or_frozen_mounted": False,
    "image_id": image,
    "immutable_launcher_sha256": sha(root / "immutable_launcher.sh"),
    "immutable_guard_sha256": sha(root / "immutable_guard.py"),
}
temporary = root / f"launch_receipt.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.link(temporary, root / "launch_receipt.json")
temporary.unlink()
PY

container_name="fcp003c-gradient-first-position-debug-20261005"
cleanup_container() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
    docker rm -f "$container_name" >/dev/null
  fi
}
trap cleanup_container EXIT INT TERM
set +e
timeout --signal=TERM --kill-after=30s 5m \
  "$guard_python" "$output/immutable_guard.py" \
    --min-free-gib 20 --allocator-fraction 0.35 --margin-gib 4 --poll-seconds 2 -- \
  docker run --rm --gpus device=0 --network none --read-only --shm-size 2g \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
    --tmpfs /tmp:rw,noexec,nosuid,size=2g --memory 90g --cpus 8 \
    --user "$(id -u):$(id -g)" -e USER=nvidia -e LOGNAME=nvidia -e HOME=/tmp \
    -e PYTHONPATH=/workspace/project/src:/workspace/project/scripts \
    --name "$container_name" \
    -v "$source_root:/workspace/project:ro" \
    -v "$base/train:/workspace/base/train:ro" \
    -v "$base/manifest.json:/workspace/base/manifest.json:ro" \
    -v "$base/normalization.json:/workspace/base/normalization.json:ro" \
    -v "$base/splits/train.json:/workspace/base/splits/train.json:ro" \
    -v "$train8/train:/workspace/train8/train:ro" \
    -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" \
    -v "$train8/normalization.json:/workspace/train8/normalization.json:ro" \
    -v "$train16/train:/workspace/train16/train:ro" \
    -v "$train16/manifest.json:/workspace/train16/manifest.json:ro" \
    -v "$train16/normalization.json:/workspace/train16/normalization.json:ro" \
    -v "$pair_manifest:/workspace/pair_manifest.json:ro" \
    -v "$sampling:/workspace/sampling_receipt.json:ro" \
    -v "$parent:/workspace/parent:ro" \
    -v "$source_manifest:/workspace/source_snapshot.sha256:ro" \
    -v "$output/results:/workspace/output:rw" \
    "$image" python /workspace/project/scripts/diagnose_fcp003c_gradient_balance.py \
      --config /workspace/project/config.yaml \
      --base /workspace/base --train8 /workspace/train8 --train16 /workspace/train16 \
      --pair-manifest /workspace/pair_manifest.json \
      --sampling-receipt /workspace/sampling_receipt.json \
      --parent /workspace/parent --output /workspace/output/result.json \
      --source-manifest /workspace/source_snapshot.sha256 \
      --image-id "$image_id" --gpu-memory-fraction 0.35 --max-positions 1
run_status=$?
set -e
printf '%s\n' "$run_status" > "$output/debug_exit_code.txt"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  cleanup_container
  [[ "$run_status" -ne 0 ]] || run_status=70
fi
trap - EXIT INT TERM
if [[ "$run_status" -ne 0 ]]; then
  echo "DEBUG_ONLY run failed; evidence retained; no retry" >&2
  exit "$run_status"
fi
python3 - "$output" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
result = json.loads((root / "results/result.json").read_text())
if (
    result.get("status") != "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_DEBUG_ONLY"
    or result.get("max_positions") != 1
    or len(result.get("rows", [])) != 1
    or result.get("optimizer_created_or_stepped") is not False
    or result.get("candidate_saved") is not False
    or result.get("validation_or_frozen_accessed") is not False
):
    raise SystemExit("DEBUG_ONLY result contract differs")
print("FC_P003C_FIRST_POSITION_DEBUG_ONLY_RESULT_READY")
PY
