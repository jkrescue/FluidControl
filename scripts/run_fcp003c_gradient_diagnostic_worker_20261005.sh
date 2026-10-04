#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--dry-run" && "${1:-}" != "--execute" ]]; then
  echo "usage: $0 --dry-run|--execute" >&2
  exit 64
fi
mode="$1"
work="/home/USER/workspace/fluid_control_fcp003c_gradient_diagnostic_20261005"
source_root="$work/source_snapshot"
source_manifest="$work/source_snapshot.sha256"
output="$work/output"
guard_python="/home/USER/.venvs/physicsnemo-94dbdf82-cu13/bin/python"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"

data_root="/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005"
base="$data_root/data/dev30"
train8="$data_root/data/train8"
train16="$data_root/data/train16"
pair_manifest="$data_root/evidence/dynamic_pair_manifest.json"
sampling="$data_root/evidence/real_sampling_contract_v2.json"
parent="$data_root/immutable_parent"
config="$source_root/config.yaml"

diagnostic_sha="abf1a4a76491f86f4a8f6fcb80321c02ae3233b771a600de0ee4a699eee19366"
validator_sha="aa42815f2978392aa07b91382e1c9b312a834e61cacc22e54a4b59bdadf20d06"
source_manifest_sha="9f665735a2ba93a00133ea43fc0f968042c63b00bad8b1e7bbc3f6871b2a7729"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
config_sha="07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
sampling_sha="da078a1c43f03bf86ee71a5010632c05a4868b169dc8f8c7c341c78134873242"
pair_manifest_sha="b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
base_manifest_sha="5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
base_train_split_sha="1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89"
train8_manifest_sha="a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
train16_manifest_sha="7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"

check_sha() {
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || {
    echo "SHA mismatch: $1" >&2
    exit 65
  }
}

check_sha "$source_root/scripts/diagnose_fcp003c_gradient_balance.py" "$diagnostic_sha"
check_sha "$source_root/scripts/validate_fcp003c_gradient_diagnostic.py" "$validator_sha"
check_sha "$work/immutable_guard.py" "$guard_sha"
check_sha "$source_manifest" "$source_manifest_sha"
(cd "$source_root" && sha256sum -c "$source_manifest")
[[ -x "$guard_python" ]] || {
  echo "Worker guard Python is unavailable" >&2
  exit 65
}
"$guard_python" -c 'import torch; assert torch.cuda.is_available()'
check_sha "$config" "$config_sha"
check_sha "$sampling" "$sampling_sha"
check_sha "$pair_manifest" "$pair_manifest_sha"
check_sha "$base/manifest.json" "$base_manifest_sha"
check_sha "$base/splits/train.json" "$base_train_split_sha"
check_sha "$train8/manifest.json" "$train8_manifest_sha"
check_sha "$train16/manifest.json" "$train16_manifest_sha"
for root in "$base" "$train8" "$train16"; do
  check_sha "$root/normalization.json" "$normalization_sha"
done
check_sha "$parent/FNO.0.2.mdlus" "$model_sha"
check_sha "$parent/checkpoint.0.2.pt" "$state_sha"
python3 - "$base" "$train8" "$train16" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

base, train8, train16 = map(Path, sys.argv[1:])
base_split = json.loads((base / "splits/train.json").read_text())
contracts = [
    (base, {f"{name}.h5": value for name, value in base_split["hdf5_sha256"].items()}),
    (train8, json.loads((train8 / "manifest.json").read_text())["hdf_sha256"]),
    (train16, json.loads((train16 / "manifest.json").read_text())["hdf_sha256"]),
]
for root, declared in contracts:
    actual = {path.name for path in (root / "train").glob("*.h5")}
    if actual != set(declared):
        raise SystemExit(f"train HDF set differs: {root}")
    for name, expected in declared.items():
        if digest(root / "train" / name) != expected:
            raise SystemExit(f"train HDF SHA differs: {root / 'train' / name}")
print("TRAIN_ONLY_HDF_SHA_PREFLIGHT_PASS")
PY
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "image differs" >&2
  exit 65
}
[[ ! -e "$output" ]] || {
  echo "exclusive output exists: $output" >&2
  exit 65
}

if [[ "$mode" == "--dry-run" ]]; then
  printf '{"status":"FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_DRY_RUN_READY","positions":16,"fixed_parent":true,"optimizer":false,"validation_or_frozen_mounted":false}\n'
  exit 0
fi

[[ "${FCP003C_GRADIENT_DIAGNOSTIC_APPROVAL_TOKEN:-}" == \
  "EXECUTE_REVIEWED_FCP003C_TRAIN_GRADIENT_DIAGNOSTIC" ]] || {
  echo "approval token missing" >&2
  exit 77
}

mkdir "$output"
mkdir "$output/results"
cp -a "$0" "$output/immutable_launcher.sh"
cp -a "$work/immutable_guard.py" "$output/immutable_guard.py"

python3 - "$output" "$image_id" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, image = Path(sys.argv[1]), sys.argv[2]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "status": "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_LAUNCH_BOUND",
    "image_id": image,
    "fixed_parent": True,
    "positions": 16,
    "optimizer_created_or_stepped": False,
    "candidate_saved": False,
    "validation_or_frozen_mounted": False,
    "immutable_launcher_sha256": sha(root / "immutable_launcher.sh"),
    "immutable_guard_sha256": sha(root / "immutable_guard.py"),
    "source_snapshot_manifest_sha256": "9f665735a2ba93a00133ea43fc0f968042c63b00bad8b1e7bbc3f6871b2a7729",
}
temporary = root / f"launch_receipt.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.link(temporary, root / "launch_receipt.json")
temporary.unlink()
PY

set +e
container_name="fcp003c-gradient-diagnostic-20261005"
cleanup_container() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
    docker rm -f "$container_name" >/dev/null
  fi
}
trap cleanup_container EXIT INT TERM
timeout --signal=TERM --kill-after=30s 20m \
  "$guard_python" "$output/immutable_guard.py" \
    --min-free-gib 20 --allocator-fraction 0.35 --margin-gib 4 --poll-seconds 2 -- \
  docker run --rm --gpus device=0 --network none --read-only --shm-size 2g \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
    --tmpfs /tmp:rw,noexec,nosuid,size=2g \
    --memory 90g --cpus 8 --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp \
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
      --image-id "$image_id" --gpu-memory-fraction 0.35
run_status=$?
set -e
if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  echo "owned container remained after diagnostic; removing and failing closed" >&2
  cleanup_container
  [[ "$run_status" -ne 0 ]] || run_status=70
fi
trap - EXIT INT TERM
printf '%s\n' "$run_status" > "$output/diagnostic_exit_code.txt"
if [[ "$run_status" -ne 0 ]]; then
  echo "diagnostic failed or timed out; partial progress retained" >&2
  exit "$run_status"
fi

validation_temporary="$output/validation.json.tmp.$$"
python3 "$source_root/scripts/validate_fcp003c_gradient_diagnostic.py" \
  --result "$output/results/result.json" \
  --progress "$output/results/result.json.progress.jsonl" \
  --config "$config" --sampling-receipt "$sampling" --parent "$parent" \
  --source-manifest "$source_manifest" \
  --base "$base" --train8 "$train8" --train16 "$train16" \
  --pair-manifest "$pair_manifest" --image-id "$image_id" \
  > "$validation_temporary"
ln "$validation_temporary" "$output/validation.json"
rm "$validation_temporary"

python3 - "$output" "$source_root" "$config" "$sampling" "$source_manifest" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root, source_root, config, sampling, source_manifest = map(Path, sys.argv[1:])
sys.path.insert(0, str(source_root / "scripts"))
from validate_fcp003c_gradient_diagnostic import validate_receipt_bindings

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
result = json.loads((root / "results/result.json").read_text())
progress = [json.loads(line) for line in (root / "results/result.json.progress.jsonl").read_text().splitlines()]
validation = json.loads((root / "validation.json").read_text())
validate_receipt_bindings(
    validation,
    result_path=root / "results/result.json",
    progress_path=root / "results/result.json.progress.jsonl",
    config=config,
    sampling_receipt=sampling,
    source_manifest=source_manifest,
)
def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True
if (
    result.get("status") != "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_COMPLETE"
    or result.get("optimizer_created_or_stepped") is not False
    or result.get("candidate_saved") is not False
    or result.get("validation_or_frozen_accessed") is not False
    or result.get("model_state_sha256_before") != result.get("model_state_sha256_after")
    or len(result.get("rows", [])) != 16
    or len(progress) != 16
    or not finite(result)
    or not finite(progress)
    or validation.get("status") != "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_VALIDATED"
):
    raise SystemExit("gradient diagnostic result contract differs")
payload = {
    "status": "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_COMPLETE",
    "scientific_admission": False,
    "optimizer_created_or_stepped": False,
    "candidate_saved": False,
    "validation_or_frozen_accessed": False,
    "result_sha256": sha(root / "results/result.json"),
    "progress_sha256": sha(root / "results/result.json.progress.jsonl"),
    "validation_sha256": sha(root / "validation.json"),
    "launch_receipt_sha256": sha(root / "launch_receipt.json"),
    "source_snapshot_manifest_sha256": "9f665735a2ba93a00133ea43fc0f968042c63b00bad8b1e7bbc3f6871b2a7729",
    "immutable_launcher_sha256": sha(root / "immutable_launcher.sh"),
    "immutable_guard_sha256": sha(root / "immutable_guard.py"),
}
temporary = root / f"completion_receipt.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.link(temporary, root / "completion_receipt.json")
temporary.unlink()
PY
