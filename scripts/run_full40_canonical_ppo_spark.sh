#!/usr/bin/env bash
# Run only the strictly gated full40 canonical HydroGym entry on Spark GPU0.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
image="fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
runtime_image_id="sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
validation_image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
data="data/curated/tandem_cylinders_matched_start_full40_v1"
dev30_data="data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
checkpoint="${CHECKPOINT_DIR:-artifacts/tandem_fno_full40_h20_seed20261003_10epoch/best}"
validation_root="${VALIDATION_ROOT:-artifacts/tandem_cylinders/full40_fno_validation_pending_review}"
promotion_receipt="${PROMOTION_RECEIPT:-artifacts/tandem_cylinders/dev30_full40_promotion.json}"
if [[ "$mode" == "--dry-run" ]]; then
  default_output="artifacts/hydrogym/full40_canonical_joint_v1/preflight.json"
else
  default_output="artifacts/hydrogym/full40_canonical_joint_v1/run_20261003"
fi
output="${OUTPUT_PATH:-$default_output}"
if [[ "$mode" == "--train-only-smoke" ]]; then
  default_baselines="artifacts/matched_start_full40_extension/train20_physics_summary.json"
else
  default_baselines="artifacts/tandem_cylinders/full40_canonical_zero_baselines.json"
fi
baselines="${BASELINES:-$default_baselines}"

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$runtime_image_id" ]] || {
  echo "pinned PhysicsNeMo+HydroGym image ID mismatch" >&2; exit 2;
}
[[ "$output" == artifacts/hydrogym/* && "$output" != *..* ]] || {
  echo "output must remain under artifacts/hydrogym" >&2; exit 2;
}
case "$mode" in
  --dry-run|--execute|--train-only-smoke) ;;
  *) echo "mode must be --dry-run, --execute, or --train-only-smoke" >&2; exit 2 ;;
esac

command=(python -u scripts/train_full40_hydrogym_ppo_canonical.py
  --data "$data" --config conf/tandem_fno_full40_h20.yaml
  --checkpoint-dir "$checkpoint" --dev30-data "$dev30_data"
  --promotion-receipt "$promotion_receipt" --baselines "$baselines"
  --validation-gate "$validation_root/gate.json"
  --validation-report "$validation_root/evaluation.json"
  --validation-segments "$validation_root/segments.json"
  --predeclaration artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json
  --window-gate "$validation_root/canonical_window_gate.json"
  --dynamic-gate "$validation_root/dynamic_action_gate.json"
  --image-id "$validation_image_id" --runtime-image-id "$runtime_image_id"
  --output "$output" --episode-steps 100 --timesteps "${TIMESTEPS:-8192}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-2048}"
  --gpu-memory-fraction 0.20 "$mode")

if [[ "$mode" == "--dry-run" ]]; then
  docker run --rm --network none --cpus 2 --memory 4g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace "$image" \
    "${command[@]}"
else
  docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --env PYTHONPATH=/workspace/.tools/hydrogym:/workspace/src:/workspace/scripts \
    --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace "$image" \
    python -u scripts/spark_ppo_gpu_guard.py --min-free-gib 20 \
    --allocator-fraction 0.20 --margin-gib 4 -- "${command[@]}"
fi
