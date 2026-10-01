#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
if [[ "$(uname -m)" == aarch64 ]]; then
    echo "Legacy host/GPU runner disabled on DGX Spark; see docs/SPARK_GPU_TRAINING.md" >&2
    exit 2
fi

output="artifacts/tandem_fno_rollout_no_tf_v1/action_sensitivity"
data="data/curated/tandem_cylinders_expanded_v1"
mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

for mode in observed zero sign_flip shuffle; do
    echo "[$(date '+%F %T %Z')] Evaluating action mode: $mode"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/evaluate_tandem_fno.py \
        --data "$data" \
        --config conf/tandem_fno_expanded.yaml \
        --checkpoint-dir artifacts/tandem_fno_rollout_no_tf_v1/best \
        --output "$output/${mode}.json" \
        --visualization-dir "$output/unused_visualizations" \
        --horizons 1 10 50 100 \
        --visualizations-per-horizon 0 \
        --action-mode "$mode"
done

.venv/bin/python scripts/summarize_tandem_action_sensitivity.py \
    "$output"/{observed,zero,sign_flip,shuffle}.json \
    --output "$output/summary.json" \
    --markdown "$output/summary.md"

echo "[$(date '+%F %T %Z')] ACTION_SENSITIVITY_OK"
