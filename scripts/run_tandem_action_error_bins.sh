#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"

output="artifacts/tandem_fno_rollout_no_tf_v1/action_error_bins"
mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/evaluate_tandem_fno.py \
    --data data/curated/tandem_cylinders_expanded_v1 \
    --config conf/tandem_fno_expanded.yaml \
    --checkpoint-dir artifacts/tandem_fno_rollout_no_tf_v1/best \
    --output "$output/evaluation.json" \
    --visualization-dir "$output/unused_visualizations" \
    --horizons 1 10 50 100 \
    --segment-stride 5 \
    --visualizations-per-horizon 0 \
    --action-mode observed \
    --segment-metrics-output "$output/segments.json"

.venv/bin/python scripts/summarize_tandem_action_error_bins.py \
    "$output/segments.json" \
    --output "$output/summary.json" \
    --markdown "$output/summary.md"

echo "[$(date '+%F %T %Z')] ACTION_ERROR_BINS_OK"
