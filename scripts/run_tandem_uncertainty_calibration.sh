#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"

output="artifacts/tandem_fno_rollout_no_tf_v1/uncertainty_calibration"
mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts .venv/bin/python \
    scripts/calibrate_tandem_surrogate_uncertainty.py \
    --production-checkpoint artifacts/tandem_fno_rollout_no_tf_v1/best \
    --committee-checkpoints \
        artifacts/tandem_fno_rollout_expanded_v1/best \
        artifacts/tandem_fno_expanded_v1/best \
    --horizons 10 50 \
    --segment-stride 10 \
    --batch-size 8 \
    --output "$output/report.json"

echo "[$(date '+%F %T %Z')] UNCERTAINTY_CALIBRATION_OK"
