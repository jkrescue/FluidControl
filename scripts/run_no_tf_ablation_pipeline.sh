#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"

output="artifacts/tandem_fno_rollout_no_tf_v1"
data="data/curated/tandem_cylinders_expanded_v1"
pipeline_log="$output/pipeline.log"
mkdir -p "$output"
exec > >(tee -a "$pipeline_log") 2>&1

timestamp() { date '+%F %T %Z'; }

echo "[$(timestamp)] Validating controlled ablation configuration"
PYTHONPATH=src:scripts .venv/bin/python - <<'PY'
from pathlib import Path

from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=str(Path("conf").resolve()), version_base="1.3"):
    config = compose(config_name="tandem_fno_rollout_no_tf")
assert float(config.training.teacher_forcing_start) == 0.0
assert float(config.training.teacher_forcing_end) == 0.0
assert int(config.training.epochs) == 30
assert int(config.training.batch_size) == 16
print("NO_TF_ABLATION_CONFIG_OK")
PY

echo "[$(timestamp)] Starting 30-epoch no-teacher-forcing rollout ablation"
OUTPUT_DIR="$output" CONFIG_NAME=tandem_fno_rollout_no_tf \
    BATCH_SIZE=16 SMOKE=false bash scripts/run_tandem_fno_rollout_expanded.sh

echo "[$(timestamp)] Evaluating no-teacher-forcing model"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/evaluate_tandem_fno.py \
    --data "$data" --config conf/tandem_fno_expanded.yaml \
    --checkpoint-dir "$output/best" \
    --output "$output/evaluation.json" \
    --visualization-dir "$output/rollout_visualizations" \
    --horizons 1 10 50 100 --visualizations-per-horizon 3

echo "[$(timestamp)] Comparing against scheduled-teacher-forcing model"
.venv/bin/python scripts/compare_rollout_evaluations.py \
    artifacts/tandem_fno_rollout_expanded_v1/evaluation.json \
    "$output/evaluation.json" \
    --output "$output/teacher_forcing_comparison.json" \
    --markdown "$output/teacher_forcing_comparison.md"

echo "[$(timestamp)] Comparing against one-step baseline"
.venv/bin/python scripts/compare_rollout_evaluations.py \
    artifacts/tandem_fno_expanded_v1/evaluation.json \
    "$output/evaluation.json" \
    --output "$output/one_step_comparison.json" \
    --markdown "$output/one_step_comparison.md"

echo "[$(timestamp)] NO_TF_ABLATION_PIPELINE_OK"
