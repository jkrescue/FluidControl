#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -m)" == aarch64 ]]; then
    echo "Legacy host/GPU runner disabled on DGX Spark; see docs/SPARK_GPU_TRAINING.md" >&2
    exit 2
fi
output_dir="${OUTPUT_DIR:-artifacts/tandem_fno_rollout}"
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

(
  while true; do
    date --iso-8601=seconds
    nvidia-smi \
      --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits \
      | awk -F, '$1 ~ /^[[:space:]]*[01]$/'
    sleep 5
  done
) >> "$output_dir/gpu_usage.csv" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

/usr/bin/time -v \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  scripts/train_tandem_fno_rollout.py \
  --config-name tandem_fno_rollout \
  output_dir="$output_dir" \
  "$@" \
  2>&1 | tee -a "$output_dir/train.log"
code=${PIPESTATUS[0]}
printf 'rollout_training_exit=%s\n' "$code" | tee -a "$output_dir/train.log"
printf '%s\n' "$code" > "$output_dir/exit_code"
exit "$code"
