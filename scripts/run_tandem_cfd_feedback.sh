#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

case_name="${1:-}"
steps="${STEPS:-100}"
output="${OUTPUT_DIR:-artifacts/tandem_mpc_feedback}"
if [[ "$case_name" != mpc_feedback_* ]]; then
    echo 'Usage: run_tandem_cfd_feedback.sh mpc_feedback_NAME' >&2
    exit 2
fi
mkdir -p "$output"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HYDRA_FULL_ERROR=1

set +e
/usr/bin/time -v .venv/bin/torchrun --standalone --nproc_per_node=1 \
    scripts/control_tandem_cfd_feedback.py "$case_name" \
    --steps "$steps" --output "$output" \
    2>&1 | tee "$output/run.log"
code=${PIPESTATUS[0]}
set -e
echo "tandem_cfd_feedback_exit=$code" | tee -a "$output/run.log"
echo "$code" >"$output/exit_code"
exit "$code"
