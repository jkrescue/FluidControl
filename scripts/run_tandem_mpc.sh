#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

config="${CONFIG:-conf/tandem_mpc.yaml}"
output_dir="${OUTPUT_DIR:-artifacts/tandem_mpc}"
mkdir -p "$output_dir"

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
    scripts/control_tandem_mpc.py --config "$config" "$@" \
    2>&1 | tee "$output_dir/run.log"
code=${PIPESTATUS[0]}
set -e

echo "tandem_mpc_exit=$code" | tee -a "$output_dir/run.log"
echo "$code" >"$output_dir/exit_code"
exit "$code"
