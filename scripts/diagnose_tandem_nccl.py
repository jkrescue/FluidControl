#!/usr/bin/env python3
"""Minimal two-rank NCCL collective check independent of model training."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized = False
    try:
        dist.init_process_group("nccl", device_id=device)
        initialized = True
        value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        result = float(value.cpu())
        assert result == 3.0, result
        dist.barrier(device_ids=[local_rank])
        print(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "device": torch.cuda.get_device_name(local_rank),
                    "all_reduce_result": result,
                    "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
                    "NCCL_SHM_DISABLE": os.environ.get("NCCL_SHM_DISABLE"),
                }
            ),
            flush=True,
        )
        if rank == 0:
            print("NCCL_ALL_REDUCE_OK", flush=True)
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
