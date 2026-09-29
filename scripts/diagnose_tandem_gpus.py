#!/usr/bin/env python3
"""Read-only CUDA topology and per-device smoke checks before NCCL retry."""

from __future__ import annotations

import json

import torch


def main() -> None:
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": torch.cuda.nccl.version() if torch.distributed.is_nccl_available() else None,
        "devices": [],
        "peer_access": {},
    }
    assert report["cuda_available"]
    assert report["device_count"] >= 2

    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            left = torch.randn((1024, 1024), device=f"cuda:{index}")
            right = torch.randn((1024, 1024), device=f"cuda:{index}")
            result = left @ right
            torch.cuda.synchronize(index)
            checksum = float(result[0, 0].cpu())
        report["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
                "matmul_checksum": checksum,
            }
        )

    for source in range(2):
        for target in range(2):
            if source != target:
                report["peer_access"][f"{source}->{target}"] = bool(
                    torch.cuda.can_device_access_peer(source, target)
                )

    print(json.dumps(report, indent=2))
    print("CUDA_DEVICE_CHECK_OK")


if __name__ == "__main__":
    main()
