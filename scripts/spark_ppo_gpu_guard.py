#!/usr/bin/env python3
"""Guard canonical PPO CUDA and unified memory with independently measured floors."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import torch

GIB = 1024**3


def cuda_memory() -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info(0)
    return int(free), int(total)


def mem_available() -> int:
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def terminate_group(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def resource_contract(
    *,
    cuda_free: int,
    cuda_total: int,
    available: int,
    allocator_fraction: float,
    min_free_gib: float,
    margin_gib: float,
) -> dict:
    cap = cuda_total * allocator_fraction
    reserve = min_free_gib * GIB
    margin = margin_gib * GIB
    return {
        "allocator_cap_gib": cap / GIB,
        "required_cuda_free_gib": (reserve + cap + margin) / GIB,
        "required_mem_available_gib": (reserve + cap + margin) / GIB,
        "cuda_preflight_pass": cuda_free >= reserve + cap + margin,
        "unified_preflight_pass": available >= reserve + cap + margin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--allocator-fraction", type=float, default=0.20)
    parser.add_argument("--margin-gib", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.allocator_fraction <= 0.20:
        parser.error("a command and allocator-fraction in (0, 0.20] are required")
    if args.min_free_gib < 20 or args.margin_gib < 0 or args.poll_seconds <= 0:
        parser.error("minimum free memory must be >=20 GiB; other guards must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")

    cuda_free, cuda_total = cuda_memory()
    available = mem_available()
    contract = resource_contract(
        cuda_free=cuda_free,
        cuda_total=cuda_total,
        available=available,
        allocator_fraction=args.allocator_fraction,
        min_free_gib=args.min_free_gib,
        margin_gib=args.margin_gib,
    )
    print(
        json.dumps(
            {
                "event": "ppo_gpu_preflight",
                "cuda_free_gib": cuda_free / GIB,
                "mem_available_gib": available / GIB,
                "cuda_total_gib": cuda_total / GIB,
                **contract,
                "command": command,
            }
        ),
        flush=True,
    )
    if not contract["cuda_preflight_pass"] or not contract["unified_preflight_pass"]:
        print("PPO GPU guard: insufficient initial headroom", file=sys.stderr, flush=True)
        return 75

    min_available = available
    min_cuda_free = cuda_free
    process = subprocess.Popen(command, start_new_session=True)
    stop_requested = False

    def forward_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    samples = 1
    reserve = args.min_free_gib * GIB
    while process.poll() is None:
        if stop_requested:
            terminate_group(process)
            return 143
        cuda_free, _ = cuda_memory()
        available = mem_available()
        min_available = min(min_available, available)
        min_cuda_free = min(min_cuda_free, cuda_free)
        samples += 1
        if cuda_free < reserve or available < reserve:
            print(
                json.dumps(
                    {
                        "event": "ppo_gpu_floor_violation",
                        "cuda_free_gib": cuda_free / GIB,
                        "mem_available_gib": available / GIB,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            terminate_group(process)
            return 75
        time.sleep(args.poll_seconds)
    print(
        json.dumps(
            {
                "event": "ppo_gpu_guard_complete",
                "exit_code": int(process.returncode),
                "memory_samples": samples,
                "min_observed_mem_available_gib": min_available / GIB,
                "min_observed_cuda_free_gib": min_cuda_free / GIB,
                "min_required_each_gib": args.min_free_gib,
            }
        ),
        flush=True,
    )
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
