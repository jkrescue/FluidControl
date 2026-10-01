#!/usr/bin/env python3
"""Supervise one CUDA training job and preserve a measured free-memory floor."""

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


def free_memory() -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info(0)
    return int(free), int(total)


def mem_available() -> int:
    # DGX Spark page cache is reclaimable but omitted by cudaMemGetInfo.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--allocator-fraction", type=float, default=0.20)
    parser.add_argument("--margin-gib", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.allocator_fraction <= 1:
        parser.error("a command and allocator-fraction in (0, 1] are required")
    if args.min_free_gib < 20 or args.margin_gib < 0 or args.poll_seconds <= 0:
        parser.error("minimum free memory must be >=20 GiB; other guards must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")

    free, total = free_memory()
    available = mem_available()
    maximum = total * args.allocator_fraction
    required = args.min_free_gib * GIB + maximum + args.margin_gib * GIB
    print(json.dumps({
        "event": "gpu_preflight", "cuda_free_gib": free / GIB,
        "mem_available_gib": available / GIB, "total_gib": total / GIB,
        "allocator_cap_gib": maximum / GIB, "required_available_gib": required / GIB,
        "command": command,
    }), flush=True)
    if available < required:
        print("GPU guard: insufficient initial reclaimable memory", file=sys.stderr, flush=True)
        return 75

    min_available = available
    min_cuda_free = free
    samples = 1
    process = subprocess.Popen(command, start_new_session=True)
    stop_requested = False

    def forward_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    while process.poll() is None:
        if stop_requested:
            terminate_group(process)
            return 143
        free, _ = free_memory()
        available = mem_available()
        min_available = min(min_available, available)
        min_cuda_free = min(min_cuda_free, free)
        samples += 1
        if available < args.min_free_gib * GIB:
            print(json.dumps({"event": "gpu_floor_violation",
                              "cuda_free_gib": free / GIB, "mem_available_gib": available / GIB}),
                  file=sys.stderr, flush=True)
            terminate_group(process)
            return 75
        time.sleep(args.poll_seconds)
    print(json.dumps({
        "event": "gpu_guard_complete",
        "exit_code": int(process.returncode),
        "memory_samples": samples,
        "min_observed_mem_available_gib": min_available / GIB,
        "min_observed_cuda_free_gib": min_cuda_free / GIB,
        "poll_seconds": args.poll_seconds,
        "min_required_mem_available_gib": args.min_free_gib,
    }), flush=True)
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
