#!/usr/bin/env python3
"""Bounded lifecycle wrapper for the reviewed B04 Curator-only stage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def available_gib() -> float:
    text = Path("/proc/meminfo").read_text()
    kb = int(next(line.split()[1] for line in text.splitlines() if line.startswith("MemAvailable:")))
    return kb / 2**20


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if available_gib() < 50 or os.statvfs(args.ledger.parent).f_bavail * os.statvfs(args.ledger.parent).f_frsize / 2**30 < 20:
        raise RuntimeError("curator startup memory/disk guard failed")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), str(args.driver), "--spec", str(args.spec),
        "--spec-sha256", args.spec_sha256, "--mode", "curate", "--execute",
    ]
    with args.ledger.open("x", encoding="utf-8") as ledger:
        process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + 1800
        def interrupted(signum, _frame):
            raise InterruptedError(f"received signal {signum}")
        old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while process.poll() is None:
                memory = available_gib()
                disk = os.statvfs(args.ledger.parent).f_bavail * os.statvfs(args.ledger.parent).f_frsize / 2**30
                ledger.write(json.dumps({"monotonic": time.monotonic(), "MemAvailable_GiB": memory, "disk_available_GiB": disk}) + "\n")
                ledger.flush()
                if memory < 22 or disk < 20 or time.monotonic() > deadline:
                    raise RuntimeError("curator runtime resource/deadline guard failed")
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(f"curator child failed: {process.returncode}")
        finally:
            stop(process)
            for signum, handler in old.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
