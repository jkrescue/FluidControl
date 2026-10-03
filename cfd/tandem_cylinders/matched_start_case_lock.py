#!/usr/bin/env python3
"""Atomically acquire and update a persistent matched-start case-run lock."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path


def _write_state(lock: Path, payload: dict) -> None:
    state = lock / "state.json"
    temporary = lock / ".state.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state)


def acquire(lock: Path, case: str) -> None:
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"case run lock already exists: {lock}") from error
    _write_state(
        lock,
        {
            "status": "ACQUIRED",
            "case": case,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )


def update(lock: Path, status: str, exit_code: int) -> None:
    if not (lock / "state.json").is_file():
        raise ValueError(f"case run lock is not initialized: {lock}")
    current = json.loads((lock / "state.json").read_text(encoding="utf-8"))
    if current.get("status") != "ACQUIRED":
        raise ValueError(f"case run lock is already terminal: {current.get('status')}")
    current.update(
        {
            "status": status,
            "exit_code": exit_code,
            "finished_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write_state(lock, current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--lock", required=True, type=Path)
    acquire_parser.add_argument("--case", required=True)
    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--lock", required=True, type=Path)
    update_parser.add_argument("--status", required=True, choices=("COMPLETED", "FAILED"))
    update_parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.lock, args.case)
    else:
        update(args.lock, args.status, args.exit_code)


if __name__ == "__main__":
    main()
