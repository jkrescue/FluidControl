#!/usr/bin/env python3
"""Advise Linux to release page cache for completed project CFD case files.

No files are read, edited, removed, or truncated. This is only a cache hint
for DGX Spark's unified-memory system; rerunning CFD or Curator is unnecessary.
Dry-run is the default. Do not execute while a listed case is being written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import time
from pathlib import Path

CASES = Path(__file__).resolve().parents[1] / "cfd/tandem_cylinders/cases"
NAME = re.compile(r"[a-zA-Z0-9_]+\Z")


def case_path(name: str) -> Path:
    if not NAME.fullmatch(name):
        raise ValueError(f"invalid case name: {name!r}")
    case = CASES / name
    if not case.is_dir() or case.is_symlink() or not (case / "case_config.json").is_file():
        raise ValueError(f"not a local project CFD case: {name}")
    log = case / "log.pimpleFoam"
    if not log.is_file() or not log.read_text(encoding="utf-8", errors="replace").rstrip().endswith("End"):
        raise ValueError(f"CFD case has not ended cleanly: {name}")
    return case


def candidates(case: Path, min_age: float) -> list[Path]:
    now = time.time()
    files = []
    for path in case.rglob("*"):
        try:
            meta = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(meta.st_mode) and now - meta.st_mtime >= min_age:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--min-age-seconds", type=float, default=120.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.min_age_seconds < 60:
        parser.error("minimum file age must be at least 60 seconds")
    if len(args.cases) != len(set(args.cases)):
        parser.error("duplicate cases")
    paths = [case_path(name) for name in args.cases]
    files = [path for case in paths for path in candidates(case, args.min_age_seconds)]
    before = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    bytes_total = sum(path.stat().st_size for path in files)
    advised = 0
    errors = []
    if args.execute:
        if not hasattr(os, "posix_fadvise"):
            raise RuntimeError("os.posix_fadvise is unavailable on this host")
        for path in files:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                advised += 1
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
    after = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    print(json.dumps({
        "status": "PAGE_CACHE_HINT_COMPLETE" if args.execute and not errors else
                  "PAGE_CACHE_HINT_PARTIAL" if args.execute else "PAGE_CACHE_HINT_DRY_RUN",
        "cases": args.cases,
        "files_eligible": len(files),
        "eligible_gib": bytes_total / 1024**3,
        "files_advised": advised,
        "memfree_before_gib": before / 1024**3,
        "memfree_after_gib": after / 1024**3,
        "errors": errors[:20],
        "semantics": "POSIX_FADV_DONTNEED on closed, completed project files; no data deletion or rewrite",
    }))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
