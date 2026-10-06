"""One bounded P064 exact-file cache advice operation; never modifies file bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


ROOT = Path("/workspace/fluid_control")
AUDIT = ROOT / "artifacts/fcp018_reduced_rate_training_20261005/candidate_audit.json"
AUDIT_SHA = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
OUTPUT = ROOT / "artifacts/fcp064_cache_advice_20261006/exact_files_r1.jsonl"
EXTRA = {
    ROOT / "artifacts/b00_controlled_train_conversion_20261006/b00_projected_ppo_train.h5":
        "45041e79e70043838763e5dd8da0fe9c02dba4cfe6df01356f8dcd1389e32d8f",
    ROOT / "artifacts/fcp026_history_training_k1_20261005/candidate/flow/FNO.0.0.mdlus":
        "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31",
    ROOT / "artifacts/fcp026_history_training_k1_20261005/candidate/flow/checkpoint.0.0.pt":
        "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e",
    ROOT / "artifacts/fcp026_history_training_k1_20261005/candidate/aerodynamic/FNO.0.1.mdlus":
        "e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5",
    ROOT / "artifacts/fcp026_history_training_k1_20261005/candidate/aerodynamic/checkpoint.0.1.pt":
        "ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3",
    Path("/tmp/p064-official-roundtrip-engineering-20261006/flow/FNO.0.0.mdlus"):
        "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31",
    Path("/tmp/p064-official-roundtrip-engineering-20261006/flow/checkpoint.0.0.pt"):
        "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e",
    Path("/tmp/p064-official-roundtrip-engineering-20261006/aerodynamic/FNO.0.1.mdlus"):
        "112cf6ee57350787319fe57c33cce9349652401b6ef4102fa677a75770c12bf2",
    Path("/tmp/p064-official-roundtrip-engineering-20261006/aerodynamic/checkpoint.0.1.pt"):
        "95c16b5760cebe5c5a10ff5fd158aae98d9661f67710c6720cb06bc389a4114a",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def memory():
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {
        key: int(rows[key].split()[0]) * 1024
        for key in ("MemFree", "MemAvailable", "Cached")
    }


def identity(fd):
    row = os.fstat(fd)
    require(stat.S_ISREG(row.st_mode), "exact cache target is not regular")
    return {key: getattr(row, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def exact_files():
    raw = AUDIT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == AUDIT_SHA, "train44 audit differs")
    mapping = json.loads(raw)["train_hdf_sha256"]
    require(len(mapping) == 44, "exact train44 list required")
    result = {}
    for relative, digest in mapping.items():
        path = ROOT / relative
        require(path.resolve().is_relative_to((ROOT / "data/curated").resolve())
                and "/train/" in relative and relative.endswith(".h5"),
                "train44 cache path outside approved roots")
        result[path] = digest
    result.update(EXTRA)
    require(len(result) == 53, "exact 44+9 cache list required")
    return result


def open_exact(path):
    require(path.is_absolute() and path.exists() and not path.is_symlink(),
            "exact cache target absent or symlink")
    return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    mapping = exact_files()
    prepared = {
        "status": "P064_EXACT_CACHE_ADVICE_PREPARATION_ONLY",
        "files": len(mapping),
        "bytes": sum(path.stat().st_size for path in mapping),
        "paths": [str(path) for path in mapping],
        "memory": memory(),
        "data_modified": False,
    }
    if not args.execute:
        print(json.dumps(prepared, sort_keys=True))
        return
    require(memory()["MemAvailable"] >= 50 * 2**30,
            "P064 cache advice requires 50 GiB available startup")
    require(not OUTPUT.exists(), "exclusive P064 cache receipt exists")
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with OUTPUT.open("x") as log:
        def emit(event, **values):
            log.write(json.dumps({"event": event, **values}, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        emit("begin", files=len(mapping), bytes=prepared["bytes"], memory=memory(),
             data_modified=False, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for index, (path, expected) in enumerate(mapping.items(), 1):
            require(time.monotonic() - started < 300, "cache advice deadline")
            fd = open_exact(path)
            try:
                before = identity(fd)
                digest = hashlib.sha256()
                while block := os.read(fd, 1024**2):
                    digest.update(block)
                    require(time.monotonic() - started < 300, "cache advice deadline")
                require(digest.hexdigest() == expected and identity(fd) == before,
                        "exact cache target bytes or identity differ")
                pre = memory()
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                require(identity(fd) == before, "cache advice changed file identity")
                emit("file_advised", index=index, path=str(path), sha256=expected,
                     stat=before, memory_before=pre, memory_after=memory())
            finally:
                os.close(fd)
        emit("complete", elapsed_seconds=time.monotonic() - started, memory=memory(),
             files=len(mapping), data_modified=False)


if __name__ == "__main__":
    main()
