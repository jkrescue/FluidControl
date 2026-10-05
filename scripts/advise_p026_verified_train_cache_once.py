"""One approved clean-cache pass; no writes to CFD/HDF/model files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT = Path("/workspace/fluid_control")
AUDIT = ROOT / "artifacts/fcp018_reduced_rate_training_20261005/candidate_audit.json"
AUDIT_SHA = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
FAMILIES = {
    "tandem_cylinders_matched_start_full40_dev30_v1": 20,
    "tandem_cylinders_dynamic_train8_v1": 8,
    "tandem_cylinders_directppo_train16_v1": 16,
}
OUTPUT = ROOT / "artifacts/fcp026_history_training_k1_20261005/cache_advice_20261006_r1.jsonl"
OUTPUTS = {1: OUTPUT, 4: ROOT / "artifacts/fcp026_history_training_k4_20261005/cache_advice_20261006_r1.jsonl"}
RECEIPT_NAMES = tuple(f"cache_advice_20261006_r{i}.jsonl" for i in range(1, 5))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def memory():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {key: int(values[key].split()[0]) * 1024 for key in ("MemFree", "MemAvailable")}


def guard(values, initial=False, before_hash=False):
    floor = 20.75 if initial else (20.5 if before_hash else 20)
    require(values["MemFree"] >= floor * 2**30 and
            values["MemAvailable"] >= 20 * 2**30, "unchanged dual20 / approved operational headroom")


def identity(fd):
    value = os.fstat(fd)
    require(stat.S_ISREG(value.st_mode), "not a regular file")
    return {key: getattr(value, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def open_confined(relative):
    parts = Path(relative).parts
    require(not Path(relative).is_absolute() and ".." not in parts, "path escape")
    require(all(not p.is_symlink() for p in (ROOT, *ROOT.parents)), "root symlink")
    directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--history-k", type=int, choices=(1, 4), default=1)
    parser.add_argument("--receipt-name", choices=RECEIPT_NAMES, default=OUTPUT.name)
    args = parser.parse_args(argv)
    if args.history_k == 4 and args.receipt_name != RECEIPT_NAMES[0]:
        parser.error("K4 preparation permits only the fixed r1 receipt")
    return args


def main():
    args = parse_args()
    output = OUTPUTS[args.history_k].with_name(args.receipt_name)
    raw = AUDIT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == AUDIT_SHA, "approved44 source SHA differs")
    files = json.loads(raw)["train_hdf_sha256"]
    require(len(files) == 44, "exact44 only")
    counts = dict.fromkeys(FAMILIES, 0)
    for name, digest in files.items():
        parts = Path(name).parts
        require(len(parts) == 5 and parts[:2] == ("data", "curated") and
                parts[2] in FAMILIES and parts[3] == "train" and parts[4].endswith(".h5") and
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "outside approved train scope")
        counts[parts[2]] += 1
    require(counts == FAMILIES, "family counts differ")
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NO_ADVICE", files=44,
                              audit_sha256=AUDIT_SHA, output=str(output), memory=memory())))
        return
    started = time.monotonic()
    with output.open("x") as log:
        def emit(event, **values):
            log.write(json.dumps(dict(event=event, timestamp=time.time(), **values), sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        emit("begin", audit_sha256=AUDIT_SHA, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             files=44, memory=memory(), data_writes=False, history_k=args.history_k,
             output=str(output))
        try:
            guard(memory(), initial=True)
            for index, (name, expected) in enumerate(sorted(files.items()), 1):
                before_memory = memory()
                guard(before_memory, before_hash=True)
                require(time.monotonic() - started < 300, "bounded300s deadline")
                fd = open_confined(name)
                try:
                    before = identity(fd)
                    emit("file_begin", index=index, path=name, expected_sha256=expected,
                         stat=before, memory=before_memory)
                    digest = hashlib.sha256()
                    while block := os.read(fd, 1024**2):
                        digest.update(block)
                        guard(memory())
                        require(time.monotonic() - started < 300, "bounded300s deadline")
                    actual = digest.hexdigest()
                    after_hash = identity(fd)
                    require(before == after_hash, "file identity changed while hashing")
                    require(actual == expected, "file SHA differs; no advice")
                    before_advice = memory()
                    guard(before_advice)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    after_advice = identity(fd)
                    require(after_advice == before, "file identity changed after advice")
                    after_memory = memory()
                    emit("file_advised", index=index, path=name, sha256=actual,
                         stat_before=before, stat_after_hash=after_hash, stat_after_advice=after_advice,
                         memory_before=before_memory, memory_before_advice=before_advice,
                         memory_after=after_memory)
                    guard(after_memory)
                finally:
                    os.close(fd)
            emit("complete", files=44, elapsed_seconds=time.monotonic()-started, memory=memory())
        except BaseException as error:
            emit("aborted", error=repr(error), elapsed_seconds=time.monotonic()-started, memory=memory())
            raise


if __name__ == "__main__":
    main()
