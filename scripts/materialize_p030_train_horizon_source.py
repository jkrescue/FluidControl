#!/usr/bin/env python3
"""Materialize the reviewed P029 13-file base plus four P030 source files."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

PREPARATION = "P030_TRAIN_HORIZON_SOURCE_PREPARATION_ONLY"
FROZEN = "P030_TRAIN_HORIZON_SOURCE_CLOSURE_FROZEN"
CORE_SHA256 = "90d99902d87fb07494eb87e7e5f91c6420e70c5ee773dd3b55af77e0541a3560"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/linked source: {path}")
    return path


def require_exclusive_output(path):
    path = Path(path)
    require(not os.path.lexists(path), "exclusive output already exists or is linked")
    require(path.parent.is_dir() and not path.parent.is_symlink(), "real output parent required")
    current = path.parent
    while current != current.parent:
        require(not current.is_symlink(), "output ancestor must not be linked")
        current = current.parent


def publish_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 NOREPLACE is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, "exclusive target appeared", target)
        raise OSError(code, os.strerror(code), target)


def inventory(args):
    base_manifest = regular(args.base_source_manifest)
    require(sha(base_manifest) == args.base_source_manifest_sha256,
            "base13 source manifest SHA differs")
    base = json.loads(base_manifest.read_text())
    require(base.get("status") == "P029_H10_SOURCE_CLOSURE_FROZEN",
            "reviewed P029 source status differs")
    base_files = base.get("files_sha256")
    require(isinstance(base_files, dict) and len(base_files) == 13,
            "exact reviewed base13 closure required")
    entries = {}
    for name, digest in base_files.items():
        source = regular(base_manifest.parent / name)
        require(sha(source) == digest, f"base13 source differs: {name}")
        entries[name] = source
    additions = {
        "scripts/p030_train_horizon_core.py": regular(args.core),
        "scripts/diagnose_p030_train_horizon.py": regular(args.diagnostic),
        "scripts/run_p030_train_horizon.py": regular(args.launcher),
        "scripts/prepare_p030_train_horizon_spec.py": regular(args.spec_generator),
    }
    require(sha(additions["scripts/p030_train_horizon_core.py"]) == CORE_SHA256,
            "accepted P030 core differs")
    require(not set(entries) & set(additions), "P030 additions collide with base13")
    entries.update(additions)
    hashes = {name: sha(path) for name, path in sorted(entries.items())}
    require(len(hashes) == 17, "exact base13 plus four P030 sources required")
    return entries, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source-manifest", type=Path, required=True)
    parser.add_argument("--base-source-manifest-sha256", required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--spec-generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    entries, hashes = inventory(args)
    if not args.execute:
        print(json.dumps({"status": PREPARATION, "files_sha256": hashes}, indent=2))
        return
    require_exclusive_output(args.output)
    temporary = Path(tempfile.mkdtemp(prefix=".p030-source-", dir=args.output.parent))
    try:
        for name, source in entries.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        receipt = {"status": FROZEN, "base_source_manifest_sha256":
                   args.base_source_manifest_sha256, "files_sha256": hashes,
                   "models_hdf_or_gpu_accessed": False}
        (temporary / "source_manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        for name, digest in hashes.items():
            require(sha(temporary / name) == digest, "copied source differs")
        for path in sorted(temporary.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(temporary, 0o555)
        publish_noreplace(temporary, args.output)
    except BaseException:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            for path in temporary.rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"status": FROZEN,
                      "manifest_sha256": sha(args.output / "source_manifest.json")}))


if __name__ == "__main__":
    main()
