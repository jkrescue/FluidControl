#!/usr/bin/env python3
"""Materialize a reviewed P028/P029 H10 diagnostic source-only closure."""

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


CANONICAL = (
    "scripts/train_tandem_fno.py",
    "scripts/evaluate_tandem_fno.py",
    "scripts/p026_history_inference.py",
    "scripts/p026_state_history.py",
    "src/fluid_control/__init__.py",
    "src/fluid_control/calibrated_checkpoint.py",
    "src/fluid_control/canonical_joint_v1.py",
    "src/fluid_control/dual_fno.py",
    "src/fluid_control/tandem_datapipe.py",
)
PROFILES = {
    "p028": {
        "preparation_status": "P028_H10_SOURCE_CLOSURE_PREPARATION_ONLY",
        "frozen_status": "P028_H10_SOURCE_CLOSURE_FROZEN",
    },
    "p029": {
        "preparation_status": "P029_H10_SOURCE_CLOSURE_PREPARATION_ONLY",
        "frozen_status": "P029_H10_SOURCE_CLOSURE_FROZEN",
        "required_sha256": {
            "scripts/diagnose_p028_short_horizon_comparison.py":
                "f42f74b58f89ca836ce15e7253869b967dbd70a81e09bab6d82ffb47ee5883d5",
            "scripts/run_p028_h10_comparison.py":
                "5eab8e681000d88437257c5253f1c56d941c0a8e7a7cb865012265b12b7b0153",
            "scripts/prepare_p028_h10_comparison_spec.py":
                "ae77975795c44bb99bdad3cdb00cec6a84e0458546eb65f36a6f032466ee722f",
            "src/fluid_control/dual_fno.py":
                "d769f16d649dabd17712b611b463083ef35491f095b2138033480fb029c71dba",
            "training_config.yaml":
                "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9",
        },
    },
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/linked source: {path}")
    return path


def require_exclusive_output(path):
    path = Path(path)
    require(not os.path.lexists(path), "exclusive output already exists or is linked")
    require(path.parent.is_dir() and not path.parent.is_symlink(),
            "output parent must be an existing real directory")
    current = path.parent
    while current != current.parent:
        require(not current.is_symlink(), "output ancestor must not be linked")
        current = current.parent


def publish_noreplace(source, target):
    """Atomically publish one directory without replacing any target object."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 NOREPLACE is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, "exclusive output appeared before publication", target)
        raise OSError(code, os.strerror(code), target)


def source_inventory(args):
    repo = args.repo.resolve()
    entries = {name: regular(repo / name) for name in CANONICAL}
    for name, source in (
        ("scripts/diagnose_p028_short_horizon_comparison.py", args.diagnostic),
        ("scripts/run_p028_h10_comparison.py", args.launcher),
        ("scripts/prepare_p028_h10_comparison_spec.py", args.spec_generator),
        ("training_config.yaml", args.training_config),
    ):
        entries[name] = regular(source)
    inventory = {name: sha(path) for name, path in sorted(entries.items())}
    require(len(inventory) == 13, "exact13 source closure required")
    profile = PROFILES[args.profile]
    for name, digest in profile.get("required_sha256", {}).items():
        require(inventory.get(name) == digest, f"reviewed P029 source differs: {name}")
    return entries, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="p028")
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--spec-generator", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    entries, inventory = source_inventory(args)
    if not args.execute:
        print(json.dumps({"status": profile["preparation_status"],
                          "profile": args.profile,
                          "files_sha256": inventory}, indent=2))
        return
    require_exclusive_output(args.output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.profile}-h10-source-",
                                      dir=args.output.parent))
    try:
        for name, source in entries.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        receipt = {"status": profile["frozen_status"],
                   "profile": args.profile,
                   "files_sha256": inventory,
                   "models_hdf_or_gpu_accessed": False}
        (temporary / "source_manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        for name, digest in inventory.items():
            require(sha(temporary / name) == digest, "copied source differs")
        for path in sorted(temporary.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(temporary, 0o555)
        publish_noreplace(temporary, args.output)
    except BaseException:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            for path in temporary.rglob("*"):
                if path.is_dir():
                    os.chmod(path, 0o755)
                else:
                    os.chmod(path, 0o644)
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"status": receipt["status"],
                      "manifest_sha256": sha(args.output / "source_manifest.json")}))


if __name__ == "__main__":
    main()
