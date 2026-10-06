#!/usr/bin/env python3
"""Thin three-mode resource launcher for the separately approved P029 runner.

This module reuses the reviewed P028 container lifecycle byte-for-byte.  It
only supplies P029 mode-specific approval validation, narrow mounts, commands,
deadlines, and terminal-result checks.  Without ``--execute`` it neither
creates a container nor opens a model or HDF file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys


IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
BASE_LAUNCHER_SHA256 = "62ff300395785291de790e26e5c60d072a89fa43f165f2711747e4b0afd2b8e0"
ENTRY_BASENAME = "train_p029_control_aware_flow.py"
GUARD_BASENAME = "spark_ppo_gpu_guard.py"
PROFILES = {
    "scales": {
        "status": "FC_P029_PARENT_SCALES_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": 0,
        "training_windows": 1368,
        "deadline_seconds": 900,
        "proofs": (),
    },
    "resource-probe": {
        "status": "FC_P029_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": 0,
        "training_windows": 1,
        "deadline_seconds": 900,
        "proofs": ("scales_receipt",),
    },
    "train": {
        "status": "FC_P029_TRAINING_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": 171,
        "training_windows": 1368,
        "deadline_seconds": 14400,
        "proofs": ("scales_receipt", "resource_probe_receipt"),
    },
}
PROTOCOL = {
    "experiment": "FC-P029",
    "optimized_role": "flow",
    "fixed_role": "aerodynamic",
    "horizon": 10,
    "original_window_horizon": 100,
    "training_windows": 1368,
    "optimizer_steps": 171,
    "accumulation_windows": 8,
    "seed": 20261003,
    "learning_rate": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "sampler_order_sha256": "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f",
    "objective": "half_parent_scaled_field_MSE_plus_half_parent_scaled_four_force_MSE",
    "field_weight": 0.5,
    "force_weight": 0.5,
    "scale_windows": 1368,
    "force_loss": True,
    "force_timing": "aero_current_state_and_current_next_action_predicts_next_force",
    "terminal_selection": False,
    "future_truth_inputs": False,
}
FULL_ORDER_SHA256 = PROTOCOL["sampler_order_sha256"]
RESOURCE_PROBE_ORDER_SHA256 = "bde2a44e3c6da9f32809c2f151e65342bc9e255bdab951bc107967bcb7ca891b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def regular_sha_record(value: object, label: str) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            f"exact {label} record required")
    require(isinstance(value["path"], str) and value["path"], f"{label} path absent")
    require(isinstance(value["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
            f"{label} SHA invalid")
    return value


def validate_scales(value: object) -> dict:
    require(isinstance(value, dict) and set(value) == {"field", "force"},
            "exact fixed field/force scales required")
    require(all(type(item) in (int, float) and math.isfinite(item) and item > 0
                for item in value.values()), "fixed scales must be finite and positive")
    return value


def load_base(path: Path):
    require(path.is_file() and not path.is_symlink(), "reviewed base launcher absent")
    digest = __import__("hashlib").file_digest(path.open("rb"), "sha256").hexdigest()
    require(digest == BASE_LAUNCHER_SHA256, "reviewed P028 lifecycle SHA differs")
    definition = importlib.util.spec_from_file_location("p029_reviewed_p028_lifecycle", path)
    require(definition is not None and definition.loader is not None,
            "reviewed base launcher cannot be imported")
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    try:
        definition.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(definition.name, None)
        raise
    return module


def load_spec(base, path: Path, expected_sha256: str, mode: str) -> dict:
    require(mode in PROFILES, "unknown P029 launcher mode")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "invalid exact spec SHA")
    require(path.is_file() and not path.is_symlink(), "spec must be a regular file")
    require(base.sha256(path) == expected_sha256, "exact spec SHA differs")
    spec = json.loads(path.read_text())
    require(spec.get("status") == "FC_P029_EXECUTION_APPROVED"
            and spec.get("mode") == mode, "separate exact P029 mode approval required")
    require(spec.get("protocol") == PROTOCOL, "exact P029 numerical protocol differs")
    require(spec.get("resources", {}).get("allocator_fraction") == 0.06,
            "fixed allocator fraction differs")
    require(set(spec.get("data", {})) == {"base", "train8", "train16"},
            "exact train families required")
    require(isinstance(spec.get("source_root"), str) and spec["source_root"],
            "source root absent")
    require(isinstance(spec.get("source_sha256"), dict) and spec["source_sha256"],
            "source closure is empty")
    for relative, digest in spec["source_sha256"].items():
        require(isinstance(relative, str) and relative
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "source closure record invalid")
    for key in ("config", "parent_manifest", "train_audit"):
        regular_sha_record(spec.get(key), key)
    for family in ("base", "train8", "train16"):
        record = spec["data"][family]
        require(isinstance(record, dict) and isinstance(record.get("root"), str),
                f"{family} root absent")
        for key in ("manifest_sha256", "normalization_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", record.get(key, "")) is not None,
                    f"{family} {key} invalid")
        require(isinstance(record.get("train_files"), dict) and record["train_files"],
                f"{family} train inventory absent")
    profile = PROFILES[mode]
    for proof in profile["proofs"]:
        regular_sha_record(spec.get(proof), proof)
    if mode == "scales":
        require("scales_receipt" not in spec and "resource_probe_receipt" not in spec,
                "scales approval must not depend on later receipts")
        require("fixed_scales" not in spec, "scales approval cannot predeclare fitted scales")
    else:
        validate_scales(spec.get("fixed_scales"))
        if mode == "resource-probe":
            require("resource_probe_receipt" not in spec,
                    "resource probe cannot depend on its own receipt")
    return spec


def named_source(spec: dict, basename: str) -> Path:
    found = [Path(spec["source_root"]) / relative for relative in spec["source_sha256"]
             if Path(relative).name == basename]
    require(len(found) == 1, f"exactly one {basename} is required")
    return found[0]


def readonly_mounts(base, spec: dict, spec_path: Path) -> list[tuple[Path, Path, bool]]:
    mounts: list[tuple[Path, Path, bool]] = [(spec_path, spec_path, True)]
    source_root = Path(spec["source_root"]).resolve()
    for relative in spec["source_sha256"]:
        source = (source_root / relative).resolve()
        require(source.is_relative_to(source_root), "source closure escapes root")
        mounts.append((source, source, True))
    for key in ("config", "train_audit") + tuple(PROFILES[spec["mode"]]["proofs"]):
        path = Path(spec[key]["path"])
        mounts.append((path, path, True))
    mounts.append((base.GUARD_PATH, base.GUARD_PATH, True))
    manifest = Path(spec["parent_manifest"]["path"])
    mounts.append((manifest.parent, manifest.parent, True))
    for family in ("base", "train8", "train16"):
        root = Path(spec["data"][family]["root"])
        for child in (root / "manifest.json", root / "normalization.json", root / "train"):
            mounts.append((child, Path("/workspace") / family / child.name, True))
    unique: dict[tuple[str, str], tuple[Path, Path, bool]] = {}
    for source, target, read_only in mounts:
        key = (str(source.resolve()), str(target))
        unique[key] = (source.resolve(), target, read_only)
    return list(unique.values())


def pythonpath(spec: dict) -> str:
    root = Path(spec["source_root"])
    return str(root / "scripts") + ":" + str(root / "src")


def create_command(base, spec: dict, spec_path: Path, spec_sha: str,
                   output: Path) -> list[str]:
    mode = spec["mode"]
    profile = PROFILES[mode]
    entry = named_source(spec, ENTRY_BASENAME)
    command = [
        "docker", "create", "--name", base.CONTAINER,
        "--cidfile", str((output / "evidence" / "container.cid").resolve()),
        "--gpus", "device=0", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", "--cpus", "8",
        "--memory", "12g", "--memory-swap", "12g", "--pids-limit", "1024",
        "--shm-size", "2g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "XDG_CACHE_HOME=/tmp/cache",
        "-e", "LOCAL_CACHE=/tmp/physicsnemo-cache", "-e", "WARP_CACHE_PATH=/tmp/warp",
        "-e", f"PYTHONPATH={pythonpath(spec)}",
    ]
    for source, target, _ in readonly_mounts(base, spec, spec_path):
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    command += ["--mount", f"type=bind,src={output.resolve()},dst={output.resolve()}"]
    command += [
        IMAGE, "python", "-u", str(base.GUARD_PATH), "--min-free-gib", "20",
        "--allocator-fraction", ".06", "--margin-gib", "4", "--poll-seconds", "2", "--",
        "timeout", "-k", "20", str(profile["deadline_seconds"]),
        "python", "-u", str(entry), "--spec", str(spec_path.resolve()),
        "--spec-sha256", spec_sha, "--mode", mode,
        "--output", str((output / "payload").resolve()), "--execute",
    ]
    return command


def validate_result(result: dict, spec: dict) -> None:
    mode = spec["mode"]
    profile = PROFILES[mode]
    require(result.get("status") == profile["status"] and result.get("mode") == mode,
            "P029 result status/mode differs")
    require(type(result.get("optimizer_steps")) is int
            and result["optimizer_steps"] == profile["optimizer_steps"],
            "P029 optimizer steps differ")
    require(type(result.get("training_windows")) is int
            and result["training_windows"] == profile["training_windows"],
            "P029 training windows differ")
    require(result.get("scientific_admission") is False
            and result.get("validation_accessed") is False
            and result.get("frozen_test_accessed") is False
            and result.get("source_spec") == spec, "P029 isolation/source identity differs")
    require(result.get("optimizer_created") is (mode == "train")
            and result.get("model_saved") is (mode == "train"),
            "P029 optimizer/save identity differs")
    expected_order = (RESOURCE_PROBE_ORDER_SHA256 if mode == "resource-probe"
                      else FULL_ORDER_SHA256)
    require(result.get("sampler_order_sha256") == expected_order,
            "P029 observed sampler identity differs")
    if mode == "scales":
        validate_scales(result.get("fixed_scales"))
        require(isinstance(result.get("records"), list) and len(result["records"]) == 1368,
                "parent scale record count differs")
    else:
        require(result.get("fixed_scales") == spec["fixed_scales"],
                "approved fixed scales differ")
    if mode == "train":
        require(isinstance(result.get("records"), list) and len(result["records"]) == 171,
                "terminal update records differ")
        require(result.get("official_fresh_reload_verified") is True,
                "official fresh reload proof absent")


class OuterDeadline(RuntimeError):
    """Raised only to force the reviewed lifecycle into its cleanup finally."""


def execute(base, spec: dict, spec_path: Path, spec_sha: str, output: Path) -> None:
    mode = spec["mode"]
    base.CONTAINER = f"fcp029-{mode}-20261006"
    base.ENTRY_BASENAME = ENTRY_BASENAME
    base.readonly_mounts = lambda current, path: readonly_mounts(base, current, path)
    base.create_command = lambda current, path, digest, root: create_command(
        base, current, path, digest, root)
    base.validate_training_result = validate_result

    def expired(_signum: int, _frame: object) -> None:
        raise OuterDeadline("P029 launcher outer deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(PROFILES[mode]["deadline_seconds"] + 60)
    try:
        base.execute(spec, spec_path, spec_sha, output)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-launcher", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--mode", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = load_base(args.base_launcher.resolve())
    spec_path = args.spec.resolve()
    output = args.output.resolve()
    spec = load_spec(base, spec_path, args.spec_sha256, args.mode)
    base.CONTAINER = f"fcp029-{args.mode}-20261006"
    base.ENTRY_BASENAME = ENTRY_BASENAME
    base.readonly_mounts = lambda current, path: readonly_mounts(base, current, path)
    base.create_command = lambda current, path, digest, root: create_command(
        base, current, path, digest, root)
    base.validate_training_result = validate_result
    if not args.execute:
        print(json.dumps({
            "status": f"FC_P029_{args.mode.upper().replace('-', '_')}_LAUNCH_PREPARED_NO_DOCKER_NO_GPU",
            "mode": args.mode,
            "image": IMAGE,
            "deadline_seconds": PROFILES[args.mode]["deadline_seconds"],
            "command": create_command(base, spec, spec_path, args.spec_sha256, output),
        }))
        return
    execute(base, spec, spec_path, args.spec_sha256, output)


if __name__ == "__main__":
    main()
