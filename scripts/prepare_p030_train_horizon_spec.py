#!/usr/bin/env python3
"""Prepare a non-authorizing P030 K1-vs-P029 start0/H100 spec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PENDING = "P030_TRAIN_HORIZON_DIAGNOSTIC_PREPARATION_ONLY_NOT_APPROVED"
APPROVED = "P030_TRAIN_HORIZON_DIAGNOSTIC_EXECUTION_APPROVED"
CLOSURE = "P030_TRAIN_HORIZON_SOURCE_CLOSURE_FROZEN"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def prepare(base: dict, source_manifest: dict, source_root: Path,
            reload_receipt: dict, reload_path: Path, reload_sha256: str) -> dict:
    require(base.get("status") == "P029_MATCHED_H10_COMPARISON_EXECUTION_APPROVED"
            and base.get("execution_authorized") is True
            and base.get("comparison_profile") == "p029", "reviewed P029 base spec differs")
    require(set(base.get("candidates", {})) == {"1", "4"}, "base P026 arms differ")
    require(base.get("p029_terminal_proof", {}).get("reviewed_by_lead") is True,
            "P029 terminal proof was not reviewed")
    require(source_manifest.get("status") == CLOSURE, "P030 closure status differs")
    files = source_manifest.get("files_sha256")
    require(isinstance(files, dict) and files, "P030 source closure is empty")
    for name in files:
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
                "source closure member path escapes root")
    require(reload_receipt.get("status") == "FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"
            and reload_receipt.get("official_dual_reload_verified") is True
            and reload_receipt.get("gpu_used") is False
            and reload_receipt.get("optimizer_created") is False
            and reload_receipt.get("model_saved") is False
            and reload_receipt.get("scientific_admission") is False,
            "P029 official CPU reload proof differs")
    result = {
        key: base[key] for key in (
            "config", "data", "source_phase_mapping", "train16_predeclaration", "train_audit",
        )
    }
    result.update({
        "status": PENDING,
        "intended_authorized_status": APPROVED,
        "execution_authorized": False,
        "source_files": [
            {"path": str(source_root / name), "sha256": digest}
            for name, digest in sorted(files.items())
        ],
        "candidates": {
            "k1": base["candidates"]["1"],
            "p029": base["p029_candidate"],
        },
        "p029_terminal_proof": base["p029_terminal_proof"],
        "p029_official_cpu_reload": {
            "path": str(reload_path.resolve()), "sha256": reload_sha256,
            "status": reload_receipt["status"],
        },
        "comparison_contract": {
            "split": "train", "starts": [0], "windows": 44, "rollout_steps": 100,
            "report_leads": [1, 10, 25, 50, 100], "selection_performed": False,
        },
        "resource_contract": {
            "official_image": "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e",
            "gpu": 0, "allocator_fraction": 0.06, "memory_gib": 12,
            "deadline_seconds": 900, "startup_memfree_gib": 30,
            "startup_memavailable_gib": 50, "runtime_memory_floor_gib": 20,
            "cuda_free_floor_gib": 20,
        },
        "lead_review": {"reviewed": False, "execution_authorized": False},
        "scientific_admission": False,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-p029-spec", type=Path, required=True)
    parser.add_argument("--base-p029-spec-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--p029-cpu-reload", type=Path, required=True)
    parser.add_argument("--p029-cpu-reload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    require(sha(args.base_p029_spec) == args.base_p029_spec_sha256, "base spec SHA differs")
    require(sha(args.source_manifest) == args.source_manifest_sha256, "source manifest SHA differs")
    require(sha(args.p029_cpu_reload) == args.p029_cpu_reload_sha256, "CPU reload SHA differs")
    result = prepare(read(args.base_p029_spec), read(args.source_manifest),
                     args.source_manifest.parent.resolve(), read(args.p029_cpu_reload),
                     args.p029_cpu_reload, args.p029_cpu_reload_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": PENDING, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
