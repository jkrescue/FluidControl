"""Serialize the preparation-only y-reflection spec from the reviewed B approval."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pair(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_root = args.source_root.resolve()
    base_path = repo / "docs/FC_P064_ARM_B_TRAINING_APPROVAL_20261006.json"
    base = json.loads(base_path.read_text())
    worker = source_root / "scripts/train_p064_y_reflection.py"
    supervisor = source_root / "scripts/supervise_p064_y_reflection.py"
    reflection = source_root / "src/fluid_control/p064_y_reflection.py"
    planned_output = repo / "artifacts/p064_y_reflection_paired_20261007"
    argv = list(base["argv"])
    argv[2] = str(worker)
    argv[argv.index("--source-root") + 1] = str(source_root)
    argv[argv.index("--schedule-adapter") + 1] = str(
        source_root / "src/fluid_control/p064_controlled_aero_ab.py"
    )
    argv[argv.index("--output") + 1] = str(planned_output)
    argv[argv.index("--arm") + 1] = "B"
    if "--execute" in argv:
        argv.remove("--execute")
    argv += [
        "--reflection-module", str(reflection),
        "--reflection-module-sha256", sha(reflection),
        "--execute",
    ]
    inputs = {
        "base_approval": pair(base_path),
        "physical_precheck_review": pair(repo / "docs/P064_Y_REFLECTION_PHYSICAL_CPU_PRECHECK_REVIEW_20261007.md"),
        "cpu_implementation_review": pair(repo / "docs/P064_Y_REFLECTION_CPU_IMPLEMENTATION_REVIEW_20261007.md"),
        "config": pair(value(argv, "--config")),
        "parent_manifest": pair(value(argv, "--parent-manifest")),
        "parent_order": pair(value(argv, "--parent-order")),
        "schedule_adapter": pair(value(argv, "--schedule-adapter")),
        "b00_manifest": pair(value(argv, "--b00-manifest")),
        "b00_conversion_receipt": pair(value(argv, "--b00-conversion-receipt")),
        "b00_source_hdf": pair(value(argv, "--b00-source-hdf")),
        "candidate_audit": pair(value(argv, "--candidate-audit")),
        "base_manifest": pair(Path(value(argv, "--base-data-root")) / "manifest.json"),
        "base_normalization": pair(Path(value(argv, "--base-data-root")) / "normalization.json"),
        "train8_manifest": pair(Path(value(argv, "--train8-data-root")) / "manifest.json"),
        "train8_normalization": pair(Path(value(argv, "--train8-data-root")) / "normalization.json"),
        "train16_manifest": pair(Path(value(argv, "--train16-data-root")) / "manifest.json"),
        "train16_normalization": pair(Path(value(argv, "--train16-data-root")) / "normalization.json"),
    }
    protocol = {
        "parent": "FC-P026-K1",
        "schedule": "original FC-P064 arm B: 192 original plus 64 b00",
        "seed": 20261003,
        "original_training_windows": 256,
        "transformed_window_equivalents": 512,
        "accumulation_original_windows": 8,
        "optimizer_steps": 32,
        "original_loss_weight": 0.5,
        "reflected_loss_weight": 0.5,
        "h1_weight_within_each_branch": 0.5,
        "ar_weight_within_each_branch": 0.5,
        "learning_rate": 1.5625e-7,
        "expected_training_flow_calls": 51200,
        "expected_training_aerodynamic_calls": 5120,
        "diagnostic_panels": "unchanged original unreflected panel at counts 0 and 256",
        "save": "final only",
        "selection": False,
    }
    resources = {
        "gpu": 0, "cuda_allocator_bytes": 16 * 2**30,
        "systemd_memory_bytes": 24 * 2**30, "systemd_swap_bytes": 0,
        "systemd_cpu_quota_percent": 800, "systemd_tasks_max": 2048,
        "startup_mem_available_gib": 50, "runtime_mem_available_gib": 22,
        "physical_reserve_gib": 20, "inner_deadline_seconds": 3600,
        "outer_deadline_seconds": 3660, "stop_seconds": 20,
    }
    source_manifest = source_root / "source_manifest.json"
    spec = {
        "status": "P064_Y_REFLECTION_PAIRED_TRAINING_PREPARATION_ONLY",
        "execution_authorized": False,
        "lead_authorization": "CPU preparation only; GPU training and CFD are not authorized",
        "scientific_scope": "One fixed physical-y-reflection paired-loss candidate; no sweep or selection.",
        "runtime": {
            "kind": "host_venv_curator_py312",
            "python": str(repo / ".venv-curator-py312/bin/python"),
            "python_resolved": "/home/USER/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/bin/python3.12",
            "python_sha256": "001718d5edf61e6fbc3642c9668def38d2bf28cc320e75196e6a95d49614cca8",
            "pythonpath": f"{source_root / 'src'}:{source_root / 'scripts'}",
        },
        "supervisor": pair(supervisor), "worker": pair(worker), "reflection": pair(reflection),
        "source_root": str(source_root),
        "source_manifest": pair(source_manifest), "inputs": inputs,
        "protocol": protocol, "resources": resources, "worker_argv": argv,
        "planned_unit": "fluid-control-p064-y-reflection-paired-20261007.service",
        "planned_output": str(planned_output),
        "preflight": {
            "helper_and_runner_cpu_tests": "14 PASS",
            "physical_precheck": "ACCEPT",
            "full_source_runtime_input_hashes": "PASS",
            "worker_noexecute": "PASS",
            "supervisor_dryrun": "PASS after serialization",
            "output_absent": not planned_output.exists(),
            "gpu_training_performed": False,
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
