"""Bounded host launcher for the reviewed P064 y-reflection candidate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/workspace/fluid_control")
PYTHON = ROOT / ".venv-curator-py312/bin/python"
PENDING = "P064_Y_REFLECTION_PAIRED_TRAINING_PREPARATION_ONLY"
APPROVED = "P064_Y_REFLECTION_PAIRED_TRAINING_EXECUTION_APPROVED"
RESOURCES = {
    "gpu": 0,
    "cuda_allocator_bytes": 16 * 2**30,
    "systemd_memory_bytes": 24 * 2**30,
    "systemd_swap_bytes": 0,
    "systemd_cpu_quota_percent": 800,
    "systemd_tasks_max": 2048,
    "startup_mem_available_gib": 50,
    "runtime_mem_available_gib": 22,
    "physical_reserve_gib": 20,
    "inner_deadline_seconds": 3600,
    "outer_deadline_seconds": 3660,
    "stop_seconds": 20,
}
UNIT = {
    "Type": "exec",
    "MemoryMax": str(24 * 2**30),
    "MemorySwapMax": "0",
    "CPUQuotaPerSecUSec": "8s",
    "TasksMax": "2048",
    "RuntimeMaxUSec": "1h 1min",
    "TimeoutStopUSec": "20s",
    "KillMode": "control-group",
}
PROTOCOL = {
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


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(item):
    path = Path(item["path"])
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT), "bound path escapes project")
    require(path.is_file() and not path.is_symlink(), "bound regular file required")
    require(sha256(path) == item["sha256"], "bound file SHA differs")
    return path


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 2**30
    raise ValueError("MemAvailable missing")


def validate_spec(spec, allow_pending):
    allowed = {APPROVED} | ({PENDING} if allow_pending else set())
    require(spec.get("status") in allowed, "approval status differs")
    require(spec.get("execution_authorized") is (spec["status"] == APPROVED), "authorization differs")
    require(spec.get("resources") == RESOURCES, "resource contract differs")
    require(spec.get("protocol") == PROTOCOL, "scientific protocol differs")
    runtime = spec.get("runtime")
    require(runtime == {
        "kind": "host_venv_curator_py312",
        "python": str(PYTHON),
        "python_resolved": "/home/USER/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/bin/python3.12",
        "python_sha256": "001718d5edf61e6fbc3642c9668def38d2bf28cc320e75196e6a95d49614cca8",
        "pythonpath": f"{Path(spec['source_root']).resolve() / 'src'}:{Path(spec['source_root']).resolve() / 'scripts'}",
    }, "runtime identity differs")
    require(PYTHON.is_symlink() and str(PYTHON.resolve()) == runtime["python_resolved"], "runtime link differs")
    require(sha256(PYTHON.resolve()) == runtime["python_sha256"], "runtime binary differs")
    supervisor = bind(spec["supervisor"])
    require(supervisor.resolve() == Path(__file__).resolve(), "executed supervisor differs")
    worker = bind(spec["worker"])
    reflection = bind(spec["reflection"])
    source_manifest = bind(spec["source_manifest"])
    source_root = Path(spec["source_root"]).resolve()
    require(source_root.is_dir() and source_root.is_relative_to(ROOT), "source root differs")
    source_map = json.loads(source_manifest.read_text())
    require(isinstance(source_map, dict) and len(source_map) == 434, "434 source closure required")
    for relative, expected in source_map.items():
        candidate = source_root / relative
        require(candidate.is_file() and not candidate.is_symlink(), f"source missing: {relative}")
        require(sha256(candidate) == expected, f"source differs: {relative}")
    inputs = {name: bind(item) for name, item in spec["inputs"].items()}
    require(set(inputs) == {
        "base_approval", "physical_precheck_review", "cpu_implementation_review",
        "config", "parent_manifest", "parent_order", "schedule_adapter",
        "b00_manifest", "b00_conversion_receipt", "b00_source_hdf",
        "candidate_audit", "base_manifest", "base_normalization",
        "train8_manifest", "train8_normalization", "train16_manifest",
        "train16_normalization",
    }, "exact input binding set differs")
    argv = spec.get("worker_argv")
    base_approval = json.loads(inputs["base_approval"].read_text())
    expected_argv = list(base_approval["argv"])
    expected_argv[2] = str(worker)
    expected_argv[expected_argv.index("--source-root") + 1] = str(source_root)
    expected_argv[expected_argv.index("--schedule-adapter") + 1] = str(
        source_root / "src/fluid_control/p064_controlled_aero_ab.py"
    )
    expected_argv[expected_argv.index("--output") + 1] = spec["planned_output"]
    expected_argv[expected_argv.index("--arm") + 1] = "B"
    if "--execute" in expected_argv:
        expected_argv.remove("--execute")
    expected_argv += [
        "--reflection-module", str(reflection),
        "--reflection-module-sha256", sha256(reflection),
        "--execute",
    ]
    require(argv == expected_argv, "worker argv is not the exact reviewed B binding plus reflection")
    require(argv[argv.index("--source-root") + 1] == str(source_root), "worker source root differs")
    output = Path(spec["planned_output"])
    require(output.is_absolute() and output.resolve().is_relative_to(ROOT), "output escapes project")
    require(not output.exists() and not output.is_symlink(), "exclusive output already exists")
    for parent in output.parents:
        if parent == ROOT:
            break
        require(not parent.is_symlink(), "output ancestor symlink")
    return argv, output, source_root


def verify_unit(spec):
    keys = ["MainPID", *UNIT]
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", spec["planned_unit"], *[x for key in keys for x in ("-p", key)]],
        text=True,
        timeout=10,
    )
    actual = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(actual.get("MainPID") == str(os.getpid()), "supervisor is not actual MainPID")
    require(all(actual.get(key) == value for key, value in UNIT.items()), "outer unit limits differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "explicit GPU0 required")


def execute(spec, argv, output, source_root):
    verify_unit(spec)
    for _ in range(2):
        require(available_gib() >= 50, "startup MemAvailable below50GiB")
        time.sleep(2)
    started = time.monotonic()
    process = None
    samples = []
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": f"{source_root / 'src'}:{source_root / 'scripts'}",
    })
    try:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True)
        while process.poll() is None:
            row = {"elapsed_seconds": time.monotonic() - started, "mem_available_gib": available_gib()}
            samples.append(row)
            require(row["elapsed_seconds"] < 3600, "worker deadline exceeded")
            require(row["mem_available_gib"] >= 22, "runtime MemAvailable below22GiB")
            time.sleep(2)
        require(process.returncode == 0, f"worker exited {process.returncode}")
        result = output / "result.json"
        require(result.is_file() and not result.is_symlink(), "worker result missing")
        receipt = {
            "status": "P064_Y_REFLECTION_PAIRED_SUPERVISOR_COMPLETE",
            "unit": spec["planned_unit"], "invocation_id": os.environ.get("INVOCATION_ID"),
            "result_sha256": sha256(result), "resource_samples": samples,
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "supervisor_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.spec.is_file() and not args.spec.is_symlink(), "spec must be regular file")
    require(sha256(args.spec) == args.spec_sha256, "spec SHA differs")
    spec = json.loads(args.spec.read_text())
    argv, output, source_root = validate_spec(spec, not args.execute)
    if not args.execute:
        print(json.dumps({"dry_run": True, "status": spec["status"], "worker_argv": argv}))
        return
    execute(spec, argv, output, source_root)


if __name__ == "__main__":
    main()
