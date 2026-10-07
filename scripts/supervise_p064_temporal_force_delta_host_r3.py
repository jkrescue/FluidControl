"""Bounded host-py312 launcher for the reviewed temporal-force-delta probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

ROOT = Path("/workspace/fluid_control")
PYTHON = ROOT / ".venv-curator-py312/bin/python"
PENDING = "P064_TEMPORAL_FORCE_DELTA_PROBE_HOST_R3_PREPARATION_ONLY_NOT_APPROVED"
APPROVED = "P064_TEMPORAL_FORCE_DELTA_PROBE_HOST_R3_EXECUTION_APPROVED"
RESOURCES = {
    "memory_bytes": 24 * 2**30, "swap_bytes": 0, "cpu_quota_percent": 100,
    "tasks_max": 64, "worker_deadline_seconds": 600, "outer_runtime_seconds": 630,
    "stop_seconds": 20, "startup_available_gib": 50, "runtime_available_gib": 22,
    "allocator_bytes": 16 * 2**30, "cuda_visible_devices": "0",
}
UNIT = {
    "Type": "exec", "MemoryMax": str(24 * 2**30), "MemorySwapMax": "0",
    "CPUQuotaPerSecUSec": "1s", "TasksMax": "64", "RuntimeMaxUSec": "10min 30s",
    "TimeoutStopUSec": "20s", "KillMode": "control-group",
}
PROTOCOL = {
    "mode": "absolute_vs_temporal_force_delta_one_step_probe",
    "dataset": "controlled_b00_train_only", "fixed_start": 0,
    "rollout_steps": 100, "chunk_size": 10, "batch_size": 1,
    "optimizer_steps_per_arm": 1, "arms": ["absolute", "temporal_residual"],
    "models_saved": 0, "dev_or_frozen_accessed": False, "precision": "high_tf32",
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(item):
    path = ROOT / item["path"]
    require(path.resolve().is_relative_to(ROOT), "bound path escapes project")
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
    require(spec.get("execution_authorized") is (spec.get("status") == APPROVED), "authorization differs")
    require(spec.get("resources") == RESOURCES, "resource contract differs")
    require(spec.get("protocol") == PROTOCOL, "scientific protocol differs")
    require(spec.get("runtime") == {
        "kind": "host_venv_curator_py312", "python": str(PYTHON),
        "python_resolved": "/home/USER/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/bin/python3.12",
        "python_sha256": "001718d5edf61e6fbc3642c9668def38d2bf28cc320e75196e6a95d49614cca8",
        "rationale": "R1 b40 migration-layout OOM and host R2 first aerodynamic initial-loss forward hit the proven 6GiB allocator cap; R3 changes only the approved engineering cap to16GiB and cgroup to24GiB",
    }, "host runtime record differs")
    require(PYTHON.is_file() and PYTHON.is_symlink(), "host venv interpreter link differs")
    require(str(PYTHON.resolve()) == spec["runtime"]["python_resolved"], "host interpreter target differs")
    require(sha256(PYTHON.resolve()) == spec["runtime"]["python_sha256"], "host interpreter SHA differs")
    require(bind(spec["supervisor"]).resolve() == Path(__file__).resolve(), "executed supervisor differs")
    worker = bind(spec["worker"])
    required = {
        "config", "data_manifest", "normalization", "hdf", "source_manifest",
        "parent_manifest", "parent_result", "parent_protocol", "parent_flow_checkpoint",
        "parent_flow_model", "parent_aerodynamic_checkpoint", "parent_aerodynamic_model",
        "residual_module", "p013_objective", "b40_r1_approval", "b40_r1_preflight_receipt",
        "host_migration_receipt", "b40_migration_receipt", "host_r2_approval",
    }
    require(set(spec["inputs"]) == required, "exact input set differs")
    inputs = {name: bind(item) for name, item in spec["inputs"].items()}
    source_root = (ROOT / spec["python_source_root"]).resolve()
    require(source_root.is_dir() and source_root.is_relative_to(ROOT), "source root differs")
    source_map = json.loads(inputs["source_manifest"].read_text()).get("files")
    require(isinstance(source_map, dict) and len(source_map) == 445, "445 source closure required")
    for relative, expected in source_map.items():
        candidate = source_root / relative
        require(candidate.is_file() and not candidate.is_symlink() and sha256(candidate) == expected, f"source differs: {relative}")
    require(inputs["p013_objective"].resolve() == (source_root / "scripts/train_fcp013_independent_force_fno.py").resolve(), "P013 origin differs")
    output = ROOT / spec["output"]
    require(output.resolve().is_relative_to(ROOT) and not output.is_symlink(), "output escapes project")
    for parent in output.parents:
        if parent == ROOT:
            break
        require(not parent.is_symlink(), "output ancestor symlink")
    return worker, inputs, source_root, output


def verify_unit(spec):
    keys = ["MainPID", *UNIT]
    raw = subprocess.check_output(["systemctl", "--user", "show", spec["unit"], *[x for k in keys for x in ("-p", k)]], text=True, timeout=10)
    actual = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(actual.get("MainPID") == str(os.getpid()), "supervisor is not actual MainPID")
    require(all(actual.get(k) == v for k, v in UNIT.items()), "outer unit limits differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "explicit GPU0 required")


def command(worker, inputs, output):
    return [
        str(PYTHON), "-u", str(worker), "--config", str(inputs["config"]),
        "--parent", str(inputs["parent_manifest"].parent), "--data-root", str(inputs["data_manifest"].parent),
        "--hdf", str(inputs["hdf"]), "--hdf-sha256", sha256(inputs["hdf"]),
        "--normalization-sha256", sha256(inputs["normalization"]),
        "--data-manifest-sha256", sha256(inputs["data_manifest"]),
        "--residual-module", str(inputs["residual_module"]), "--p013-objective", str(inputs["p013_objective"]),
        "--output", str(output / "payload"), "--execute",
    ]


def execute(spec, worker, inputs, source_root, output):
    verify_unit(spec)
    require(not output.exists(), "exclusive output exists")
    for _ in range(2):
        require(available_gib() >= 50, "startup MemAvailable below50GiB")
        time.sleep(2)
    output.mkdir(parents=False)
    token = secrets.token_hex(32)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": f"{source_root / 'scripts'}:{source_root / 'src'}",
        "P064_TEMPORAL_DELTA_CHILD_TOKEN": token,
    })
    started = time.monotonic()
    samples = []
    process = None
    def interrupted(signum, _frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        process = subprocess.Popen(command(worker, inputs, output), cwd=ROOT, env=env, start_new_session=True)
        while process.poll() is None:
            row = {"elapsed_seconds": time.monotonic() - started, "mem_available_gib": available_gib()}
            samples.append(row)
            require(row["elapsed_seconds"] < 600, "worker deadline exceeded")
            require(row["mem_available_gib"] >= 22, "runtime MemAvailable below22GiB")
            time.sleep(2)
        require(process.returncode == 0, f"worker exited {process.returncode}")
        result = output / "payload/result.json"
        require(result.is_file() and not result.is_symlink(), "worker result missing")
        receipt = {
            "status": "P064_TEMPORAL_FORCE_DELTA_HOST_R3_COMPLETE_NOT_A_CANDIDATE",
            "unit": spec["unit"], "invocation_id": os.environ.get("INVOCATION_ID"),
            "result_sha256": sha256(result), "resource_samples": samples,
            "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 2,
            "models_saved": 0, "runtime": spec["runtime"],
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
    require(args.spec.is_file() and not args.spec.is_symlink() and sha256(args.spec) == args.spec_sha256, "spec differs")
    spec = json.loads(args.spec.read_text())
    worker, inputs, source_root, output = validate_spec(spec, not args.execute)
    if not args.execute:
        print(json.dumps({"dry_run": True, "status": spec["status"], "command": command(worker, inputs, output)}))
        return
    execute(spec, worker, inputs, source_root, output)


if __name__ == "__main__":
    main()
