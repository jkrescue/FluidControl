"""Bounded launcher for the reviewed P064 signed teacher-forced H1 diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT = Path("/workspace/fluid_control")
PENDING = "P064_TEACHER_FORCED_H1_PREPARATION_ONLY_NOT_APPROVED"
APPROVED = "P064_TEACHER_FORCED_H1_EXECUTION_APPROVED"
RESOURCES = {
    "memory_bytes": 12 * 2**30,
    "swap_bytes": 0,
    "cpu_quota_percent": 100,
    "tasks_max": 64,
    "worker_deadline_seconds": 600,
    "outer_runtime_seconds": 630,
    "stop_seconds": 120,
    "startup_available_gib": 50,
    "runtime_available_gib": 22,
    "allocator_bytes": 6 * 2**30,
    "cuda_visible_devices": "0",
}
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
UNIT = {
    "Type": "exec",
    "MemoryMax": str(12 * 2**30),
    "MemorySwapMax": "0",
    "CPUQuotaPerSecUSec": "1s",
    "TasksMax": "64",
    "RuntimeMaxUSec": "10min 30s",
    "TimeoutStopUSec": "2min",
    "KillMode": "control-group",
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
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        rows[key] = int(value.strip().split()[0]) * 1024
    return rows["MemAvailable"] / 2**30


def validate_spec(spec, *, allow_pending):
    allowed = {APPROVED} | ({PENDING} if allow_pending else set())
    require(spec.get("status") in allowed, "approval status differs")
    require(spec.get("execution_authorized") is (spec.get("status") == APPROVED), "authorization differs")
    require(spec.get("resources") == RESOURCES, "resource contract differs")
    require(spec.get("official_image_id") == IMAGE, "official runtime image differs")
    require(spec.get("container_name") == "p064-teacher-forced-h1-20261006", "container name differs")
    require(spec.get("protocol") == {
        "mode": "teacher_forced_h1", "cases": 6, "starts_per_case": 100,
        "dual_steps": 600, "flow_forwards": 600, "aerodynamic_forwards": 600,
        "total_submodel_forwards": 1200, "batch_size": 1, "history_k": 1,
        "optimizer_steps": 0, "precision": "high_tf32",
    }, "fixed scientific protocol differs")
    require(bind(spec["supervisor"]).resolve() == Path(__file__).resolve(), "executed supervisor differs")
    worker = bind(spec["worker"])
    inputs = {name: bind(item) for name, item in spec["inputs"].items()}
    source_root = (ROOT / spec["python_source_root"]).resolve()
    require(source_root.is_dir() and source_root.is_relative_to(ROOT), "source root differs")
    receipt = json.loads(inputs["source_receipt"].read_text())
    require(receipt.get("status") == "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN", "source receipt status differs")
    source_map = receipt.get("final_files_sha256")
    require(isinstance(source_map, dict) and len(source_map) == 411, "complete 411-file source map required")
    for relative, expected in source_map.items():
        candidate = source_root / relative
        require(
            candidate.resolve().is_relative_to(source_root)
            and candidate.is_file()
            and not candidate.is_symlink()
            and sha256(candidate) == expected,
            f"source closure differs: {relative}",
        )
    output = ROOT / spec["output"]
    require(output.resolve().is_relative_to(ROOT) and not output.is_symlink(), "output escapes project")
    for parent in output.parents:
        if parent == ROOT:
            break
        require(not parent.is_symlink(), "output ancestor symlink")
    return worker, inputs, source_root, output


def verify_unit(spec):
    unit = spec["unit"]
    require(isinstance(unit, str) and unit.endswith(".service"), "unit missing")
    keys = ["MainPID", *UNIT]
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", unit, *[x for key in keys for x in ("-p", key)]],
        text=True, timeout=10,
    )
    actual = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(actual.get("MainPID") == str(os.getpid()), "supervisor is not actual MainPID")
    require(all(actual.get(key) == value for key, value in UNIT.items()), "outer unit limits differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "explicit GPU0 required")


def cleanup_owned_container(spec):
    existing = subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"name=^{spec['container_name']}$"],
        text=True, timeout=5,
    ).strip()
    if not existing:
        return
    inspect = json.loads(
        subprocess.check_output(["docker", "inspect", existing], text=True, timeout=5)
    )[0]
    require(
        inspect["Name"] == "/" + spec["container_name"]
        and inspect["Config"]["Image"] == IMAGE
        and inspect["Config"]["Labels"].get("fluid-control.p064-teacher-forced-h1") == "true",
        "refusing cleanup of unowned container",
    )
    subprocess.run(
        ["docker", "rm", "-f", existing], check=True, timeout=20, stdout=subprocess.DEVNULL
    )


def command(spec, worker, inputs, source_root, output):
    def mounted(path):
        return "/workspace/" + Path(path).resolve().relative_to(ROOT).as_posix()
    worker_command = [
        "python", "-u", mounted(worker),
        "--data", mounted(inputs["data_manifest"].parent),
        "--normalization-data", mounted(inputs["normalization"].parent),
        "--config", mounted(inputs["config"]),
        "--checkpoint-dir", mounted(inputs["aerodynamic_model"].parent),
        "--output", "/output/result.json",
        "--expected-model-sha", spec["inputs"]["aerodynamic_model"]["sha256"],
        "--dual-fno-manifest", mounted(inputs["candidate_manifest"]),
        "--expected-dual-fno-manifest-sha256", spec["inputs"]["candidate_manifest"]["sha256"],
        "--dual-training-config", mounted(inputs["training_config"]),
        "--fno-history-profile", "p026_k1",
        "--mode", "teacher_forced_h1",
        "--autoregressive-reference", mounted(inputs["autoregressive_reference"]),
        "--expected-autoregressive-reference-sha256", spec["inputs"]["autoregressive_reference"]["sha256"],
    ]
    return [
        "docker", "run", "--name", spec["container_name"], "--network", "none",
        "--cpus", "1", "--memory", "12g", "--memory-swap", "12g", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--gpus", "device=0",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--label", "fluid-control.p064-teacher-forced-h1=true",
        "-e", "XDG_CACHE_HOME=/tmp/cache", "-e", "LOCAL_CACHE=/tmp/physicsnemo-cache",
        "-e", "WARP_CACHE_PATH=/tmp/warp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", f"PYTHONPATH={mounted(source_root / 'src')}:{mounted(source_root / 'scripts')}",
        "-v", f"{ROOT.resolve()}:/workspace:ro", "-v", f"{output.resolve()}:/output:rw",
        IMAGE, *worker_command,
    ]


def execute(spec, worker, inputs, source_root, output):
    verify_unit(spec)
    require(not output.exists(), "exclusive output already exists")
    for _ in range(2):
        require(available_gib() >= 50, "startup MemAvailable below 50 GiB")
        time.sleep(2)
    output.mkdir(parents=False)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True, timeout=10
    ).strip()
    require(image_id == IMAGE, "resolved official image differs")
    existing = subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"name=^{spec['container_name']}$"], text=True, timeout=10
    ).strip()
    require(not existing, "owned container name already exists")
    started = time.monotonic()
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{source_root / 'src'}:{source_root / 'scripts'}",
        "P064_DIAGNOSTIC_TOKEN": secrets.token_hex(32),
    })
    cmd = command(spec, worker, inputs, source_root, output)
    def interrupted(signum, _frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = None
    samples = []
    try:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            row = {"elapsed_seconds": elapsed, "mem_available_gib": available_gib()}
            samples.append(row)
            require(elapsed < 600, "worker deadline exceeded")
            require(row["mem_available_gib"] >= 22, "runtime MemAvailable below 22 GiB")
            time.sleep(2)
        require(process.returncode == 0, f"worker exited {process.returncode}")
        result = output / "result.json"
        require(result.is_file() and not result.is_symlink(), "worker result missing")
        receipt = {
            "status": "P064_TEACHER_FORCED_H1_COMPLETE_NOT_ADMISSION",
            "unit": spec["unit"], "invocation_id": os.environ.get("INVOCATION_ID"),
            "result_sha256": sha256(result), "resource_samples": samples,
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_steps": 0, "execution_authorized": False,
        }
        (output / "supervisor_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        # Remove the daemon-owned container first; then reap its local CLI. Repeat
        # once after reaping to cover a create that completed at the signal edge.
        cleanup_owned_container(spec)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        cleanup_owned_container(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.spec.is_file() and not args.spec.is_symlink(), "spec regular file required")
    require(sha256(args.spec) == args.spec_sha256, "spec SHA differs")
    spec = json.loads(args.spec.read_text())
    worker, inputs, source_root, output = validate_spec(spec, allow_pending=not args.execute)
    if not args.execute:
        print(json.dumps({"dry_run": True, "status": spec["status"], "command": command(spec, worker, inputs, source_root, output)}))
        return
    execute(spec, worker, inputs, source_root, output)


if __name__ == "__main__":
    main()
