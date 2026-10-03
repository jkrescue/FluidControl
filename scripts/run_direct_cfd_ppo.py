#!/usr/bin/env python3
"""Host launcher for the isolated two-environment direct real-CFD PPO track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT / "artifacts" / "direct_cfd"
PHYSICS_SUMMARY = (
    PROJECT / "artifacts" / "matched_start_full40_extension" / "train20_physics_summary.json"
)
RUNTIME_IMAGE = "fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
RUNTIME_IMAGE_ID = "sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
HYDROGYM_COMMIT = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
PHASES = ("b00", "b02")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is unavailable")


def baselines() -> dict:
    payload = json.loads(PHYSICS_SUMMARY.read_text(encoding="utf-8"))
    if payload.get("scope", {}).get("split") != "train":
        raise ValueError("baseline summary is not train-only")
    result = {}
    for phase in PHASES:
        zero = payload["phases"][phase]["same_phase_zero"]
        metrics = zero["metrics"]
        result[phase] = {
            "total_drag": float(metrics["mean_cd_total"]),
            "rear_cl_fluctuation_rms": float(metrics["rms_cl_rear_fluctuation"]),
            "source": (
                f"{PHYSICS_SUMMARY.relative_to(PROJECT)}::{phase}::"
                f"{zero['case']}::predeclared_final_60D/U_train_only"
            ),
        }
    return result


def request(socket_path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2.0)
        connection.connect(str(socket_path))
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as reader:
            line = reader.readline(1024 * 1024)
    response = json.loads(line)
    if response.get("ok") is not True:
        raise RuntimeError(response)
    return response


def wait_for_workers(paths: list[Path], processes: list[subprocess.Popen]) -> list[dict]:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("direct-CFD worker exited during startup")
        try:
            return [request(path, {"command": "hello"}) for path in paths]
        except (FileNotFoundError, ConnectionError, OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("direct-CFD workers did not become ready within 45 seconds")


def current_pimplefoam() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-a", "pimpleFoam"], text=True, capture_output=True
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("cannot audit current pimpleFoam processes")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def preflight(run_id: str, output: Path, probe_transitions: int) -> dict:
    if not re.fullmatch(r"[a-z0-9_]{4,48}", run_id):
        raise ValueError("run-id must be 4..48 lowercase alphanumeric/underscore characters")
    output = output.resolve()
    if not output.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("output must stay under artifacts/direct_cfd")
    if output.exists():
        raise FileExistsError(output)
    if probe_transitions not in (0, 20):
        raise ValueError("probe-transitions must be zero (2048 PPO) or exactly 20")
    if available_memory_gib() < 20.0:
        raise RuntimeError("MemAvailable below 20 GiB")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", RUNTIME_IMAGE],
        text=True,
    ).strip()
    if image_id != RUNTIME_IMAGE_ID:
        raise ValueError(f"HydroGym runtime image mismatch: {image_id}")
    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT / ".tools" / "hydrogym"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != HYDROGYM_COMMIT:
        raise ValueError(f"HydroGym checkout mismatch: {commit}")
    active = current_pimplefoam()
    return {
        "status": "DIRECT_REAL_CFD_PPO_PREFLIGHT_PASS" if not active else "BLOCKED_ACTIVE_OPENFOAM",
        "run_id": run_id,
        "output": str(output),
        "mode": "20-transition runtime probe" if probe_transitions else "2048-transition PPO",
        "phases": list(PHASES),
        "episode_steps": 128,
        "environment_count": 2,
        "timesteps": probe_transitions or 2048,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_id": image_id,
        "hydrogym_commit": commit,
        "physics_summary": str(PHYSICS_SUMMARY),
        "physics_summary_sha256": sha256(PHYSICS_SUMMARY),
        "baselines": baselines(),
        "mem_available_gib": available_memory_gib(),
        "active_pimplefoam": active,
        "frozen_or_validation_access": False,
        "surrogate_used": False,
        "scientific_scope": (
            "train-only direct real-CFD RL/basic online closure; short-window training "
            "diagnostics are not final 80-D/U paired-CFD acceptance"
        ),
    }


def stop_owned(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def execute(args, audit: dict) -> int:
    if audit["status"] != "DIRECT_REAL_CFD_PPO_PREFLIGHT_PASS":
        raise RuntimeError("refusing launch while another pimpleFoam process is active")
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    baseline_path = output / "train_only_baselines.json"
    baseline_path.write_text(json.dumps(audit["baselines"], indent=2) + "\n", encoding="utf-8")
    socket_root = Path(tempfile.mkdtemp(prefix=f"dcfd_{args.run_id[:8]}_", dir="/tmp"))
    sockets = [socket_root / f"e{index}.sock" for index in range(2)]
    if any(len(os.fsencode(path)) > 103 for path in sockets):
        raise ValueError("AF_UNIX socket path exceeds the conservative 103-byte limit")
    (output / "socket_runtime.json").write_text(
        json.dumps(
            {
                "host_socket_root": str(socket_root),
                "container_socket_root": "/run/direct_cfd_sockets",
                "af_unix_path_limit_checked": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workers: list[subprocess.Popen] = []
    worker_logs = []
    try:
        for index, (phase, socket_path) in enumerate(zip(PHASES, sockets, strict=True)):
            log = (output / f"worker_env{index}.log").open("w", encoding="utf-8")
            worker_logs.append(log)
            command = [
                sys.executable,
                "-u",
                str(PROJECT / "scripts" / "direct_cfd_openfoam_worker.py"),
                "--socket",
                str(socket_path),
                "--journal",
                str(output / f"worker_env{index}.jsonl"),
                "--run-id",
                args.run_id,
                "--env-index",
                str(index),
                "--phase",
                phase,
            ]
            workers.append(
                subprocess.Popen(
                    command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
                )
            )
        hello = wait_for_workers(sockets, workers)
        (output / "workers_ready.json").write_text(
            json.dumps(hello, indent=2) + "\n", encoding="utf-8"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"direct-cfd-policy-{args.run_id}",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=2g",
            "--cpus",
            "2",
            "--memory",
            "16g",
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "PYTHONPATH=/workspace/src:/workspace/.tools/hydrogym",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"USER={os.environ.get('USER', 'USER')}",
            "--env",
            f"LOGNAME={os.environ.get('LOGNAME', 'USER')}",
            "--env",
            "HOME=/tmp",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "TORCH_NUM_THREADS=1",
            "--mount",
            f"type=bind,src={PROJECT / 'src'},dst=/workspace/src,readonly",
            "--mount",
            f"type=bind,src={PROJECT / 'scripts'},dst=/workspace/scripts,readonly",
            "--mount",
            (
                f"type=bind,src={PROJECT / '.tools' / 'hydrogym'},"
                "dst=/workspace/.tools/hydrogym,readonly"
            ),
            "--mount",
            f"type=bind,src={output},dst=/run/direct_cfd",
            "--mount",
            f"type=bind,src={socket_root},dst=/run/direct_cfd_sockets",
            "--workdir",
            "/workspace",
            RUNTIME_IMAGE,
            "python",
            "-u",
            "scripts/train_direct_cfd_ppo.py",
            "--socket",
            "/run/direct_cfd_sockets/e0.sock",
            "--socket",
            "/run/direct_cfd_sockets/e1.sock",
            "--baselines",
            "/run/direct_cfd/train_only_baselines.json",
            "--output",
            "/run/direct_cfd/training",
            "--seed",
            str(args.seed),
        ]
        if args.probe_transitions:
            command.extend(["--probe-transitions", str(args.probe_transitions)])
        with (output / "policy_runtime.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "DIRECT_REAL_CFD_RUN_FAILED",
                        "returncode": completed.returncode,
                        "scientific_scope": "failed run; no PPO or physical claim",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return completed.returncode
    finally:
        subprocess.run(
            ["docker", "rm", "--force", f"direct-cfd-policy-{args.run_id}"],
            cwd=PROJECT,
            text=True,
            capture_output=True,
        )
        stop_owned(workers)
        for log in worker_logs:
            log.close()
        try:
            socket_root.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--probe-transitions", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    audit = preflight(args.run_id, args.output, args.probe_transitions)
    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return
    returncode = execute(args, audit)
    if returncode:
        raise subprocess.CalledProcessError(returncode, "policy runtime container")
    print("DIRECT_REAL_CFD_LAUNCHER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
