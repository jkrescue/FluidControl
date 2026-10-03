#!/usr/bin/env python3
"""Host launcher for an 800-step b00 frozen-PPO versus zero CFD pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT / "artifacts" / "direct_cfd"
RUNTIME_IMAGE = "fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
RUNTIME_IMAGE_ID = "sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
HYDROGYM_COMMIT = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
sys.path.insert(0, str(PROJECT / "scripts"))
from run_tandem_phase_feedback_pair import (  # noqa: E402
    compare_metrics,
    force_metrics,
    read_force_window,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2.0)
        connection.connect(str(path))
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline(1024 * 1024))
    if response.get("ok") is not True:
        raise RuntimeError(response)
    return response


def stop_owned(processes: list[subprocess.Popen], policy_container: str) -> None:
    subprocess.run(
        ["docker", "rm", "--force", policy_container], capture_output=True, text=True
    )
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_ready(paths: list[Path], processes: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("evaluation worker exited during startup")
        try:
            for path in paths:
                request(path, {"command": "hello"})
            return
        except (OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("evaluation workers did not become ready")


def validate_rollout_contract(rollout: dict) -> None:
    if set(rollout.get("branches", {})) != {"ppo", "zero"}:
        raise ValueError("paired rollout must contain exactly ppo and zero roles")
    rows = rollout.get("rows", [])
    for role in ("ppo", "zero"):
        branch = [row for row in rows if row.get("role") == role]
        if len(branch) != 800 or [row["step"] for row in branch] != list(range(1, 801)):
            raise ValueError(f"{role} branch does not contain exactly steps 1..800")
        expected_time = np.asarray([148.0 + 0.1 * step for step in range(1, 801)])
        if not np.allclose(
            [row["cfd_time"] for row in branch], expected_time, rtol=0.0, atol=2e-6
        ):
            raise ValueError(f"{role} branch violated the physical control clock")
        omega = np.asarray([row["applied_omega"] for row in branch])
        delta = np.asarray([row["applied_delta_omega"] for row in branch])
        if np.max(np.abs(omega)) > 0.75 + 1e-8 or np.max(np.abs(delta)) > 0.1 + 1e-8:
            raise ValueError(f"{role} branch violated the canonical action contract")
        if role == "zero" and (np.any(omega != 0.0) or np.any(delta != 0.0)):
            raise ValueError("zero branch applied a nonzero action")


def physical_summary(output: Path, rollout: dict) -> dict:
    validate_rollout_contract(rollout)
    cases = {
        role: PROJECT / "cfd" / "tandem_cylinders" / "cases" / name
        for role, name in rollout["branches"].items()
    }
    metrics = {}
    for role, case in cases.items():
        front = read_force_window(case, "forceFront", 168.0, 228.0)
        rear = read_force_window(case, "forceRear", 168.0, 228.0)
        for label, values in (("front", front), ("rear", rear)):
            if len(values) != 12001 or not np.isfinite(values).all():
                raise ValueError(f"{role} {label} force window is incomplete/non-finite")
            if not np.allclose(np.diff(values[:, 0]), 0.005, rtol=0.0, atol=1e-8):
                raise ValueError(f"{role} {label} force window is not continuous dt=0.005")
        metrics[role] = force_metrics(front, rear)
    comparison = compare_metrics(metrics["ppo"], metrics["zero"])
    ppo_rows = [row for row in rollout["rows"] if row["role"] == "ppo"]
    omega = np.asarray([row["applied_omega"] for row in ppo_rows], dtype=float)
    delta = np.asarray([row["applied_delta_omega"] for row in ppo_rows], dtype=float)
    result = {
        "status": "DIRECT_CFD_B00_FROZEN_PPO_PAIR_EVALUATED",
        "phase": "b00_train",
        "window": [168.0, 228.0],
        "metrics": metrics,
        "comparison": comparison,
        "action": {
            "max_abs_omega": float(np.max(np.abs(omega))),
            "max_abs_delta_omega": float(np.max(np.abs(delta))),
            "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        },
        "policy_sha256": rollout["policy_sha256"],
        "vecnormalize_sha256": rollout["vecnormalize_sha256"],
        "policy_deterministic": rollout["policy_deterministic"],
        "vecnormalize_training": rollout["vecnormalize_training"],
        "vecnormalize_norm_reward": rollout["vecnormalize_norm_reward"],
        "scientific_scope": (
            "training-phase preliminary paired physical validation only; not independent "
            "generalization, frozen-test evidence, net-energy evidence, or final paper claim"
        ),
    }
    (output / "physical_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def execute(args) -> int:
    output = args.output.resolve()
    if not output.is_relative_to(ARTIFACT_ROOT.resolve()) or output.exists():
        raise ValueError("new output must stay under artifacts/direct_cfd")
    active = subprocess.run(
        ["pgrep", "-a", "pimpleFoam"], text=True, capture_output=True
    )
    if active.returncode == 0:
        raise RuntimeError(f"refusing evaluation with active OpenFOAM: {active.stdout}")
    if active.returncode != 1:
        raise RuntimeError("cannot audit active OpenFOAM processes")
    training = args.training.resolve()
    result = json.loads((training / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "DIRECT_REAL_CFD_PPO_TRAINING_COMPLETE":
        raise ValueError("completed direct-CFD PPO result is required")
    policy = training / Path(result["policy"]).name
    vecnormalize = training / Path(result["vecnormalize"]).name
    if not policy.is_file() or not vecnormalize.is_file():
        raise FileNotFoundError("frozen policy or VecNormalize artifact is absent")
    if sha256(policy) != result.get("policy_sha256"):
        raise ValueError("frozen policy SHA differs from training result")
    if sha256(vecnormalize) != result.get("vecnormalize_sha256"):
        raise ValueError("VecNormalize SHA differs from training result")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", RUNTIME_IMAGE],
        text=True,
    ).strip()
    if image_id != RUNTIME_IMAGE_ID:
        raise ValueError("HydroGym runtime image ID mismatch")
    hydrogym_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT / ".tools" / "hydrogym"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if hydrogym_commit != HYDROGYM_COMMIT:
        raise ValueError("HydroGym commit mismatch")
    output.mkdir(parents=True)
    (output / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "status": "DIRECT_CFD_B00_EVALUATION_PREFLIGHT_PASS",
                "runtime_image": RUNTIME_IMAGE,
                "runtime_image_id": image_id,
                "hydrogym_commit": hydrogym_commit,
                "training_result": str(training / "result.json"),
                "training_result_sha256": sha256(training / "result.json"),
                "policy_sha256": sha256(policy),
                "vecnormalize_sha256": sha256(vecnormalize),
                "phase": "b00_train",
                "steps_per_branch": 800,
                "analysis_window": [168.0, 228.0],
                "validation_or_frozen_access": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    socket_root = Path(tempfile.mkdtemp(prefix="dcfd_eval_", dir="/tmp"))
    sockets = [socket_root / f"e{index}.sock" for index in range(2)]
    workers = []
    logs = []
    policy_container = f"direct-cfd-eval-{args.run_id}"
    try:
        for index, path in enumerate(sockets):
            log = (output / f"worker_env{index}.log").open("w", encoding="utf-8")
            logs.append(log)
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(PROJECT / "scripts" / "direct_cfd_openfoam_worker.py"),
                        "--socket",
                        str(path),
                        "--journal",
                        str(output / f"worker_env{index}.jsonl"),
                        "--run-id",
                        args.run_id,
                        "--env-index",
                        str(index),
                        "--phase",
                        "b00",
                        "--episode-steps",
                        "800",
                    ],
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        wait_ready(sockets, workers)
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            policy_container,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=2g",
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
            "HOME=/tmp",
            "--env",
            f"USER={os.environ.get('USER', 'USER')}",
            "--env",
            f"LOGNAME={os.environ.get('LOGNAME', 'USER')}",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
            "--mount",
            f"type=bind,src={PROJECT / 'src'},dst=/workspace/src,readonly",
            "--mount",
            f"type=bind,src={PROJECT / 'scripts'},dst=/workspace/scripts,readonly",
            "--mount",
            f"type=bind,src={PROJECT / '.tools' / 'hydrogym'},dst=/workspace/.tools/hydrogym,readonly",
            "--mount",
            f"type=bind,src={output},dst=/run/output",
            "--mount",
            f"type=bind,src={socket_root},dst=/run/sockets",
            "--mount",
            f"type=bind,src={training},dst=/input,readonly",
            "--mount",
            f"type=bind,src={training.parent / 'train_only_baselines.json'},dst=/run/baselines.json,readonly",
            "--workdir",
            "/workspace",
            RUNTIME_IMAGE,
            "python",
            "-u",
            "scripts/evaluate_direct_cfd_ppo_pair.py",
            "--socket",
            "/run/sockets/e0.sock",
            "--socket",
            "/run/sockets/e1.sock",
            "--baseline",
            "/run/baselines.json",
            "--policy",
            f"/input/{policy.name}",
            "--vecnormalize",
            f"/input/{vecnormalize.name}",
            "--output",
            "/run/output/rollout",
        ]
        with (output / "runtime.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            return completed.returncode
        rollout = json.loads((output / "rollout" / "rollout_result.json").read_text())
        physical_summary(output, rollout)
        return 0
    finally:
        stop_owned(workers, policy_container)
        for log in logs:
            log.close()
        try:
            socket_root.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    returncode = execute(args)
    if returncode:
        raise subprocess.CalledProcessError(returncode, "direct-CFD evaluation")


if __name__ == "__main__":
    main()
