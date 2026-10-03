#!/usr/bin/env python3
"""Replay the immutable b00 PPO action endpoints open-loop from the b01 restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
ARTIFACT_ROOT = PROJECT / "artifacts/direct_cfd"
CASES = PROJECT / "cfd/tandem_cylinders/cases"
SOURCE_ROLLOUT = ARTIFACT_ROOT / "directppo2048_b00_eval80_v1/rollout/rollout_result.json"
B01_RESULT = ARTIFACT_ROOT / "directppo2048_b01_eval80_v1/physical_result.json"
PREDECLARATION = PROJECT / "docs/results/direct_cfd_b00_actions_b01_openloop_predeclaration.json"
SOURCE_ROLLOUT_SHA256 = "70741e495c5ecd9ad53c7820c886a33a1e3483c00bdc2f7cdc74dd84dc340cf6"
ACTION_SEQUENCE_SHA256 = "e81be75c361f2b82afcdccc2151ca7746b76f7133bfea3ae3348c7ab8a7ecb5b"
B01_RESULT_SHA256 = "da42b6018f35677af8205ce9ba78cc9753ca348abf5f5994f2189554c4eaba36"
START = 130.0
END = 210.0
WINDOW = (150.0, 210.0)
STEPS = 800

sys.path.insert(0, str(PROJECT / "scripts"))
from run_tandem_phase_feedback_pair import (  # noqa: E402
    compare_metrics,
    force_metrics,
    read_force_window,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_sequence_sha256(actions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f8").tobytes()).hexdigest()


def load_action_sequence(path: Path = SOURCE_ROLLOUT) -> np.ndarray:
    if sha256(path) != SOURCE_ROLLOUT_SHA256:
        raise ValueError("immutable b00 rollout SHA differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("rows", []) if row.get("role") == "ppo"]
    if len(rows) != STEPS or [row.get("step") for row in rows] != list(range(1, STEPS + 1)):
        raise ValueError("b00 action source must contain exact PPO steps 1..800")
    expected_time = START + 18.0 + 0.1 * np.arange(1, STEPS + 1)
    if not np.allclose([row["cfd_time"] for row in rows], expected_time, rtol=0, atol=2e-6):
        raise ValueError("b00 source clock differs")
    actions = np.asarray([row["applied_omega"] for row in rows], dtype=np.float64)
    delta = np.diff(np.r_[0.0, actions])
    if (
        not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 0.75 + 1e-8
        or np.max(np.abs(delta)) > 0.1 + 1e-8
        or action_sequence_sha256(actions) != ACTION_SEQUENCE_SHA256
    ):
        raise ValueError("b00 action endpoint contract differs")
    return actions


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def request(path: Path, payload: dict, timeout: float = 180.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline(8 * 1024 * 1024))
    if response.get("ok") is not True:
        raise RuntimeError(response.get("error", response))
    return response


def wait_ready(path: Path, worker: subprocess.Popen) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if worker.poll() is not None:
            raise RuntimeError("OpenFOAM worker exited during startup")
        try:
            request(path, {"command": "hello"}, timeout=2.0)
            return
        except (OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("OpenFOAM worker did not become ready")


def validate_predeclaration(run_id: str, actions: np.ndarray) -> dict:
    document = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    expected = {
        "status": "B00_ACTIONS_B01_OPENLOOP_PREDECLARED_NOT_EXECUTED",
        "run_id": run_id,
        "source_rollout_sha256": SOURCE_ROLLOUT_SHA256,
        "action_sequence_sha256_float64_le": action_sequence_sha256(actions),
        "action_endpoints": STEPS,
        "source_phase": "b00_train",
        "replay_phase": "b01_validation",
        "source_restart_time": START,
        "end_time": END,
        "analysis_window": list(WINDOW),
        "control_dt": 0.1,
        "solver_dt": 0.005,
        "frozen_test_access": False,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("open-loop predeclaration differs")
    return document


def summarize(output: Path, case: Path, rows: list[dict], journal: Path) -> dict:
    front = read_force_window(case, "forceFront", *WINDOW)
    rear = read_force_window(case, "forceRear", *WINDOW)
    for label, values in (("front", front), ("rear", rear)):
        if len(values) != 12001 or not np.isfinite(values).all():
            raise ValueError(f"{label} final force window is incomplete")
        if not np.allclose(np.diff(values[:, 0]), 0.005, rtol=0, atol=1e-8):
            raise ValueError(f"{label} final force grid differs")
    metrics = force_metrics(front, rear)
    existing = json.loads(B01_RESULT.read_text(encoding="utf-8"))
    if sha256(B01_RESULT) != B01_RESULT_SHA256 or set(existing["metrics"]) != {"ppo", "zero"}:
        raise ValueError("audited b01 feedback/zero result differs")
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    steps = [event for event in events if event.get("event") == "step"]
    if len(rows) != STEPS or len(steps) != STEPS:
        raise ValueError("open-loop replay does not contain 800 completed steps")
    health = [event["solver_health"] for event in steps]
    result = {
        "status": "B00_ACTIONS_B01_OPENLOOP_REPLAY_COMPLETE",
        "protocol": {
            "source": "immutable b00 final2048 feedback applied endpoints",
            "source_rollout_sha256": SOURCE_ROLLOUT_SHA256,
            "action_sequence_sha256_float64_le": ACTION_SEQUENCE_SHA256,
            "replay_start": START,
            "replay_end": END,
            "analysis_window": list(WINDOW),
            "control_dt": 0.1,
            "solver_dt": 0.005,
            "linear_endpoint_ramp": True,
        },
        "case": case.name,
        "metrics": {
            "openloop_b00_sequence_at_b01": metrics,
            "existing_b01_feedback": existing["metrics"]["ppo"],
            "existing_b01_zero": existing["metrics"]["zero"],
        },
        "comparison": {
            "openloop_vs_zero": compare_metrics(metrics, existing["metrics"]["zero"]),
            "feedback_vs_zero": existing["comparison"],
            "feedback_vs_openloop_reference": compare_metrics(existing["metrics"]["ppo"], metrics),
        },
        "action": {
            "max_abs_omega": max(abs(row["applied_omega"]) for row in rows),
            "max_abs_delta_omega": max(abs(row["applied_delta_omega"]) for row in rows),
            "endpoints_reproduced_exactly": all(row["endpoint_matches_source"] for row in rows),
        },
        "solver_audit": {
            "segments": len(health),
            "all_ended_cleanly": all(item["solver_ended_cleanly"] for item in health),
            "all_20_steps": all(item["steps"] == 20 for item in health),
            "max_courant": max(item["max_courant"] for item in health),
            "max_abs_global_continuity_per_step": max(item["max_abs_global_continuity_per_step"] for item in health),
            "min_mem_available_gib": min(event["mem_available_gib"] for event in steps),
        },
        "interpretation": (
            "Post-hoc single-start discriminator. Similar open-loop and feedback performance means "
            "the implemented feedback loop remains real, but feedback superiority is not established."
        ),
        "scientific_scope": (
            "held-out-time b01 restart only; not an independent physical sample, frozen-test result, "
            "policy-selection input, broad phase-generalization proof, or net-energy claim"
        ),
        "rows": rows,
    }
    write_atomic(output / "result.json", result)
    return result


def execute(args) -> int:
    output = args.output.resolve()
    if not output.is_relative_to(ARTIFACT_ROOT.resolve()) or output.exists():
        raise ValueError("new output must stay under artifacts/direct_cfd")
    actions = load_action_sequence()
    predeclaration = validate_predeclaration(args.run_id, actions)
    if sha256(B01_RESULT) != B01_RESULT_SHA256:
        raise ValueError("b01 result SHA differs")
    if available_memory_gib() < 40.0:
        raise RuntimeError("preflight MemAvailable below 40 GiB")
    disk_free_gib = os.statvfs(PROJECT).f_bavail * os.statvfs(PROJECT).f_frsize / 1024**3
    if disk_free_gib < 20.0:
        raise RuntimeError("preflight free disk below 20 GiB")
    active = subprocess.run(["pgrep", "-a", "pimpleFoam"], text=True, capture_output=True)
    if active.returncode == 0:
        raise RuntimeError(f"active OpenFOAM detected: {active.stdout}")
    if active.returncode != 1:
        raise RuntimeError("could not audit active OpenFOAM processes")
    output.mkdir(parents=True)
    write_atomic(
        output / "preflight_receipt.json",
        {
            "status": "B00_ACTIONS_B01_OPENLOOP_PREFLIGHT_PASS",
            "predeclaration_sha256": sha256(PREDECLARATION),
            "source_rollout_sha256": sha256(SOURCE_ROLLOUT),
            "action_sequence_sha256_float64_le": action_sequence_sha256(actions),
            "b01_result_sha256": sha256(B01_RESULT),
            "mem_available_gib": available_memory_gib(),
            "disk_free_gib": disk_free_gib,
            "estimated_additional_disk_gib_upper_bound": predeclaration["estimated_additional_disk_gib_upper_bound"],
            "execution_performed": False,
        },
    )
    socket_root = Path(tempfile.mkdtemp(prefix="dcfd_openloop_", dir="/tmp"))
    socket_path = socket_root / "e0.sock"
    journal = output / "worker.jsonl"
    log_handle = (output / "worker.log").open("w", encoding="utf-8")
    worker = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(PROJECT / "scripts/direct_cfd_openfoam_worker.py"),
            "--socket",
            str(socket_path),
            "--journal",
            str(journal),
            "--run-id",
            args.run_id,
            "--env-index",
            "0",
            "--phase",
            "b01",
            "--episode-steps",
            "800",
        ],
        cwd=PROJECT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_ready(socket_path, worker)
        reset = request(socket_path, {"command": "reset"})
        case = CASES / reset["worker_info"]["case"]
        rows = []
        previous = 0.0
        for index, endpoint in enumerate(actions):
            response = request(
                socket_path,
                {"command": "step", "requested_omega": float(endpoint), "iteration": index},
            )
            applied = float(response["applied_omega"])
            expected_time = START + 0.1 * (index + 1)
            matches = math.isclose(applied, float(endpoint), abs_tol=1e-10)
            if not matches or not math.isclose(float(response["time"]), expected_time, abs_tol=2e-6):
                raise ValueError("worker did not reproduce the predeclared endpoint/clock")
            rows.append(
                {
                    "step": index + 1,
                    "cfd_time": float(response["time"]),
                    "source_applied_omega": float(endpoint),
                    "applied_omega": applied,
                    "applied_delta_omega": applied - previous,
                    "endpoint_matches_source": matches,
                }
            )
            previous = applied
            if (index + 1) % 100 == 0:
                write_atomic(
                    output / "progress.json",
                    {
                        "status": "B00_ACTIONS_B01_OPENLOOP_RUNNING",
                        "completed_steps": index + 1,
                        "case": case.name,
                        "min_mem_guard_gib": 20.0,
                    },
                )
        summarize(output, case, rows, journal)
        return 0
    finally:
        if worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        log_handle.close()
        try:
            socket_root.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="b00seq_b01_openloop_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "b00seq_b01_openloop_v1",
    )
    args = parser.parse_args()
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
