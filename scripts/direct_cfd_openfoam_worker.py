#!/usr/bin/env python3
"""Host-side, single-environment OpenFOAM worker for direct CFD PPO.

The worker owns one persistent pinned OpenFOAM container and preserves every
episode case. It exposes a small AF_UNIX JSON-lines protocol to the isolated
HydroGym/SB3 policy container; no Docker socket is mounted into that container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CFD = PROJECT / "cfd" / "tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import load_coefficients, log_health  # noqa: E402
from make_expanded_control_dataset import replace_rear_patch  # noqa: E402
from make_probe_feedback_case import substitute  # noqa: E402

sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import total_drag_observation_at  # noqa: E402

OPENFOAM_IMAGE = (
    "opencfd/openfoam-default@"
    "sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
CONTROL_DT = 0.1
ACTION_LIMIT = 0.75
MAX_DELTA_OMEGA = 0.1
PREHISTORY_SECONDS = 6.15
MEMORY_FLOOR_GIB = 20.0
DISK_FREE_FLOOR_GIB = 200.0
MAX_COURANT = 0.8
MAX_CONTINUITY = 1.0e-5
HARD_FORCE_LIMIT = 10.0
ALLOWED_SOURCES = {
    "b00": ("matched_start_acquisition_train_b00_zero", "train"),
    "b02": ("matched_start_acquisition_train_b02_zero", "train"),
    "b01": ("matched_start_acquisition_validation_b01_zero", "validation"),
}


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


def latest_time(case: Path) -> float:
    values = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                values.append(float(path.name))
            except ValueError:
                pass
    if not values:
        raise ValueError(f"no OpenFOAM time directories: {case}")
    return max(values)


def force_rows(case: Path, object_name: str) -> dict[float, tuple[float, float]]:
    paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {object_name} force history in {case}")
    rows: dict[float, tuple[float, float]] = {}
    for path in paths:
        for timestamp, cd, cl in load_coefficients(path):
            key = round(timestamp, 8)
            value = (cd, cl)
            if key in rows and not np.allclose(rows[key], value, rtol=0.0, atol=1e-8):
                raise ValueError(f"conflicting force restart sample at t={timestamp}")
            rows[key] = value
    return rows


def actual_causal_prehistory(reference: Path, restart_time: float) -> tuple[list, list, dict]:
    front = force_rows(reference, "forceFront")
    rear = force_rows(reference, "forceRear")
    # 62 uniform 0.1-D/U point samples cover 6.2 D/U under the canonical
    # point-as-interval convention. Every sample is real and no later than t0.
    times = [round(restart_time - 6.1 + CONTROL_DT * index, 8) for index in range(62)]
    forces = []
    for timestamp in times:
        if timestamp not in front or timestamp not in rear:
            raise FileNotFoundError(f"missing causal force sample at t={timestamp}")
        forces.append([*front[timestamp], *rear[timestamp]])
    if times[-1] > restart_time + 1e-8:
        raise AssertionError("future force sample entered prehistory")
    sources = {}
    for object_name in ("forceFront", "forceRear"):
        paths = sorted(reference.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
        sources[object_name] = [
            {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
            for path in paths
        ]
    return times, forces, sources


class OpenFOAMWorker:
    def __init__(
        self,
        *,
        run_id: str,
        env_index: int,
        phase: str,
        journal: Path,
        episode_steps: int = 128,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9_]{4,48}", run_id):
            raise ValueError("run_id must be 4..48 lowercase alphanumeric/underscore characters")
        if env_index not in (0, 1):
            raise ValueError("first protocol has exactly env indexes 0 and 1")
        if phase not in ALLOWED_SOURCES:
            raise ValueError("only fixed phases b00, b02, and validation b01 are allowed")
        if episode_steps not in (128, 800):
            raise ValueError("episode_steps must be 128 for train or 800 for eval")
        if phase == "b01" and episode_steps != 800:
            raise ValueError("validation b01 is allowed only for the frozen 800-step eval")
        self.run_id = run_id
        self.env_index = env_index
        self.phase = phase
        self.episode_steps = episode_steps
        source_name, expected_split = ALLOWED_SOURCES[phase]
        self.source = CASES / source_name
        config = json.loads((self.source / "case_config.json").read_text(encoding="utf-8"))
        if config.get("split") != expected_split or config.get("action_target") != 0.0:
            raise ValueError("direct-CFD source split/action differs from fixed protocol")
        self.split = expected_split
        self.restart_time = float(config["start_time"])
        self.reference = CASES / str(config["source_restart_case"])
        if not math.isclose(float(config["source_restart_time"]), self.restart_time, abs_tol=1e-8):
            raise ValueError("source restart provenance mismatch")
        self.prehistory_times, self.prehistory_forces, self.prehistory_sources = (
            actual_causal_prehistory(self.reference, self.restart_time)
        )
        self.initial_observation, self.initial_observation_sources = total_drag_observation_at(
            self.reference, self.restart_time, 0.0
        )
        self.journal = journal.resolve()
        if not self.journal.is_relative_to(PROJECT / "artifacts" / "direct_cfd"):
            raise ValueError("worker journal must stay under artifacts/direct_cfd")
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.step_index = 0
        self.case: Path | None = None
        self.previous_omega = 0.0
        self.container_name = f"direct-cfd-{run_id}-env{env_index}"
        self.container_started = False

    def append_journal(self, payload: dict) -> None:
        record = {"wall_time": time.time(), "env_index": self.env_index, **payload}
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def start_container(self) -> None:
        inspect = subprocess.run(
            ["docker", "container", "inspect", self.container_name],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            raise FileExistsError(f"refusing to reuse container {self.container_name}")
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--cpus",
            "1",
            "--memory",
            "8g",
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={CFD},dst=/case",
            "--workdir",
            "/case",
            OPENFOAM_IMAGE,
            "bash",
            "-lc",
            "while :; do sleep 3600; done",
        ]
        completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"persistent OpenFOAM container failed: {completed.stderr[-1200:]}")
        self.container_started = True
        self.append_journal(
            {
                "event": "container_started",
                "container_name": self.container_name,
                "container_id": completed.stdout.strip(),
                "image": OPENFOAM_IMAGE,
            }
        )

    def stop_container(self) -> None:
        if not self.container_started:
            return
        completed = subprocess.run(
            ["docker", "rm", "--force", self.container_name],
            cwd=PROJECT,
            text=True,
            capture_output=True,
        )
        self.append_journal(
            {
                "event": "container_stopped",
                "container_name": self.container_name,
                "returncode": completed.returncode,
            }
        )
        self.container_started = False

    def _new_case(self) -> Path:
        self.episode += 1
        name = (
            f"direct_cfd_{self.run_id}_env{self.env_index}_"
            f"ep{self.episode:04d}_{self.phase}"
        )
        target = CASES / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite preserved episode case {target}")
        stage = CASES / f".{name}.staging_{os.getpid()}"
        if stage.exists():
            raise FileExistsError(stage)
        stage.mkdir()
        try:
            shutil.copytree(self.source / "constant", stage / "constant")
            shutil.copytree(self.source / "system", stage / "system")
            shutil.copytree(self.source / f"{self.restart_time:g}", stage / f"{self.restart_time:g}")
            end_time = self.restart_time + self.episode_steps * CONTROL_DT
            velocity = stage / f"{self.restart_time:g}" / "U"
            velocity.write_text(
                replace_rear_patch(
                    velocity.read_text(encoding="utf-8"),
                    [(self.restart_time, 0.0), (end_time, 0.0)],
                ),
                encoding="utf-8",
            )
            control = stage / "system" / "controlDict"
            substitute(control, "startTime", self.restart_time)
            substitute(control, "endTime", end_time)
            substitute(control, "writeInterval", CONTROL_DT)
            metadata = {
                "case": name,
                "status": "direct_cfd_ppo_episode_initialized",
                "split": self.split,
                "phase": self.phase,
                "source_case": self.source.name,
                "source_restart_case": self.reference.name,
                "source_restart_time": self.restart_time,
                "prehistory_sources": self.prehistory_sources,
                "observation_sources": self.initial_observation_sources,
                "protocol": {
                    "episode_steps": self.episode_steps,
                    "control_dt": CONTROL_DT,
                    "action_limit": ACTION_LIMIT,
                    "max_delta_omega": MAX_DELTA_OMEGA,
                },
                "scientific_scope": (
                    "independent validation-phase paired physical check; not training data, "
                    "frozen-test evidence, or final paper claim"
                    if self.split == "validation"
                    else "train-only direct real-CFD RL; not physical gate evidence"
                ),
            }
            (stage / "case_config.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            stage.rename(target)
        except Exception:
            # A never-promoted hidden staging directory is safe to leave for
            # forensic inspection; do not perform broad or recursive cleanup.
            raise
        return target

    def reset(self) -> dict:
        if available_memory_gib() < MEMORY_FLOOR_GIB:
            raise RuntimeError("MemAvailable below 20 GiB before reset")
        disk_free_gib = shutil.disk_usage(PROJECT).free / 1024**3
        if disk_free_gib < DISK_FREE_FLOOR_GIB:
            raise RuntimeError("free disk space below 200 GiB before episode reset")
        self.case = self._new_case()
        self.step_index = 0
        self.previous_omega = 0.0
        response = {
            "ok": True,
            "observation": self.initial_observation.tolist(),
            "time": self.restart_time,
            "prehistory_times": self.prehistory_times,
            "prehistory_forces": self.prehistory_forces,
            "worker_info": {
                "episode": self.episode,
                "case": self.case.name,
                "phase": self.phase,
                "source_restart_time": self.restart_time,
                "prehistory_latest_time": self.prehistory_times[-1],
                "prehistory_sources": self.prehistory_sources,
                "disk_free_gib_before_reset": disk_free_gib,
            },
        }
        self.append_journal({"event": "reset", **response["worker_info"]})
        return response

    def step(self, requested: float, iteration: int) -> dict:
        if self.case is None:
            raise RuntimeError("step requested before reset")
        if iteration != self.step_index:
            raise ValueError(f"iteration mismatch: {iteration} != {self.step_index}")
        if not math.isfinite(requested) or abs(requested) > ACTION_LIMIT + 1e-6:
            raise ValueError("requested action outside canonical support")
        if available_memory_gib() < MEMORY_FLOOR_GIB:
            raise RuntimeError("MemAvailable below 20 GiB before CFD step")
        applied = min(
            max(requested, self.previous_omega - MAX_DELTA_OMEGA),
            self.previous_omega + MAX_DELTA_OMEGA,
        )
        applied = min(max(applied, -ACTION_LIMIT), ACTION_LIMIT)
        start = round(self.restart_time + self.step_index * CONTROL_DT, 10)
        end = round(start + CONTROL_DT, 10)
        if not math.isclose(latest_time(self.case), start, abs_tol=2e-6):
            raise ValueError("episode case is not at the expected restart time")
        velocity = self.case / f"{start:g}" / "U"
        velocity.write_text(
            replace_rear_patch(
                velocity.read_text(encoding="utf-8"),
                [(start, self.previous_omega), (end, applied)],
            ),
            encoding="utf-8",
        )
        control = self.case / "system" / "controlDict"
        substitute(control, "startTime", start)
        substitute(control, "endTime", end)
        log_path = self.case / f"log.pimpleFoam.direct_cfd_{self.step_index + 1:04d}"
        if log_path.exists():
            raise FileExistsError(log_path)
        started = time.perf_counter()
        command = [
            "docker",
            "exec",
            self.container_name,
            "bash",
            "-lc",
            (
                "source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1 "
                '&& exec pimpleFoam -case "$1"'
            ),
            "direct-cfd-segment",
            f"/case/cases/{self.case.name}",
        ]
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT
            )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not math.isclose(latest_time(self.case), end, abs_tol=2e-6):
            raise ValueError("OpenFOAM segment did not reach its expected end time")
        health = log_health(log_path)
        if not health["solver_ended_cleanly"] or health["steps"] != 20:
            raise ValueError(f"bad OpenFOAM segment: {health}")
        if (
            health["max_courant"] >= MAX_COURANT
            or health["max_abs_global_continuity_per_step"] >= MAX_CONTINUITY
        ):
            raise ValueError(f"unsafe OpenFOAM segment: {health}")
        observation, observation_sources = total_drag_observation_at(self.case, end, applied)
        if np.max(np.abs(observation[64:68])) > HARD_FORCE_LIMIT:
            raise FloatingPointError("real-CFD force safety guard exceeded")
        self.step_index += 1
        delta = applied - self.previous_omega
        self.previous_omega = applied
        worker_info = {
            "episode": self.episode,
            "step": self.step_index,
            "case": self.case.name,
            "phase": self.phase,
            "solver_health": health,
            "solver_wall_seconds": elapsed,
            "mem_available_gib": available_memory_gib(),
            "observation_sources": observation_sources,
        }
        self.append_journal(
            {
                "event": "step",
                **worker_info,
                "requested_omega": requested,
                "applied_omega": applied,
                "applied_delta_omega": delta,
                "cfd_time": end,
                "forces": observation[64:68].tolist(),
            }
        )
        return {
            "ok": True,
            "observation": observation.tolist(),
            "time": end,
            "applied_omega": applied,
            "rate_limited": not math.isclose(requested, applied, abs_tol=1e-10),
            "worker_info": worker_info,
        }


def serve(socket_path: Path, worker: OpenFOAMWorker) -> None:
    socket_path = socket_path.resolve()
    if not str(socket_path.parent).startswith("/tmp/dcfd_"):
        raise ValueError("socket must stay in the launcher's short run-owned /tmp directory")
    if socket_path.exists():
        raise FileExistsError(socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(1)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        server.close()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    worker.start_container()
    try:
        while not stopping:
            try:
                connection, _ = server.accept()
            except OSError:
                break
            with connection, connection.makefile("rwb") as stream:
                while not stopping:
                    line = stream.readline(1024 * 1024 + 1)
                    if not line:
                        break
                    try:
                        if len(line) > 1024 * 1024:
                            raise ValueError("request exceeds 1 MiB")
                        request = json.loads(line)
                        command = request.get("command")
                        if command == "hello":
                            response = {
                                "ok": True,
                                "phase": worker.phase,
                                "container": worker.container_name,
                                "protocol": "direct_cfd_jsonl_v1",
                            }
                        elif command == "reset":
                            response = worker.reset()
                        elif command == "step":
                            response = worker.step(
                                float(request["requested_omega"]), int(request["iteration"])
                            )
                        else:
                            raise ValueError(f"unsupported command: {command}")
                    except Exception as error:
                        worker.append_journal(
                            {
                                "event": "request_failed",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        response = {
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    stream.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                    stream.flush()
    finally:
        worker.stop_container()
        server.close()
        # Socket removal is narrow and recoverable state is not stored there.
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-index", type=int, required=True)
    parser.add_argument("--phase", choices=tuple(ALLOWED_SOURCES), required=True)
    parser.add_argument("--episode-steps", type=int, choices=(128, 800), default=128)
    args = parser.parse_args()
    worker = OpenFOAMWorker(
        run_id=args.run_id,
        env_index=args.env_index,
        phase=args.phase,
        journal=args.journal,
        episode_steps=args.episode_steps,
    )
    serve(args.socket, worker)


if __name__ == "__main__":
    main()
