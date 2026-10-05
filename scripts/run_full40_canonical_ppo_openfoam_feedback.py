#!/usr/bin/env python3
"""Guarded canonical PPO online feedback on paired real OpenFOAM cases.

Dry-run is the default scientific state. Real execution requires an exact
predeclaration SHA and all full40 FNO/PPO lineage gates; this script never
reads the frozen-test split.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import pickle
import secrets
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CFD = PROJECT / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import log_health
from make_expanded_control_dataset import replace_rear_patch
from make_probe_feedback_case import substitute

sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.canonical_joint_v1 import apply_action_rate_limit
from fluid_control.openfoam_observation import total_drag_observation_at

sys.path.insert(0, str(PROJECT / "scripts"))
from run_tandem_phase_feedback_pair import (
    compare_metrics,
    force_metrics,
    read_force_window,
)

RUNTIME_IMAGE = "fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
RUNTIME_IMAGE_ID = (
    "sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
)
OPENFOAM_IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
HYDROGYM_COMMIT = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
CONTROL_INTERVAL = 0.1
SOLVER_DT = 0.005
ACTION_LIMIT = 0.75
MAX_DELTA_OMEGA = 0.1
MIN_STEPS = 800
ANALYSIS_DURATION = 60.0
SPARK_MEMORY_FLOOR_GIB = 40.0
WORKER_MEMORY_FLOOR_GIB = 40.0
SPARK_DISK_FLOOR_GIB = 200.0
WORKER_DISK_FLOOR_GIB = 100.0
EXECUTION_TOKEN = "EXECUTE_REVIEWED_CANONICAL_OPENFOAM_FEEDBACK"
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")
SOURCE_CONFIGURATION_FILES = (
    "constant/polyMesh/boundary",
    "constant/polyMesh/faces",
    "constant/polyMesh/neighbour",
    "constant/polyMesh/owner",
    "constant/polyMesh/points",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def seconds(begin_ns: int, end_ns: int) -> float:
    return (end_ns - begin_ns) / 1e9


def timing_summary(rows: list[dict]) -> dict:
    """Summarize observed wall durations without treating CFD time as wall time."""
    names = (
        "policy_inference_wall_seconds",
        "action_configuration_wall_seconds",
        "parallel_cfd_wall_seconds",
        "observation_extraction_wall_seconds",
        "progress_record_wall_seconds",
        "control_step_wall_seconds",
    )
    result = {}
    for name in names:
        values = np.asarray([row["wall_timing"][name] for row in rows], dtype=np.float64)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError(f"invalid wall timing samples: {name}")
        result[name] = {
            "samples": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
    misses = sum(bool(row["wall_timing"]["deadline_missed"]) for row in rows)
    result["deadline"] = {
        "wall_deadline_seconds": rows[0]["wall_timing"]["wall_deadline_seconds"],
        "missed_steps": misses,
        "deadline_is_wall_time_not_control_interval": True,
    }
    return result


def wait_solver_processes(launched: list[tuple]) -> list[tuple[int, float]]:
    """Collect both already-launched solvers concurrently, preserving pair order."""
    def wait_one(item):
        process, _, _, started_ns = item
        code = process.wait()
        return code, seconds(started_ns, time.perf_counter_ns())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(wait_one, launched))


def write_final_progress(output: Path, lineage: dict, rows: list[dict]) -> None:
    """Publish a final progress snapshot only after every timing row is complete."""
    if not rows or any(
        row.get("wall_timing", {}).get("timing_record_status") != "COMPLETE"
        or row["wall_timing"].get("progress_record_wall_seconds") is None
        or row["wall_timing"].get("control_step_wall_seconds") is None
        for row in rows
    ):
        raise ValueError("cannot finalize incomplete wall timing rows")
    write_atomic(
        output / "progress.json",
        {
            "status": "CANONICAL_REAL_OPENFOAM_FEEDBACK_RUNNING",
            "completed_steps": len(rows),
            "lineage": lineage,
            "runtime_reuse": {
                "policy_container_starts": 1,
                "openfoam_container_starts": 2,
            },
            "timing_rows_complete": True,
            "rows": rows,
        },
    )


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_atomic(path: Path, payload: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def latest_time(case: Path) -> float:
    times = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                times.append(float(path.name))
            except ValueError:
                pass
    if not times:
        raise ValueError(f"no OpenFOAM time directories: {case}")
    return max(times)


def resource_snapshot() -> dict:
    spark_disk = shutil.disk_usage(PROJECT)
    process_text = subprocess.run(
        ["ps", "-eo", "comm=,args="],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    active = [
        line.strip()
        for line in process_text.splitlines()
        if "pimpleFoam" in line
        or "curate_matched_start" in line
        or "orchestrate_matched_start_curator" in line
    ]
    worker_command = (
        "awk '/MemAvailable:/{print $2}' /proc/meminfo; "
        "df -Pk /home/USER/workspace | tail -1 | awk '{print $4}'; "
        "ps -eo comm=,args= | grep -E '[p]impleFoam|[c]urate_matched_start|"
        "[o]rchestrate_matched_start_curator' || true"
    )
    worker = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "USER@WORKER_HOST",
            worker_command,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    worker_lines = worker.stdout.splitlines()
    runtime_image = subprocess.run(
        ["docker", "image", "inspect", RUNTIME_IMAGE, "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    openfoam_image = subprocess.run(
        ["docker", "image", "inspect", OPENFOAM_IMAGE, "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    hydrogym = subprocess.run(
        ["git", "-C", str(PROJECT / ".tools/hydrogym"), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    worker_memory = (
        float(worker_lines[0]) / 1024**2
        if worker.returncode == 0 and len(worker_lines) >= 2
        else None
    )
    worker_disk = (
        float(worker_lines[1]) / 1024**2
        if worker.returncode == 0 and len(worker_lines) >= 2
        else None
    )
    return {
        "spark_mem_available_gib": available_memory_gib(),
        "spark_disk_free_gib": spark_disk.free / 1024**3,
        "spark_conflicting_processes": active,
        "worker_reachable": worker.returncode == 0,
        "worker_mem_available_gib": worker_memory,
        "worker_disk_free_gib": worker_disk,
        "worker_conflicting_processes": worker_lines[2:] if len(worker_lines) > 2 else [],
        "runtime_image_id": runtime_image.stdout.strip(),
        "openfoam_image_available": openfoam_image.returncode == 0,
        "hydrogym_commit": hydrogym.stdout.strip(),
    }


def validate_dual_policy_contract(
    *, audit: dict, readiness: dict, gate: dict, policy: Path,
    normalization_sha256: str, vecnormalize: Path | None,
    expected_vecnormalize_sha256: str | None,
) -> dict:
    """Bind a dual-trained raw-observation policy before any CFD is staged."""
    def declares_dual(value):
        if isinstance(value, dict):
            return (bool({"dual_control_binding", "dual_manifest_sha256", "dual_model_system", "fno_history_runtime"}.intersection(value))
                    or any(declares_dual(item) for item in value.values()))
        if isinstance(value, list):
            return any(declares_dual(item) for item in value)
        return isinstance(value, str) and (
            value.startswith(("FC_P013_", "FC_P015_", "FC_P018_", "FC_P026_"))
            or value in ("fcp013_independent_force_dual_fno", "fcp015_window_accumulation_dual_fno", "fcp018_reduced_rate_dual_fno", "FC-P015", "FC-P018", "FC-P026", "p026_k1", "p026_k4")
        )

    dual = any(declares_dual(value) for value in (audit, readiness, gate))
    if not dual:
        if vecnormalize is not None or expected_vecnormalize_sha256 is not None:
            raise ValueError("explicit dual VecNormalize arguments require dual policy evidence")
        return {}
    binding = audit.get("dual_control_binding")
    if not isinstance(binding, dict) or binding != readiness.get("dual_control_binding"):
        raise ValueError("PPO audit/readiness dual_control_binding differs or is absent")
    if any("FC_P026_" in json.dumps(value) or "p026_k" in json.dumps(value)
           or "FC-P026" in json.dumps(value) or "fno_history_runtime" in value
           for value in (audit, readiness, gate)):
        from adapt_candidate_ppo_openfoam_readiness import validate_p026_history_binding

        validate_p026_history_binding(binding, audit, readiness)
    required = {
        "status": "DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS",
        "canonical_endpoint_window_dynamic_gates_still_required": True,
        "policy_trained": False, "real_cfd_control_validated": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise ValueError("dual control binding contract differs")
    hashes = ("checkpoint_sha256", "checkpoint_state_sha256", "dual_manifest_sha256",
              "flow_model_sha256", "flow_state_sha256", "posteval_receipt_sha256",
              "training_config_sha256", "normalization_sha256")
    if any(not isinstance(binding.get(key), str) or len(binding[key]) != 64
           or any(character not in "0123456789abcdef" for character in binding[key]) for key in hashes):
        raise ValueError("dual control identity hashes are incomplete")
    if (binding["checkpoint_sha256"] != audit.get("physicsnemo_checkpoint_sha256")
            or binding["normalization_sha256"] != normalization_sha256
            or audit.get("training_executed") is not True):
        raise ValueError("dual policy model/normalization/training identity differs")
    identity_contract = "identity: norm_obs=false, norm_reward=false; preserves legacy PPO numerics"
    if audit.get("vecnormalize_contract") != identity_contract:
        raise ValueError("dual policy requires the canonical identity VecNormalize contract")
    if vecnormalize is None or expected_vecnormalize_sha256 is None:
        raise ValueError("dual policy requires VecNormalize path and expected SHA256")
    # Canonical trainer saves output/checkpoints/policy.zip and output/vecnormalize.pkl.
    # Resolve the caller's host path rather than trusting a /workspace audit path.
    vecnormalize = vecnormalize.resolve()
    expected_path = (policy.resolve().parent.parent / "vecnormalize.pkl").resolve()
    if vecnormalize != expected_path:
        raise ValueError("VecNormalize host path does not belong to this policy output")
    artifact_bytes = vecnormalize.read_bytes()
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    if digest != expected_vecnormalize_sha256 or digest != audit.get("vecnormalize_sha256"):
        raise ValueError("dual policy VecNormalize SHA differs")
    try:
        from stable_baselines3.common.vec_env import VecNormalize

        # Only deserialize the explicitly supplied, SHA-bound training artifact.
        vec = pickle.loads(artifact_bytes)
        if (type(vec) is not VecNormalize or vec.norm_obs is not False or vec.norm_reward is not False
                or vec.observation_space.shape != (69,) or vec.action_space.shape != (1,)
                or not np.array_equal(vec.action_space.low, np.array([-ACTION_LIMIT], dtype=np.float32))
                or not np.array_equal(vec.action_space.high, np.array([ACTION_LIMIT], dtype=np.float32))):
            raise ValueError("saved VecNormalize is not canonical identity normalization")
        probes = np.linspace(-20.0, 20.0, 69, dtype=np.float32)[None]
        if not np.array_equal(vec.normalize_obs(probes), probes):
            raise ValueError("saved VecNormalize alters physical observations")
    except Exception as error:
        raise ValueError(f"cannot verify identity VecNormalize: {error}") from error
    return {
        "dual_control_binding": binding,
        "vecnormalize_sha256": digest,
        "vecnormalize_host_path": str(vecnormalize),
        "vecnormalize_contract": identity_contract,
    }


def validate_policy_and_gates(
    *,
    policy: Path,
    ppo_audit: Path,
    ppo_readiness: Path,
    fno_gate: Path,
    data: Path,
    vecnormalize: Path | None = None,
    expected_vecnormalize_sha256: str | None = None,
) -> dict:
    audit = load_json(ppo_audit)
    readiness = load_json(ppo_readiness)
    gate = load_json(fno_gate)
    manifest = data / "manifest.json"
    normalization = data / "normalization.json"
    if audit.get("status") != "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE":
        raise ValueError("formal canonical PPO audit did not complete")
    if audit.get("frozen_test_accessed") is not False:
        raise ValueError("PPO audit frozen-test contract differs")
    if readiness.get("status") != "FULL40_CANONICAL_PPO_EXECUTION_READY":
        raise ValueError("canonical PPO readiness did not pass")
    if gate.get("status") != "FULL40_VALIDATION_SURROGATE_READINESS_PASS":
        raise ValueError("full40 FNO validation gate did not pass")
    checkpoint_sha = audit.get("physicsnemo_checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha, str)
        or readiness.get("checkpoint_sha256") != checkpoint_sha
        or gate.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("FNO checkpoint lineage differs")
    lineage = readiness.get("dev30_full40_promotion_lineage", {})
    if (
        lineage.get("full40_manifest_sha256") != sha256(manifest)
        or lineage.get("full40_normalization_sha256") != sha256(normalization)
        or gate.get("data_manifest_sha256") != sha256(manifest)
        or gate.get("normalization_sha256") != sha256(normalization)
    ):
        raise ValueError("full40 data lineage differs")
    iterations = audit.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("PPO checkpoint iterations are absent")
    final = iterations[-1]
    if Path(str(final.get("checkpoint", ""))).resolve() != policy.resolve():
        raise ValueError("selected policy is not the final PPO checkpoint")
    policy_sha = sha256(policy)
    if final.get("checkpoint_sha256") != policy_sha:
        raise ValueError("final PPO checkpoint SHA differs")
    dual_lineage = validate_dual_policy_contract(
        audit=audit, readiness=readiness, gate=gate, policy=policy,
        normalization_sha256=sha256(normalization), vecnormalize=vecnormalize,
        expected_vecnormalize_sha256=expected_vecnormalize_sha256,
    )
    return {
        **dual_lineage,
        "policy_sha256": policy_sha,
        "ppo_audit_sha256": sha256(ppo_audit),
        "ppo_readiness_sha256": sha256(ppo_readiness),
        "fno_gate_sha256": sha256(fno_gate),
        "fno_checkpoint_sha256": checkpoint_sha,
        "full40_manifest_sha256": sha256(manifest),
        "full40_normalization_sha256": sha256(normalization),
    }


def validate_predeclaration(
    path: Path,
    expected_sha256: str,
    *,
    lineage: dict,
) -> dict:
    if len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise ValueError("canonical OpenFOAM predeclaration SHA differs")
    document = load_json(path)
    required = {
        "status": "FULL40_CANONICAL_OPENFOAM_FEEDBACK_PREDECLARED_NO_EXECUTION",
        "geometry": "tandem circular cylinders Re=100 L/D=5 rear rotation",
        "solver": "OpenFOAM v2512 pimpleFoam",
        "control_interval": CONTROL_INTERVAL,
        "solver_delta_t": SOLVER_DT,
        "steps": MIN_STEPS,
        "analysis_duration": ANALYSIS_DURATION,
        "max_abs_omega": ACTION_LIMIT,
        "max_abs_domega_dt": 1.0,
        "frozen_test_accessed": False,
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ValueError("canonical OpenFOAM predeclaration contract differs")
    start = float(document.get("source_restart_time", math.nan))
    end = start + MIN_STEPS * CONTROL_INTERVAL
    if document.get("analysis_window") != [end - ANALYSIS_DURATION, end]:
        raise ValueError("predeclared final 60D/U window differs")
    if document.get("lineage_sha256") != lineage:
        raise ValueError("predeclared policy/data lineage differs")
    source_case = str(document.get("source_restart_case", ""))
    if not source_case.startswith("tandem_backward_dt005"):
        raise ValueError("source must be the audited Re100 matched-start baseline")
    source = CASES / source_case / f"{start:g}"
    hashes = document.get("source_state_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(STATE_FIELDS):
        raise ValueError("source-state hash contract differs")
    if any(not (source / field).is_file() or sha256(source / field) != hashes[field] for field in STATE_FIELDS):
        raise ValueError("source matched-start fields differ")
    source_case_path = CASES / source_case
    configuration_hashes = document.get("source_configuration_sha256")
    if not isinstance(configuration_hashes, dict) or set(configuration_hashes) != set(
        SOURCE_CONFIGURATION_FILES
    ):
        raise ValueError("source mesh/solver configuration hash contract differs")
    if any(
        not (source_case_path / relative).is_file()
        or sha256(source_case_path / relative) != configuration_hashes[relative]
        for relative in SOURCE_CONFIGURATION_FILES
    ):
        raise ValueError("source mesh/solver configuration differs")
    pair = document.get("pair", {})
    names = (pair.get("feedback"), pair.get("zero"))
    if (
        not all(isinstance(name, str) and name.startswith("canonical_ppo_feedback_") for name in names)
        or names[0] == names[1]
    ):
        raise ValueError("predeclared pair names differ")
    return document


def build_preflight(
    *,
    policy: Path,
    ppo_audit: Path,
    ppo_readiness: Path,
    fno_gate: Path,
    data: Path,
    predeclaration: Path,
    predeclaration_sha256: str,
    output: Path,
    resources: dict,
    vecnormalize: Path | None = None,
    expected_vecnormalize_sha256: str | None = None,
) -> dict:
    blockers = []
    lineage = None
    declaration = None
    try:
        lineage = validate_policy_and_gates(
            policy=policy,
            ppo_audit=ppo_audit,
            ppo_readiness=ppo_readiness,
            fno_gate=fno_gate,
            data=data,
            vecnormalize=vecnormalize,
            expected_vecnormalize_sha256=expected_vecnormalize_sha256,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"policy_or_gate_invalid:{error}")
    if lineage is not None:
        try:
            declaration = validate_predeclaration(
                predeclaration, predeclaration_sha256, lineage=lineage
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            blockers.append(f"predeclaration_invalid:{error}")
    else:
        blockers.append("predeclaration_blocked_by_lineage")
    if output.exists():
        blockers.append("output_already_exists")
    if declaration is not None:
        for case in declaration["pair"].values():
            if (CASES / case).exists():
                blockers.append(f"case_already_exists:{case}")
    resource_checks = {
        "spark_memory": resources.get("spark_mem_available_gib", 0)
        >= SPARK_MEMORY_FLOOR_GIB,
        "spark_disk": resources.get("spark_disk_free_gib", 0)
        >= SPARK_DISK_FLOOR_GIB,
        "spark_idle": not resources.get("spark_conflicting_processes"),
        "worker_reachable": resources.get("worker_reachable") is True,
        "worker_memory": (resources.get("worker_mem_available_gib") or 0)
        >= WORKER_MEMORY_FLOOR_GIB,
        "worker_disk": (resources.get("worker_disk_free_gib") or 0)
        >= WORKER_DISK_FLOOR_GIB,
        "worker_idle": not resources.get("worker_conflicting_processes"),
        "runtime_image": resources.get("runtime_image_id") == RUNTIME_IMAGE_ID,
        "openfoam_image": resources.get("openfoam_image_available") is True,
        "hydrogym_commit": resources.get("hydrogym_commit") == HYDROGYM_COMMIT,
    }
    blockers.extend(f"resource_{key}_failed" for key, passed in resource_checks.items() if not passed)
    return {
        "status": (
            "CANONICAL_REAL_OPENFOAM_FEEDBACK_READY"
            if not blockers
            else "CANONICAL_REAL_OPENFOAM_FEEDBACK_BLOCKED"
        ),
        "execution_performed": False,
        "lineage": lineage,
        "predeclaration_sha256": (
            sha256(predeclaration) if declaration is not None else None
        ),
        "resource_snapshot": resources,
        "resource_checks": resource_checks,
        "blockers": blockers,
        "feedback_chain": (
            "69D real sensors -> final canonical PPO -> |omega|/slew guard -> "
            "0.1D/U OpenFOAM segment -> four-force/probe feedback"
        ),
        "scientific_scope": (
            "preflight only; no CFD executed and no closed-loop benefit claim"
        ),
        "frozen_test_accessed_or_enumerated": False,
    }


class PolicyInferenceSession:
    """Load the immutable PPO once, then exchange one JSON line per CFD step."""

    def __init__(self, policy: Path, policy_sha: str, timeout: float = 120.0) -> None:
        relative_policy = policy.resolve().relative_to(PROJECT)
        self.policy_sha = policy_sha
        self.timeout = timeout
        self.command = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g",
            "--cpus",
            "2",
            "--memory",
            "8g",
            "--pids-limit",
            "128",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONPATH=/workspace/src:/workspace/scripts",
            "--mount",
            f"type=bind,src={PROJECT},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            RUNTIME_IMAGE,
            "python",
            "-u",
            "scripts/infer_full40_canonical_ppo_action.py",
            "--policy",
            str(Path("/workspace") / relative_policy),
            "--policy-sha256",
            policy_sha,
            "--serve-jsonl",
        ]
        # The session owns this file until ``close``; a lexical context would
        # close it before the persistent child exits.
        self.stderr = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            self.command,
            cwd=PROJECT,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            bufsize=1,
        )

    def __enter__(self):
        return self

    def _error_tail(self) -> str:
        self.stderr.flush()
        self.stderr.seek(0)
        return self.stderr.read()[-1000:]

    def request(self, observation: np.ndarray) -> tuple[float, str]:
        if observation.shape != (69,) or not np.isfinite(observation).all():
            raise ValueError("expected 69 finite real-CFD channels")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("canonical PPO inference pipes are unavailable")
        self.process.stdin.write(json.dumps(observation.tolist()) + "\n")
        self.process.stdin.flush()
        prefix = "CANONICAL_POLICY_ACTION_JSON="
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("canonical PPO inference timed out")
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        "canonical PPO inference exited before a response: "
                        + self._error_tail()
                    )
                if line.startswith(prefix):
                    break
        marker = line.partition(prefix)[2].strip()
        payload = json.loads(marker)
        if (
            payload.get("observation_channels") != 69
            or payload.get("deterministic") is not True
            or payload.get("policy_sha256") != self.policy_sha
        ):
            raise ValueError("canonical PPO inference metadata differs")
        return float(payload["requested_omega"]), marker

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            code = self.process.wait(timeout=10)
        self.stderr.close()
        if code != 0:
            raise RuntimeError(f"canonical PPO inference exited with {code}")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except RuntimeError:
            if exc_type is None:
                raise


def stage_case(name: str, source_case: str, start: float, steps: int, role: str) -> Path:
    target = CASES / name
    source = CASES / source_case
    if target.exists():
        raise FileExistsError(target)
    end = start + steps * CONTROL_INTERVAL
    with tempfile.TemporaryDirectory(prefix=".canonical_feedback_", dir=CASES) as directory:
        stage = Path(directory) / name
        stage.mkdir()
        shutil.copytree(source / "constant", stage / "constant")
        shutil.copytree(source / "system", stage / "system")
        shutil.copytree(source / f"{start:g}", stage / f"{start:g}")
        velocity = stage / f"{start:g}" / "U"
        velocity.write_text(
            replace_rear_patch(
                velocity.read_text(encoding="utf-8"), [(start, 0.0), (end, 0.0)]
            ),
            encoding="utf-8",
        )
        control = stage / "system/controlDict"
        for key, value in (
            ("startTime", start),
            ("endTime", end),
            ("writeInterval", CONTROL_INTERVAL),
        ):
            substitute(control, key, value)
        config = {
            "case": name,
            "status": "canonical_feedback_initialized_not_solved",
            "role": role,
            "source_restart_case": source_case,
            "source_restart_time": start,
            "reynolds_number": 100,
            "geometry": "tandem cylinders D=1 L/D=5 rear rotation only",
            "delta_t": SOLVER_DT,
            "control_interval": CONTROL_INTERVAL,
            "steps": steps,
            "omega_limit": ACTION_LIMIT,
            "delta_omega_limit": MAX_DELTA_OMEGA,
        }
        (stage / "case_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return target


def configure_interval(case: Path, start: float, end: float, before: float, after: float) -> None:
    velocity = case / f"{start:g}" / "U"
    velocity.write_text(
        replace_rear_patch(
            velocity.read_text(encoding="utf-8"), [(start, before), (end, after)]
        ),
        encoding="utf-8",
    )
    substitute(case / "system/controlDict", "startTime", start)
    substitute(case / "system/controlDict", "endTime", end)


class OpenFOAMPairSession:
    """Keep two pinned solver containers alive across all 800 short segments."""

    def __init__(self, names: tuple[str, str]) -> None:
        token = f"{os.getpid()}-{secrets.token_hex(4)}"
        self.names = names
        self.containers = tuple(
            f"canonical-feedback-{token}-{index}" for index in range(len(names))
        )
        self.started: list[str] = []

    def __enter__(self):
        try:
            for container in self.containers:
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "--detach",
                    "--name",
                    container,
                    "--network",
                    "none",
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=512m",
                    "--cpus",
                    "4",
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
                    "sh",
                    "-c",
                    "while :; do sleep 3600; done",
                ]
                completed = subprocess.run(
                    command, cwd=PROJECT, text=True, capture_output=True, check=False
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "persistent OpenFOAM container failed: "
                        + completed.stderr[-1000:]
                    )
                self.started.append(container)
        except Exception:
            self.close()
            raise
        return self

    def launch(self, step: int) -> list[tuple[subprocess.Popen, object, Path, int]]:
        launched = []
        for name, container in zip(self.names, self.containers, strict=True):
            case = CASES / name
            log = case / f"log.pimpleFoam.canonical_feedback_{step:04d}"
            if log.exists():
                raise FileExistsError(log)
            handle = log.open("w", encoding="utf-8")
            command = [
                "docker",
                "exec",
                container,
                "pimpleFoam",
                "-case",
                f"/case/cases/{name}",
            ]
            try:
                started_ns = time.perf_counter_ns()
                process = subprocess.Popen(
                    command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT
                )
            except Exception:
                handle.close()
                for earlier, earlier_handle, _, _ in launched:
                    earlier.terminate()
                    earlier.wait(timeout=10)
                    earlier_handle.close()
                raise
            launched.append((process, handle, log, started_ns))
        return launched

    def close(self) -> None:
        for container in reversed(self.started):
            subprocess.run(
                ["docker", "stop", "--time", "10", container],
                cwd=PROJECT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.started.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def check_segment(case: Path, log: Path, end: float) -> dict:
    if not math.isclose(latest_time(case), end, abs_tol=2e-6):
        raise ValueError(f"{case.name} did not reach {end}")
    health = log_health(log)
    if (
        not health["solver_ended_cleanly"]
        or health["steps"] != 20
        or health["max_courant"] >= 0.8
        or health["max_abs_global_continuity_per_step"] >= 1e-5
    ):
        raise ValueError(f"unsafe OpenFOAM segment: {case.name}: {health}")
    return health


def validate_final_force_grid(front: np.ndarray, rear: np.ndarray, begin: float) -> None:
    expected = round(ANALYSIS_DURATION / SOLVER_DT) + 1
    expected_grid = begin + np.arange(expected, dtype=np.float64) * SOLVER_DT
    if (
        len(front) != expected
        or len(rear) != expected
        or not np.allclose(front[:, 0], expected_grid, rtol=0.0, atol=1e-8)
        or not np.allclose(rear[:, 0], expected_grid, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("final 60D/U force grid differs")


def run_feedback(
    declaration: dict,
    *,
    policy: Path,
    lineage: dict,
    output: Path,
    wall_deadline_seconds: float | None = None,
) -> dict:
    if wall_deadline_seconds is not None and (
        not math.isfinite(wall_deadline_seconds) or wall_deadline_seconds <= 0.0
    ):
        raise ValueError("optional wall deadline must be positive and finite")
    run_started_ns = time.perf_counter_ns()
    pair = declaration["pair"]
    names = (pair["feedback"], pair["zero"])
    start = float(declaration["source_restart_time"])
    steps = int(declaration["steps"])
    cases = (
        stage_case(names[0], declaration["source_restart_case"], start, steps, "ppo"),
        stage_case(names[1], declaration["source_restart_case"], start, steps, "zero"),
    )
    source_hashes = {
        field: [sha256(case / f"{start:g}" / field) for case in cases]
        for field in STATE_FIELDS
    }
    if any(len(set(values)) != 1 for values in source_hashes.values()):
        raise ValueError("paired matched-start fields differ")
    output.mkdir(parents=True)
    observation, initial_sources = total_drag_observation_at(
        CASES / declaration["source_restart_case"], start, 0.0
    )
    previous = 0.0
    rows = []
    with PolicyInferenceSession(
        policy, lineage["policy_sha256"]
    ) as inference, OpenFOAMPairSession(names) as solvers:
        for step in range(1, steps + 1):
            step_started_ns = time.perf_counter_ns()
            step_started_utc = datetime.now(UTC).isoformat()
            if available_memory_gib() < SPARK_MEMORY_FLOOR_GIB:
                raise RuntimeError("Spark MemAvailable fell below 40 GiB")
            interval_start = round(start + CONTROL_INTERVAL * (step - 1), 10)
            interval_end = round(interval_start + CONTROL_INTERVAL, 10)
            if not all(
                math.isclose(latest_time(case), interval_start, abs_tol=2e-6)
                for case in cases
            ):
                raise ValueError(f"paired restart mismatch before step {step}")
            input_sha = array_sha256(observation)
            policy_started_ns = time.perf_counter_ns()
            requested, marker = inference.request(observation)
            policy_ended_ns = time.perf_counter_ns()
            configuration_started_ns = policy_ended_ns
            action = apply_action_rate_limit(requested, previous)
            applied = float(action["applied_omega"])
            configure_interval(cases[0], interval_start, interval_end, previous, applied)
            configure_interval(cases[1], interval_start, interval_end, 0.0, 0.0)
            boundary_sha256 = {
                name: sha256(case / f"{interval_start:g}" / "U")
                for name, case in zip(names, cases, strict=True)
            }
            configuration_ended_ns = time.perf_counter_ns()
            cfd_started_ns = configuration_ended_ns
            launched = solvers.launch(step)
            try:
                completed = wait_solver_processes(launched)
            finally:
                for _, handle, _, _ in launched:
                    handle.close()
            codes = [item[0] for item in completed]
            if any(codes):
                raise RuntimeError(f"paired OpenFOAM segment failed: {codes}")
            health = {
                name: check_segment(case, log, interval_end)
                for name, case, (_, _, log, _) in zip(
                    names, cases, launched, strict=True
                )
            }
            cfd_ended_ns = time.perf_counter_ns()
            observation_started_ns = cfd_ended_ns
            observation, feedback_sources = total_drag_observation_at(
                cases[0], interval_end, applied
            )
            zero_observation, zero_sources = total_drag_observation_at(
                cases[1], interval_end, 0.0
            )
            observation_ended_ns = time.perf_counter_ns()
            row = {
                "step": step,
                "start_time": interval_start,
                "end_time": interval_end,
                "input_observation_sha256": input_sha,
                "output_observation_sha256": array_sha256(observation),
                "policy_marker": marker,
                "applied_boundary_U_sha256": boundary_sha256,
                **action,
                "feedback_forces": {
                    key: float(observation[index])
                    for key, index in zip(
                        ("front_cd", "front_cl", "rear_cd", "rear_cl"),
                        range(64, 68),
                        strict=True,
                    )
                },
                "zero_forces": {
                    key: float(zero_observation[index])
                    for key, index in zip(
                        ("front_cd", "front_cl", "rear_cd", "rear_cl"),
                        range(64, 68),
                        strict=True,
                    )
                },
                "observation_sources": {
                    "feedback": feedback_sources,
                    "zero": zero_sources,
                },
                "solver_health": health,
                "wall_timing": {
                    "timing_record_status": "PENDING_PROGRESS_WRITE",
                    "step_started_utc": step_started_utc,
                    "policy_inference_wall_seconds": seconds(
                        policy_started_ns, policy_ended_ns
                    ),
                    "action_configuration_wall_seconds": seconds(
                        configuration_started_ns, configuration_ended_ns
                    ),
                    "parallel_cfd_wall_seconds": seconds(cfd_started_ns, cfd_ended_ns),
                    "per_branch_cfd_wall_seconds": {
                        name: elapsed
                        for name, (_, elapsed) in zip(names, completed, strict=True)
                    },
                    "observation_extraction_wall_seconds": seconds(
                        observation_started_ns, observation_ended_ns
                    ),
                    "progress_record_wall_seconds": None,
                    "control_step_wall_seconds": None,
                    "wall_deadline_seconds": wall_deadline_seconds,
                    "deadline_missed": False,
                    "control_interval_D_over_U": CONTROL_INTERVAL,
                },
            }
            rows.append(row)
            previous = applied
            record_started_ns = time.perf_counter_ns()
            write_atomic(
                output / "progress.json",
                {
                    "status": "CANONICAL_REAL_OPENFOAM_FEEDBACK_RUNNING",
                    "completed_steps": step,
                    "lineage": lineage,
                    "runtime_reuse": {
                        "policy_container_starts": 1,
                        "openfoam_container_starts": 2,
                    },
                    "rows": rows,
                },
            )
            step_ended_ns = time.perf_counter_ns()
            row["wall_timing"]["progress_record_wall_seconds"] = seconds(
                record_started_ns, step_ended_ns
            )
            row["wall_timing"]["control_step_wall_seconds"] = seconds(
                step_started_ns, step_ended_ns
            )
            row["wall_timing"]["deadline_missed"] = bool(
                wall_deadline_seconds is not None
                and row["wall_timing"]["control_step_wall_seconds"]
                > wall_deadline_seconds
            )
            row["wall_timing"]["timing_record_status"] = "COMPLETE"
    # The live progress file briefly contains the current row as PENDING while
    # its own atomic write is measured. Publish all completed rows once more;
    # this finalization write is intentionally outside per-step timing.
    write_final_progress(output, lineage, rows)
    end = start + steps * CONTROL_INTERVAL
    begin = end - ANALYSIS_DURATION
    metrics = {}
    for label, case in zip(("feedback", "zero"), cases, strict=True):
        front = read_force_window(case, "forceFront", begin, end)
        rear = read_force_window(case, "forceRear", begin, end)
        validate_final_force_grid(front, rear, begin)
        metrics[label] = force_metrics(front, rear)
    comparison = compare_metrics(metrics["feedback"], metrics["zero"])
    omega = np.asarray([row["applied_omega"] for row in rows])
    result = {
        "status": "CANONICAL_REAL_OPENFOAM_FEEDBACK_COMPLETED",
        "scientific_status": (
            "one_predeclared_matched_start_real_cfd_pair_not_generalized_control_proof"
        ),
        "solver": "OpenFOAM v2512 pimpleFoam",
        "geometry": "tandem circular cylinders Re=100 L/D=5 rear rotation",
        "lineage": lineage,
        "predeclaration": declaration,
        "source_state_pair_hashes": source_hashes,
        "initial_observation_sources": initial_sources,
        "steps": steps,
        "analysis_window": [begin, end],
        "runtime_reuse": {
            "policy_container_starts": 1,
            "openfoam_container_starts": 2,
            "openfoam_exec_segments": 2 * steps,
        },
        "wall_timing": {
            "run_wall_seconds": seconds(run_started_ns, time.perf_counter_ns()),
            "summary": timing_summary(rows),
            "scope": (
                "observed execution latency; 0.1 D/U is CFD time, not a wall-time deadline"
            ),
        },
        "metrics": metrics,
        "canonical_comparison": comparison,
        "canonical_joint_gate_pass": comparison["canonical_physical_joint_check"],
        "action_audit": {
            "max_abs_omega": float(np.max(np.abs(omega))),
            "max_abs_delta_omega": float(
                np.max(np.abs(np.diff(np.concatenate(([0.0], omega)))))
            ),
            "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        },
        "feedback_evidence_chain": rows,
        "frozen_test_accessed_or_enumerated": False,
    }
    write_atomic(output / "result.json", result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--expected-vecnormalize-sha256")
    parser.add_argument("--ppo-audit", type=Path, required=True)
    parser.add_argument("--ppo-readiness", type=Path, required=True)
    parser.add_argument("--fno-gate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--predeclaration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wall-deadline-seconds",
        type=float,
        help="optional observational wall deadline; unset by default",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.wall_deadline_seconds is not None and (
        not math.isfinite(args.wall_deadline_seconds)
        or args.wall_deadline_seconds <= 0.0
    ):
        parser.error("--wall-deadline-seconds must be positive and finite")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "artifacts/tandem_cylinders"):
        parser.error("output must stay under artifacts/tandem_cylinders")
    resources = resource_snapshot()
    preflight = build_preflight(
        policy=args.policy.resolve(),
        ppo_audit=args.ppo_audit.resolve(),
        ppo_readiness=args.ppo_readiness.resolve(),
        fno_gate=args.fno_gate.resolve(),
        data=args.data.resolve(),
        predeclaration=args.predeclaration.resolve(),
        predeclaration_sha256=args.predeclaration_sha256,
        output=output,
        resources=resources,
        vecnormalize=args.vecnormalize,
        expected_vecnormalize_sha256=args.expected_vecnormalize_sha256,
    )
    if args.dry_run:
        target = output.with_suffix(".preflight.json")
        write_atomic(target, preflight, exclusive=True)
        print(json.dumps(preflight, indent=2))
        return
    if preflight["status"] != "CANONICAL_REAL_OPENFOAM_FEEDBACK_READY":
        raise RuntimeError(f"execution is blocked: {preflight['blockers']}")
    if os.environ.get("CANONICAL_OPENFOAM_EXECUTION_TOKEN") != EXECUTION_TOKEN:
        raise RuntimeError("explicit reviewed canonical OpenFOAM token is required")
    declaration = validate_predeclaration(
        args.predeclaration.resolve(),
        args.predeclaration_sha256,
        lineage=preflight["lineage"],
    )
    lock_path = PROJECT / "artifacts/tandem_cylinders/.canonical_feedback.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_lineage = validate_policy_and_gates(
            policy=args.policy.resolve(), ppo_audit=args.ppo_audit.resolve(),
            ppo_readiness=args.ppo_readiness.resolve(), fno_gate=args.fno_gate.resolve(),
            data=args.data.resolve(), vecnormalize=args.vecnormalize,
            expected_vecnormalize_sha256=args.expected_vecnormalize_sha256,
        )
        if current_lineage != preflight["lineage"]:
            raise ValueError("policy evidence changed after preflight")
        result = run_feedback(
            declaration,
            policy=args.policy.resolve(),
            lineage=preflight["lineage"],
            output=output,
            wall_deadline_seconds=args.wall_deadline_seconds,
        )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
