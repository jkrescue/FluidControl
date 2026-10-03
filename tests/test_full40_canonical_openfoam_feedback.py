from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_full40_canonical_ppo_openfoam_feedback.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "run_full40_canonical_ppo_openfoam_feedback.py"
SPEC = importlib.util.spec_from_file_location("canonical_openfoam_feedback", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(tmp_path: Path, monkeypatch):
    cases = tmp_path / "cases"
    monkeypatch.setattr(MODULE, "CASES", cases)
    data = tmp_path / "full40"
    data.mkdir()
    write_json(data / "manifest.json", {"profile": "matched_start_full40_v1"})
    write_json(data / "normalization.json", {"state_mean": [0.0, 0.0, 0.0]})
    checkpoint = "a" * 64
    readiness = tmp_path / "readiness.json"
    write_json(
        readiness,
        {
            "status": "FULL40_CANONICAL_PPO_EXECUTION_READY",
            "checkpoint_sha256": checkpoint,
            "dev30_full40_promotion_lineage": {
                "full40_manifest_sha256": MODULE.sha256(data / "manifest.json"),
                "full40_normalization_sha256": MODULE.sha256(
                    data / "normalization.json"
                ),
            },
        },
    )
    gate = tmp_path / "fno_gate.json"
    write_json(
        gate,
        {
            "status": "FULL40_VALIDATION_SURROGATE_READINESS_PASS",
            "checkpoint_sha256": checkpoint,
            "data_manifest_sha256": MODULE.sha256(data / "manifest.json"),
            "normalization_sha256": MODULE.sha256(data / "normalization.json"),
        },
    )
    policy = tmp_path / "ppo_00002048.zip"
    policy.write_bytes(b"real-stable-baselines-checkpoint-fixture")
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "status": "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE",
            "frozen_test_accessed": False,
            "physicsnemo_checkpoint_sha256": checkpoint,
            "iterations": [
                {
                    "timesteps": 2048,
                    "checkpoint": str(policy.resolve()),
                    "checkpoint_sha256": MODULE.sha256(policy),
                }
            ],
        },
    )
    lineage = MODULE.validate_policy_and_gates(
        policy=policy,
        ppo_audit=audit,
        ppo_readiness=readiness,
        fno_gate=gate,
        data=data,
    )
    start = 84.0
    source = cases / "tandem_backward_dt005" / "84"
    source.mkdir(parents=True)
    for field in MODULE.STATE_FIELDS:
        (source / field).write_text(f"real-{field}\n")
    source_case = source.parent
    for relative in MODULE.SOURCE_CONFIGURATION_FILES:
        target = source_case / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"reviewed-{relative}\n", encoding="utf-8")
    end = start + MODULE.MIN_STEPS * MODULE.CONTROL_INTERVAL
    declaration = {
        "status": "FULL40_CANONICAL_OPENFOAM_FEEDBACK_PREDECLARED_NO_EXECUTION",
        "geometry": "tandem circular cylinders Re=100 L/D=5 rear rotation",
        "solver": "OpenFOAM v2512 pimpleFoam",
        "control_interval": 0.1,
        "solver_delta_t": 0.005,
        "steps": 800,
        "analysis_duration": 60.0,
        "max_abs_omega": 0.75,
        "max_abs_domega_dt": 1.0,
        "frozen_test_accessed": False,
        "source_restart_case": "tandem_backward_dt005",
        "source_restart_time": start,
        "source_state_sha256": {
            field: MODULE.sha256(source / field) for field in MODULE.STATE_FIELDS
        },
        "source_configuration_sha256": {
            relative: MODULE.sha256(source_case / relative)
            for relative in MODULE.SOURCE_CONFIGURATION_FILES
        },
        "analysis_window": [end - 60.0, end],
        "pair": {
            "feedback": "canonical_ppo_feedback_b01_policy_v1",
            "zero": "canonical_ppo_feedback_b01_zero_v1",
        },
        "lineage_sha256": lineage,
    }
    predeclaration = tmp_path / "predeclaration.json"
    write_json(predeclaration, declaration)
    resources = {
        "spark_mem_available_gib": 100.0,
        "spark_disk_free_gib": 500.0,
        "spark_conflicting_processes": [],
        "worker_reachable": True,
        "worker_mem_available_gib": 100.0,
        "worker_disk_free_gib": 500.0,
        "worker_conflicting_processes": [],
        "runtime_image_id": MODULE.RUNTIME_IMAGE_ID,
        "openfoam_image_available": True,
        "hydrogym_commit": MODULE.HYDROGYM_COMMIT,
    }
    return {
        "data": data,
        "policy": policy,
        "audit": audit,
        "readiness": readiness,
        "gate": gate,
        "lineage": lineage,
        "predeclaration": predeclaration,
        "resources": resources,
    }


def test_preflight_binds_policy_data_predeclaration_and_resources(
    tmp_path, monkeypatch
) -> None:
    item = fixture(tmp_path, monkeypatch)
    result = MODULE.build_preflight(
        policy=item["policy"],
        ppo_audit=item["audit"],
        ppo_readiness=item["readiness"],
        fno_gate=item["gate"],
        data=item["data"],
        predeclaration=item["predeclaration"],
        predeclaration_sha256=MODULE.sha256(item["predeclaration"]),
        output=tmp_path / "artifacts/result",
        resources=item["resources"],
    )
    assert result["status"] == "CANONICAL_REAL_OPENFOAM_FEEDBACK_READY"
    assert result["execution_performed"] is False
    assert result["frozen_test_accessed_or_enumerated"] is False
    assert result["lineage"]["policy_sha256"] == MODULE.sha256(item["policy"])


def test_tampered_policy_and_busy_nodes_fail_closed(tmp_path, monkeypatch) -> None:
    item = fixture(tmp_path, monkeypatch)
    item["policy"].write_bytes(b"tampered")
    resources = dict(item["resources"])
    resources["spark_conflicting_processes"] = ["pimpleFoam existing-case"]
    result = MODULE.build_preflight(
        policy=item["policy"],
        ppo_audit=item["audit"],
        ppo_readiness=item["readiness"],
        fno_gate=item["gate"],
        data=item["data"],
        predeclaration=item["predeclaration"],
        predeclaration_sha256=MODULE.sha256(item["predeclaration"]),
        output=tmp_path / "artifacts/result",
        resources=resources,
    )
    assert result["status"] == "CANONICAL_REAL_OPENFOAM_FEEDBACK_BLOCKED"
    assert any("policy_or_gate_invalid" in row for row in result["blockers"])
    assert "resource_spark_idle_failed" in result["blockers"]


def test_predeclaration_sha_and_final_60_window_are_immutable(
    tmp_path, monkeypatch
) -> None:
    item = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="SHA"):
        MODULE.validate_predeclaration(
            item["predeclaration"], "0" * 64, lineage=item["lineage"]
        )
    document = json.loads(item["predeclaration"].read_text())
    document["analysis_window"][0] += 0.1
    write_json(item["predeclaration"], document)
    with pytest.raises(ValueError, match="final 60"):
        MODULE.validate_predeclaration(
            item["predeclaration"],
            MODULE.sha256(item["predeclaration"]),
            lineage=item["lineage"],
        )


def test_predeclaration_binds_mesh_solver_and_force_configuration(
    tmp_path, monkeypatch
) -> None:
    item = fixture(tmp_path, monkeypatch)
    control = (
        MODULE.CASES
        / "tandem_backward_dt005"
        / "system"
        / "controlDict"
    )
    control.write_text("tampered force functions\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        MODULE.validate_predeclaration(
            item["predeclaration"],
            MODULE.sha256(item["predeclaration"]),
            lineage=item["lineage"],
        )


def test_69d_policy_boundary_is_cpu_only_and_sha_bound() -> None:
    policy = MODULE.PROJECT / "artifacts/hydrogym/test/ppo.zip"
    marker = {
        "requested_omega": 0.25,
        "observation_channels": 69,
        "deterministic": True,
        "policy_sha256": "a" * 64,
    }
    process = MagicMock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO(
        "container startup banner\n"
        + 2 * ("CANONICAL_POLICY_ACTION_JSON=" + json.dumps(marker) + "\n")
    )
    process.wait.return_value = 0
    selector = MagicMock()
    selector.__enter__.return_value = selector
    selector.select.return_value = [(object(), object())]
    with (
        patch.object(MODULE.subprocess, "Popen", return_value=process) as popen,
        patch.object(MODULE.selectors, "DefaultSelector", return_value=selector),
    ):
        session = MODULE.PolicyInferenceSession(policy, "a" * 64)
        first, _ = session.request(np.zeros(69, dtype=np.float32))
        second, _ = session.request(np.ones(69, dtype=np.float32))
        request_lines = process.stdin.getvalue().splitlines()
        session.close()
    assert first == second == 0.25
    assert popen.call_count == 1
    assert len(request_lines) == 2
    command = popen.call_args.args[0]
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert "--gpus" not in command
    assert "--serve-jsonl" in command
    assert any(
        item.endswith("/infer_full40_canonical_ppo_action.py") for item in command
    )


def test_openfoam_pair_containers_start_once_and_exec_each_segment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE, "CASES", tmp_path / "cases")
    names = ("canonical_ppo_feedback_pair_policy", "canonical_ppo_feedback_pair_zero")
    for name in names:
        (MODULE.CASES / name).mkdir(parents=True)
    completed = MagicMock(returncode=0, stdout="container-id\n", stderr="")
    processes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    with (
        patch.object(MODULE.subprocess, "run", return_value=completed) as run,
        patch.object(MODULE.subprocess, "Popen", side_effect=processes) as popen,
        MODULE.OpenFOAMPairSession(names) as session,
    ):
        launched_1 = session.launch(1)
        launched_2 = session.launch(2)
        for _, handle, _ in launched_1 + launched_2:
            handle.close()
    docker_runs = [
        call.args[0] for call in run.call_args_list if call.args[0][:2] == ["docker", "run"]
    ]
    docker_stops = [
        call.args[0]
        for call in run.call_args_list
        if call.args[0][:2] == ["docker", "stop"]
    ]
    assert len(docker_runs) == 2
    assert len(docker_stops) == 2
    assert popen.call_count == 4
    assert all(call.args[0][:2] == ["docker", "exec"] for call in popen.call_args_list)


def test_action_guard_is_exact_full40_rate_contract() -> None:
    action = MODULE.apply_action_rate_limit(0.75, 0.0)
    assert action["applied_omega"] == pytest.approx(0.1)
    assert action["applied_abs_rate"] == pytest.approx(1.0)
    assert action["rate_limited"] is True


def test_final_window_requires_every_exact_solver_time(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "ANALYSIS_DURATION", 0.02)
    monkeypatch.setattr(MODULE, "SOLVER_DT", 0.005)
    grid = np.arange(5, dtype=np.float64) * 0.005 + 10.0
    front = np.column_stack((grid, np.ones(5), np.zeros(5)))
    rear = front.copy()
    MODULE.validate_final_force_grid(front, rear, 10.0)
    rear[2, 0] += 0.001
    with pytest.raises(ValueError, match="force grid differs"):
        MODULE.validate_final_force_grid(front, rear, 10.0)
