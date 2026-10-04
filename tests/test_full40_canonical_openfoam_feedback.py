from __future__ import annotations

import importlib.util
import io
import json
import time
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
        for _, handle, _, started_ns in launched_1 + launched_2:
            assert isinstance(started_ns, int) and started_ns > 0
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


def test_wall_timing_summary_is_observational_and_reports_deadline_misses() -> None:
    rows = []
    for total in (2.0, 3.0, 4.0):
        rows.append(
            {
                "wall_timing": {
                    "policy_inference_wall_seconds": 0.01,
                    "action_configuration_wall_seconds": 0.02,
                    "parallel_cfd_wall_seconds": total - 0.05,
                    "observation_extraction_wall_seconds": 0.01,
                    "progress_record_wall_seconds": 0.01,
                    "control_step_wall_seconds": total,
                    "wall_deadline_seconds": 2.5,
                    "deadline_missed": total > 2.5,
                }
            }
        )
    summary = MODULE.timing_summary(rows)
    assert summary["control_step_wall_seconds"]["median"] == 3.0
    assert summary["control_step_wall_seconds"]["p95"] == pytest.approx(3.9)
    assert summary["deadline"] == {
        "wall_deadline_seconds": 2.5,
        "missed_steps": 2,
        "deadline_is_wall_time_not_control_interval": True,
    }


def test_wall_timing_rejects_nonfinite_measurements() -> None:
    row = {
        "wall_timing": {
            "policy_inference_wall_seconds": np.nan,
            "action_configuration_wall_seconds": 0.1,
            "parallel_cfd_wall_seconds": 1.0,
            "observation_extraction_wall_seconds": 0.1,
            "progress_record_wall_seconds": 0.1,
            "control_step_wall_seconds": 1.3,
            "wall_deadline_seconds": None,
            "deadline_missed": False,
        }
    }
    with pytest.raises(ValueError, match="invalid wall timing"):
        MODULE.timing_summary([row])


def test_pair_wait_collects_both_processes_and_preserves_launch_order() -> None:
    first, second = MagicMock(), MagicMock()

    def wait_first():
        time.sleep(0.02)
        return 3

    def wait_second():
        time.sleep(0.005)
        return 4

    first.wait.side_effect = wait_first
    second.wait.side_effect = wait_second
    started = time.perf_counter_ns()
    result = MODULE.wait_solver_processes(
        [(first, None, Path("feedback.log"), started),
         (second, None, Path("zero.log"), started)]
    )
    assert [item[0] for item in result] == [3, 4]
    assert all(item[1] > 0.0 for item in result)
    first.wait.assert_called_once_with()
    second.wait.assert_called_once_with()


def test_final_progress_rewrites_last_row_with_complete_timing(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    row = {
        "wall_timing": {
            "timing_record_status": "PENDING_PROGRESS_WRITE",
            "progress_record_wall_seconds": None,
            "control_step_wall_seconds": None,
        }
    }
    with pytest.raises(ValueError, match="incomplete wall timing"):
        MODULE.write_final_progress(output, {"policy_sha256": "a" * 64}, [row])
    row["wall_timing"].update(
        timing_record_status="COMPLETE",
        progress_record_wall_seconds=0.01,
        control_step_wall_seconds=2.0,
    )
    MODULE.write_final_progress(output, {"policy_sha256": "a" * 64}, [row])
    stored = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert stored["timing_rows_complete"] is True
    assert stored["rows"][-1]["wall_timing"]["timing_record_status"] == "COMPLETE"
    assert stored["rows"][-1]["wall_timing"]["progress_record_wall_seconds"] == 0.01


def test_instrumented_mock_loop_preserves_actions_observations_and_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE, "CASES", tmp_path / "cases")
    names = ("canonical_ppo_feedback_mock", "canonical_ppo_feedback_mock_zero")
    configured = []
    policy_inputs = []

    def stage(name, source_case, start, steps, role):
        del source_case, steps, role
        case = MODULE.CASES / name
        initial = case / f"{start:g}"
        initial.mkdir(parents=True)
        for field in MODULE.STATE_FIELDS:
            (initial / field).write_text(f"same-{field}\n", encoding="utf-8")
        return case

    def configure(case, start, end, before, after):
        configured.append((case.name, start, end, before, after))
        next_time = case / f"{end:g}"
        next_time.mkdir()
        (next_time / "U").write_text("next-U\n", encoding="utf-8")

    class FakeInference:
        def __init__(self, policy, policy_sha):
            del policy, policy_sha

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def request(self, observation):
            assert observation.shape == (69,)
            policy_inputs.append(observation.copy())
            return 0.75, '{"requested_omega": 0.75}'

    class FakeSolvers:
        def __init__(self, pair_names):
            assert pair_names == names

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def launch(self, step):
            launched = []
            for name in names:
                process = MagicMock()
                process.wait.return_value = 0
                launched.append(
                    (
                        process,
                        MagicMock(),
                        tmp_path / f"{name}-{step}.log",
                        time.perf_counter_ns(),
                    )
                )
            return launched

    observations = []
    for omega, force in (
        (0.0, [1.0, 2.0, 3.0, 4.0]),
        (0.1, [1.1, 2.1, 3.1, 4.1]),
        (0.0, [1.0, 2.0, 3.0, 4.0]),
        (0.2, [1.2, 2.2, 3.2, 4.2]),
        (0.0, [1.0, 2.0, 3.0, 4.0]),
    ):
        value = np.zeros(69, dtype=np.float32)
        value[64:68] = force
        value[68] = omega
        observations.append((value, {"mock": True}))
    feedback_metrics = {"total_cd_mean": 2.0}
    zero_metrics = {"total_cd_mean": 2.2}
    comparison = {"canonical_physical_joint_check": True}
    declaration = {
        "pair": {"feedback": names[0], "zero": names[1]},
        "source_restart_case": "source",
        "source_restart_time": 0.0,
        "steps": 2,
    }
    with (
        patch.object(MODULE, "stage_case", side_effect=stage),
        patch.object(MODULE, "configure_interval", side_effect=configure),
        patch.object(MODULE, "PolicyInferenceSession", FakeInference),
        patch.object(MODULE, "OpenFOAMPairSession", FakeSolvers),
        patch.object(MODULE, "total_drag_observation_at", side_effect=observations),
        patch.object(MODULE, "available_memory_gib", return_value=100.0),
        patch.object(MODULE, "check_segment", return_value={"clean": True}),
        patch.object(MODULE, "read_force_window", return_value=np.zeros((2, 3))),
        patch.object(MODULE, "validate_final_force_grid"),
        patch.object(
            MODULE, "force_metrics", side_effect=(feedback_metrics, zero_metrics)
        ),
        patch.object(MODULE, "compare_metrics", return_value=comparison),
    ):
        result = MODULE.run_feedback(
            declaration,
            policy=tmp_path / "policy.zip",
            lineage={"policy_sha256": "a" * 64},
            output=tmp_path / "output",
            wall_deadline_seconds=100.0,
        )

    assert [row["applied_omega"] for row in result["feedback_evidence_chain"]] == [
        0.1,
        0.2,
    ]
    np.testing.assert_array_equal(policy_inputs[0], observations[0][0])
    np.testing.assert_array_equal(policy_inputs[1], observations[1][0])
    assert [row["requested_omega"] for row in result["feedback_evidence_chain"]] == [
        0.75,
        0.75,
    ]
    assert [
        row["input_observation_sha256"] for row in result["feedback_evidence_chain"]
    ] == [MODULE.array_sha256(observations[0][0]), MODULE.array_sha256(observations[1][0])]
    assert [
        row["output_observation_sha256"] for row in result["feedback_evidence_chain"]
    ] == [MODULE.array_sha256(observations[1][0]), MODULE.array_sha256(observations[3][0])]
    assert configured == [
        (names[0], 0.0, 0.1, 0.0, 0.1),
        (names[1], 0.0, 0.1, 0.0, 0.0),
        (names[0], 0.1, 0.2, 0.1, 0.2),
        (names[1], 0.1, 0.2, 0.0, 0.0),
    ]
    assert result["metrics"] == {"feedback": feedback_metrics, "zero": zero_metrics}
    assert result["canonical_joint_gate_pass"] is True
    for row in result["feedback_evidence_chain"]:
        assert row["wall_timing"]["timing_record_status"] == "COMPLETE"
        assert row["wall_timing"]["deadline_missed"] is False
        for key, value in row["wall_timing"].items():
            if (
                key.endswith("_wall_seconds")
                and value is not None
                and isinstance(value, int | float)
            ):
                assert np.isfinite(value) and value >= 0.0
    final_progress = json.loads(
        (tmp_path / "output/progress.json").read_text(encoding="utf-8")
    )
    assert final_progress["timing_rows_complete"] is True
    assert all(
        row["wall_timing"]["timing_record_status"] == "COMPLETE"
        for row in final_progress["rows"]
    )
