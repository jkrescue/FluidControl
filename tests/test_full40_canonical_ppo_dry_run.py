from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/train_full40_hydrogym_ppo_canonical.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_ppo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phase_contract_is_exact_and_excludes_frozen() -> None:
    assert MODULE.TRAIN_CASES == (
        "matched_start_acquisition_train_b00_zero",
        "matched_start_acquisition_train_b02_zero",
        "matched_start_acquisition_train_b04_zero",
        "matched_start_acquisition_train_b06_zero",
    )
    assert MODULE.VALIDATION_CASES == (
        "matched_start_acquisition_validation_b01_zero",
        "matched_start_acquisition_validation_b05_zero",
    )
    assert not any("frozen" in case for case in (*MODULE.TRAIN_CASES, *MODULE.VALIDATION_CASES))


def test_preflight_fails_closed_before_runtime_imports_and_never_reads_frozen(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "full40"
    for split in ("train", "validation", "frozen_test"):
        (data / split).mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "profile": MODULE.PROFILE,
                "max_abs_omega": 0.75,
                "trajectory_counts": MODULE.EXPECTED_COUNTS,
            }
        )
    )
    for split, cases in (
        ("train", MODULE.TRAIN_CASES),
        ("validation", MODULE.VALIDATION_CASES),
    ):
        for case in cases:
            (data / split / f"{case}.h5").touch()
    (data / "frozen_test" / "must_not_be_opened.h5").write_text("invalid")
    config = tmp_path / "model.yaml"
    config.write_text("model: full40\n")

    class FailingGate:
        @staticmethod
        def audit(*args, **kwargs):
            raise ValueError("endpoint evidence missing")

    monkeypatch.setattr(MODULE, "_load_gate_module", lambda: FailingGate)
    result = MODULE.preflight(
        data=data,
        config=config,
        checkpoint_dir=tmp_path / "checkpoint",
        validation_gate=tmp_path / "gate.json",
        validation_report=tmp_path / "report.json",
        validation_segments=tmp_path / "segments.json",
        predeclaration=tmp_path / "predeclaration.json",
        window_gate=tmp_path / "window.json",
        dynamic_gate=tmp_path / "dynamic.json",
        baselines=tmp_path / "baselines.json",
        image_id="sha256:missing",
        runtime_image_id="sha256:missing-runtime",
        episode_steps=100,
    )
    assert result["status"] == "FULL40_CANONICAL_PPO_EXECUTION_BLOCKED"
    assert any("validation_gate_invalid" in item for item in result["blockers"])
    assert result["training_executed"] is False
    assert result["frozen_test_directory_enumerated_or_opened"] is False


def test_evidence_gate_rejects_tampering(tmp_path) -> None:
    producer = tmp_path / "producer.py"
    evidence = tmp_path / "evidence.json"
    producer.write_text("# reviewed producer\n")
    evidence.write_text("{}\n")
    gate = tmp_path / "window.json"
    gate.write_text(
        json.dumps(
            {
                "status": MODULE.WINDOW_GATE_STATUS,
                "profile": MODULE.PROFILE,
                "checkpoint_sha256": "a" * 64,
                "frozen_test_accessed": False,
                "validation_phases": ["b01", "b05"],
                "causal_window_seconds": 6.15,
                "total_drag_window_fidelity_pass": True,
                "rear_cl_fluctuation_window_fidelity_pass": True,
                "rear_cl_mean_bias_window_fidelity_pass": True,
                "producer_script": str(producer),
                "producer_script_sha256": MODULE.sha256(producer),
                "evidence_path": str(evidence),
                "evidence_sha256": MODULE.sha256(evidence),
            }
        )
    )
    MODULE.validate_evidence_gate(
        gate,
        expected_status=MODULE.WINDOW_GATE_STATUS,
        checkpoint_sha256="a" * 64,
    )
    evidence.write_text('{"tampered": true}\n')
    with pytest.raises(ValueError, match="evidence"):
        MODULE.validate_evidence_gate(
            gate,
            expected_status=MODULE.WINDOW_GATE_STATUS,
            checkpoint_sha256="a" * 64,
        )


def test_train_only_smoke_reads_exact_train20_zero_baselines(tmp_path) -> None:
    phases = {}
    for phase in MODULE.TRAIN_PHASES:
        phases[f"b{phase}"] = {
            "same_phase_zero": {
                "case": f"matched_start_acquisition_train_b{phase}_zero",
                "metrics": {
                    "mean_cd_total": 2.3,
                    "rms_cl_rear_fluctuation": 1.1,
                },
            }
        }
    path = tmp_path / "train20.json"
    path.write_text(
        json.dumps(
            {
                "status": "FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY",
                "scope": {
                    "split": "train",
                    "phase_bins": [0, 2, 4, 6],
                    "validation_or_frozen_results_read": False,
                },
                "phases": phases,
            }
        )
    )
    result = MODULE.validate_train20_baselines(path)
    assert set(result) == {"b00", "b02", "b04", "b06"}
    assert result["b00"]["total_drag"] == 2.3


def test_train_diagnostics_are_checkpoint_local_and_not_a_physics_claim() -> None:
    accumulator = MODULE.TrainingDiagnosticsAccumulator()
    accumulator.record(
        [
            {
                "applied_omega": -0.1,
                "rate_limited": True,
                "reward_drag_screen": 0.2,
            },
            {
                "applied_omega": 0.3,
                "rate_limited": False,
                "reward_drag_screen": -0.4,
                "episode": {"r": -3.5, "l": 100},
            },
        ]
    )
    result = accumulator.snapshot(reset=True)
    assert result["environment_steps"] == 2
    assert result["episodes_completed"] == 1
    assert result["episode_return"]["mean"] == -3.5
    assert result["episode_length"]["mean"] == 100
    assert result["applied_omega"]["minimum"] == -0.1
    assert result["applied_omega"]["maximum"] == 0.3
    assert result["rate_limited_fraction"] == 0.5
    assert result["canonical_reward_component_mean_per_environment_step"][
        "reward_drag_screen"
    ] == pytest.approx(-0.1)
    assert "not validation" in result["diagnostic_scope"]
    assert accumulator.snapshot()["environment_steps"] == 0


def test_source_has_real_execute_path_but_no_frozen_enumeration() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "def execute(" in text
    assert 'mode.add_argument("--execute"' in text
    assert "PPO(" in text
    assert "data / \"frozen_test\"" not in text


def test_shell_dry_run_branch_does_not_request_gpu() -> None:
    shell = (
        SCRIPT.with_name("run_full40_canonical_ppo_spark.sh")
        .read_text(encoding="utf-8")
    )
    branch = shell.rsplit('if [[ "$mode" == "--dry-run" ]]', maxsplit=1)[1]
    dry_run_branch = branch.split("else", maxsplit=1)[0]
    assert "--gpus" not in dry_run_branch
    assert "--cpus 2 --memory 4g" in dry_run_branch
