from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_full40_validation_gate.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "audit_full40_validation_gate.py"
spec = importlib.util.spec_from_file_location("full40_validation_gate", SCRIPT)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def fixtures():
    expected = {}
    cases = []
    segments = []
    for phase in ("01", "05"):
        for action_name, action in MODULE.ACTIONS.items():
            name = f"matched_start_acquisition_validation_b{phase}_{action_name}"
            expected[name] = (phase, action)
            cases.append(
                {
                    "case": name,
                    "horizons": {
                        "100": {
                            "segments": 2,
                            "stable": True,
                            "failed_segments": 0,
                            "total_drag_rmse": 0.1,
                            "total_drag_mae": 0.08,
                            "persistence_total_drag_mae": 0.2,
                            "total_drag_target_rms": 2.0,
                            "front_cl_mae": 0.02,
                            "rear_cl_mae": 0.03,
                        }
                    },
                }
            )
            target = 2.0 + 0.1 * action
            segments.append(
                {
                    "case": name,
                    "horizon": 100,
                    "start": 0,
                    "target_total_drag": target,
                    "predicted_total_drag": target + 0.005 * action,
                }
            )
            segments.append(
                {
                    "case": name,
                    "horizon": 100,
                    "start": 25,
                    "target_total_drag": 99.0,
                    "predicted_total_drag": -99.0,
                }
            )
    report = {"split": "validation", "action_mode": "observed", "cases": cases}
    segment_doc = {
        "split": "validation",
        "action_mode": "observed",
        "segments": segments,
    }
    return expected, report, segment_doc


def test_pooled_h100_uses_segment_weighted_sum_of_squares() -> None:
    expected, report, _ = fixtures()
    result = MODULE.pooled_h100(report, expected)
    assert result["pooled_total_cd_nrmse"] == pytest.approx(0.05)
    assert result["passes_fixed_10pct_gate"] is True
    assert result["front_cl_mae"] == pytest.approx(0.02)
    assert result["rear_cl_mae"] == pytest.approx(0.03)
    assert result["beats_persistence"] is True


def test_checkpoint_is_bound_to_exact_single_physicsnemo_generation(tmp_path) -> None:
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    model = checkpoint / "FNO.10.0.mdlus"
    model.write_bytes(b"fixed-full40-checkpoint")
    report = {"checkpoint_dir": str(checkpoint), "checkpoint_epoch": 10}
    result = MODULE.validate_checkpoint(report, checkpoint)
    assert result["checkpoint_sha256"] == MODULE.sha256(model)
    (checkpoint / "FNO.11.0.mdlus").write_bytes(b"second")
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.validate_checkpoint(report, checkpoint)


def test_action_gate_uses_only_eight_strict_start0_zero_relative_pairs() -> None:
    expected, _, segments = fixtures()
    result = MODULE.strict_start0_differences(segments, expected)
    assert result["strict_pairs"] == 8
    assert result["zero_relative_sign_accuracy"] == 1.0
    assert result["passes_delta_cd_mae_gate"] is True
    assert result["passes_sign_ranking_gate"] is True
    assert result["cross_action_ordering_pairs"] == 20
    assert result["cross_action_ordering_accuracy"] == 1.0
    assert result["passes_cross_action_ordering_gate"] is True
    assert "action-diverged" in result["start_gt_zero_policy"]


def test_validation_runner_never_names_frozen_split() -> None:
    text = (ROOT / "scripts/run_full40_validation_only_spark.sh").read_text()
    assert "--split validation" in text
    assert "--horizons 1 10 50 100" in text
    assert "frozen_test" not in text
