import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/produce_canonical_surrogate_compatibility_gates.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_compat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def force_rows(rear_amplitude=1.0, cd=2.0):
    import math

    return [
        [cd / 2, 0.0, cd / 2, rear_amplitude * math.sin(index * 0.2)]
        for index in range(101)
    ]


def force_document(prediction_offset=0.0):
    cases = []
    for name in MODULE.CASES:
        truth = force_rows()
        predicted = [row[:] for row in truth]
        for row in predicted:
            row[2] += prediction_offset
        cases.append(
            {
                "case": name,
                "horizon_steps": 100,
                "times": [index / 10 for index in range(101)],
                "true_forces": truth,
                "predicted_forces": predicted,
            }
        )
    return {
        "status": "SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE",
        "model_sha256": "a" * 64,
        "manifest_sha256": MODULE.DYNAMIC_MANIFEST_SHA256,
        "normalization_sha256": MODULE.NORMALIZATION_SHA256,
        "force_channels": MODULE.FORCE_CHANNELS,
        "frozen_test_accessed": False,
        "ppo_authorized": False,
        "cases": cases,
    }


def dynamic_documents(error=0.0):
    cases = []
    segments = []
    for case_index, name in enumerate(MODULE.CASES):
        horizons = {
            str(horizon): {
                "stable": True,
                "failed_segments": 0,
                "segments": 101,
                "total_drag_rmse": error,
                "total_drag_target_rms": 2.0,
            }
            for horizon in (1, 10, 50, 100)
        }
        cases.append({"case": name, "horizons": horizons})
        target = 2.0 + 0.01 * case_index
        for start in range(101):
            segments.append(
                {
                    "case": name,
                    "horizon": 100,
                    "start": start,
                    "target_total_drag": target,
                    "predicted_total_drag": target + error,
                }
            )
    return (
        {
            "split": "validation",
            "action_mode": "observed",
            "force_channels": MODULE.FORCE_CHANNELS,
            "cases": cases,
        },
        {"segments": segments},
    )


def test_sampled_window_is_exactly_last_62_endpoints():
    result = MODULE.sampled_window(
        [index / 10 for index in range(101)], force_rows()
    )
    assert result["sample_count"] == 62
    assert result["first_sample_time"] == pytest.approx(3.9)
    assert result["last_sample_time"] == pytest.approx(10.0)
    assert result["sample_span_D_over_U"] == pytest.approx(6.1)


def test_window_recomputes_branch_metrics_and_fails_large_error():
    passed = MODULE.recompute_window(force_document(0.0), "a" * 64)
    assert all(passed["metric_all_branches_pass"].values())
    failed = MODULE.recompute_window(force_document(0.03), "a" * 64)
    assert failed["metric_all_branches_pass"]["total_drag"] is False
    assert all(not row["metric_pass"]["total_drag"] for row in failed["branches"])


def test_window_rejects_duplicate_or_wrong_checkpoint():
    document = force_document()
    document["cases"].append(document["cases"][0])
    with pytest.raises(ValueError, match="exactly dynamic6"):
        MODULE.recompute_window(document, "a" * 64)
    with pytest.raises(ValueError, match="identity differs"):
        MODULE.recompute_window(force_document(), "b" * 64)


def test_dynamic_uses_all_rolling_h100_and_start0_pairs():
    evaluation, segments = dynamic_documents(0.01)
    result = MODULE.recompute_dynamic(evaluation, segments)
    assert result["pooled_all_rolling_h100_segment_count"] == 606
    assert result["pooled_all_rolling_h100_total_cd_nrmse"] == pytest.approx(0.005)
    # The common additive prediction error cancels in action-minus-zero deltas.
    assert result["strict_start0_action_minus_zero_total_cd_mae"] == pytest.approx(0.0)
    assert all(result["metric_pass"].values())


def test_dynamic_threshold_failures_are_independent():
    evaluation, segments = dynamic_documents(0.25)
    result = MODULE.recompute_dynamic(evaluation, segments)
    assert result["metric_pass"]["pooled_all_rolling_h100_total_cd_nrmse"] is False
    # Still cancels at start0; this demonstrates the two required computations.
    assert result["metric_pass"]["strict_start0_action_minus_zero_total_cd_mae"] is True
    segments["segments"][0]["predicted_total_drag"] += 0.2
    result = MODULE.recompute_dynamic(evaluation, segments)
    assert result["strict_start0_action_minus_zero_total_cd_mae"] > 0.023


def test_dynamic_rejects_missing_case_and_nonfinite():
    evaluation, segments = dynamic_documents()
    evaluation["cases"].pop()
    with pytest.raises(ValueError, match="case matrix"):
        MODULE.recompute_dynamic(evaluation, segments)
    evaluation, segments = dynamic_documents()
    evaluation["cases"][0]["horizons"]["100"]["total_drag_rmse"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.recompute_dynamic(evaluation, segments)


def test_step_receipt_binds_checkpoint_and_required_files(tmp_path):
    import hashlib
    import json

    evaluation = tmp_path / "evaluation.json"
    segments = tmp_path / "segments.json"
    evaluation.write_text("{}")
    segments.write_text("{}")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    payload = {
        "status": "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE",
        "step": "dynamic6",
        "checkpoint_sha256": "a" * 64,
        "sha256": {
            "/worker/evaluation.json": digest(evaluation),
            "/worker/segments.json": digest(segments),
        },
    }
    receipt.write_text(json.dumps(payload))
    MODULE.verify_step_receipt(
        receipt,
        step="dynamic6",
        checkpoint_sha256="a" * 64,
        required_files=(evaluation, segments),
    )
    payload["sha256"]["/worker/segments.json"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not bind segments.json"):
        MODULE.verify_step_receipt(
            receipt,
            step="dynamic6",
            checkpoint_sha256="a" * 64,
            required_files=(evaluation, segments),
        )


def test_dynamic_rejects_truncated_or_duplicate_start0_segments():
    evaluation, segments = dynamic_documents()
    segments["segments"].pop()
    with pytest.raises(ValueError, match="exactly 606"):
        MODULE.recompute_dynamic(evaluation, segments)
    _, segments = dynamic_documents()
    duplicate = dict(segments["segments"][0])
    segments["segments"][1] = duplicate
    with pytest.raises(ValueError, match="start indices must equal 0..100"):
        MODULE.recompute_dynamic(evaluation, segments)


def test_dynamic_rejects_duplicate_nonzero_start_with_count_unchanged():
    evaluation, segments = dynamic_documents()
    assert segments["segments"][1]["start"] == 1
    assert segments["segments"][2]["start"] == 2
    segments["segments"][2]["start"] = 1
    with pytest.raises(ValueError, match="start indices must equal 0..100"):
        MODULE.recompute_dynamic(evaluation, segments)
