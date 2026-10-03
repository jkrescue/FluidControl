from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/audit_dev30_validation_diagnostic.py"
    spec = importlib.util.spec_from_file_location("dev30_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_module()


def expected() -> dict[str, tuple[str, float]]:
    result = {}
    for phase in ("01", "05"):
        for label, action in AUDIT.ACTIONS.items():
            result[f"matched_start_acquisition_validation_b{phase}_{label}"] = (
                phase,
                action,
            )
    return result


def report() -> dict:
    cases = []
    for index, name in enumerate(sorted(expected())):
        horizons = {}
        for horizon in AUDIT.HORIZONS:
            horizons[str(horizon)] = {
                "segments": 2,
                "stable": True,
                "failed_segments": 0,
                "total_drag_rmse": 0.1 + 0.01 * index,
                "total_drag_target_rms": 2.0,
                "total_drag_mae": 0.08,
                "persistence_total_drag_mae": 0.12,
                "front_cl_mae": 0.03,
                "rear_cl_mae": 0.04,
            }
        cases.append({"case": name, "horizons": horizons})
    return {
        "split": "validation",
        "action_mode": "observed",
        "evaluation_data": "/workspace/devdata",
        "normalization_data": "/workspace/devdata",
        "checkpoint_dir": "/workspace/checkpoint",
        "force_channels": AUDIT.FORCE_CHANNELS,
        "action_scale": 0.75,
        "evaluation_action_limit": 0.75,
        "cases": cases,
    }


def segments() -> dict:
    rows = []
    for name, (_, action) in expected().items():
        rows.append(
            {
                "case": name,
                "horizon": 100,
                "start": 0,
                "predicted_total_drag": 2.0 + 0.2 * action,
                "target_total_drag": 2.0 + 0.1 * action,
            }
        )
        rows.append(
            {
                "case": name,
                "horizon": 100,
                "start": 25,
                "predicted_total_drag": 9.0 - action,
                "target_total_drag": 1.0 + action,
            }
        )
    return {"split": "validation", "action_mode": "observed", "segments": rows}


def test_pooled_metrics_use_segment_weighted_sums() -> None:
    metrics = AUDIT.pooled_metrics(report(), expected())
    assert set(metrics) == {"1", "10", "50", "100"}
    assert metrics["100"]["segments"] == 20
    assert metrics["100"]["beats_persistence"] is True
    expected_nrmse = (sum((0.1 + 0.01 * i) ** 2 for i in range(10)) / 10) ** 0.5 / 2.0
    assert metrics["100"]["pooled_total_cd_nrmse"] == pytest.approx(expected_nrmse)


def test_ranking_uses_only_strict_start0() -> None:
    result = AUDIT.strict_start0_ranking(segments(), expected())
    assert result["strict_start0_cases"] == 10
    assert result["zero_relative_pairs"] == 8
    assert result["cross_action_pairs"] == 20
    assert result["zero_relative_sign_accuracy"] == 1.0
    assert result["cross_action_ordering_accuracy"] == 1.0
    assert "action-diverged" in result["start_gt_zero_policy"]


def test_report_contract_rejects_wrong_split_or_case_count() -> None:
    bad = report()
    bad["split"] = "test"
    with pytest.raises(ValueError, match="runtime contract"):
        AUDIT.validate_report_contract(bad, expected())
    bad = report()
    bad["cases"].pop()
    with pytest.raises(ValueError, match="validation10"):
        AUDIT.validate_report_contract(bad, expected())


def test_main_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "result.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--report",
            str(tmp_path / "missing-report.json"),
            "--segments",
            str(tmp_path / "missing-segments.json"),
            "--data",
            str(tmp_path / "data"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        AUDIT.main()


def test_runner_mount_policy_and_nonformal_label() -> None:
    text = (ROOT / "scripts/run_full40_dev30_validation_diagnostic_spark.sh").read_text(
        encoding="utf-8"
    )
    assert 'src=$root,dst=/workspace' not in text
    for mount in ("scripts", "src", "conf"):
        assert f"src=$root/{mount},dst=/workspace/{mount},readonly" in text
    assert "src=$data_host,dst=/workspace/devdata,readonly" in text
    assert "src=$checkpoint_host,dst=/workspace/checkpoint,readonly" in text
    assert "frozen_test" not in text
    assert "--horizons 1 10 50 100" in text
    assert "--min-free-gib 20" in text
    assert "refusing existing diagnostic output" in text
    assert "realpath -e" in text
    assert "realpath -m" in text
    assert "resolved output escapes" in text
    assert "DIAGNOSTIC_ONLY" in text
