from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_is_sequential_frozen_blind_and_fail_closed() -> None:
    text = (ROOT / "scripts/run_full40_dev30_quickscreen_pipeline_spark.sh").read_text(
        encoding="utf-8"
    )
    one = text.index("onestep --execute")
    h20 = text.index("h20 --execute")
    validation = text.index("run_full40_dev30_validation_diagnostic_spark.sh --execute")
    assert one < h20 < validation
    assert "set -euo pipefail" in text
    assert "trap on_exit EXIT" in text
    assert "downstream stages were not started" in text
    assert "flock -n" in text
    assert "refusing existing pipeline output" in text
    assert "FULL40_DEV30_FNO_RETRAIN_READY" in text
    assert "dev30_manifest_sha256" in text
    assert "train_only_normalization_sha256" in text
    assert "FULL40_DEV30_QUICKSCREEN_PARENT_SHA256=\"$one_sha\"" in text
    assert "checkpoint_lineage.json" in text
    assert "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE" in text
    assert '"frozen_test_mounted_or_accessed": False' in text
    assert '"formal_gate_authorized": False' in text
    assert '"ppo_authorized": False' in text
    assert "dst=/workspace/frozen" not in text


def test_pipeline_uses_independent_outputs_and_fixed_stage_lengths() -> None:
    text = (ROOT / "scripts/run_full40_dev30_quickscreen_pipeline_spark.sh").read_text(
        encoding="utf-8"
    )
    assert "quickscreen_onestep_${run_id}" in text
    assert "quickscreen_h20_${run_id}" in text
    assert "validation_quickscreen_${run_id}" in text
    assert "quickscreen_pipeline_${run_id}" in text
    assert '"stages": ["onestep_10epoch", "h20_5epoch", "validation_diagnostic"]' in text
