from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plan_full40_fno_retrain.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "plan_full40_fno_retrain.py"
spec = importlib.util.spec_from_file_location("full40_train_plan", SCRIPT)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_normalization_requires_train20_and_four_force_channels() -> None:
    valid = {
        "computed_from": "train split only",
        "state_channels": ["u", "v", "gauge_pressure"],
        "all_force_channels": MODULE.FORCE_CHANNELS,
        "state_mean": [0.0] * 3,
        "state_std": [1.0] * 3,
        "state_abs_normalized_channel_max_train": [2.0, 3.0, 4.0],
        "state_abs_normalized_max_train": 4.0,
        "force_channels": ["rear_cd", "rear_cl"],
        "force_mean": [0.0, 0.0],
        "force_std": [1.0, 1.0],
        "all_force_mean": [0.0] * 4,
        "all_force_std": [1.0] * 4,
    }
    MODULE.validate_normalization(valid)
    with pytest.raises(ValueError, match="train-only"):
        MODULE.validate_normalization({**valid, "computed_from": "all splits"})
    with pytest.raises(ValueError, match="channel"):
        MODULE.validate_normalization({**valid, "all_force_channels": []})


def test_configs_keep_official_shape_and_all_h20_checkpoints() -> None:
    for relative in MODULE.CONFIGS:
        MODULE.validate_config(ROOT, relative)
    h20 = (ROOT / "conf/tandem_fno_full40_h20.yaml").read_text()
    assert "checkpoint_interval: 1" in h20
    assert "checkpoint_keep_last: 10" in h20
    onestep = (ROOT / "conf/tandem_fno_full40_onestep.yaml").read_text()
    assert "num_fno_modes: [32, 32]" in onestep
    assert "[48, 48]" not in onestep
    assert "gpu_memory_fraction: 0.20" in onestep


def test_plan_fails_closed_before_final_dataset_exists(tmp_path: Path) -> None:
    result = MODULE.build_plan(tmp_path)
    assert result == {
        "status": "FULL40_FNO_RETRAIN_BLOCKED",
        "blockers": ["PREDECLARATION_MISMATCH"],
    }


def test_runner_is_guarded_and_does_not_reference_frozen_data() -> None:
    path = ROOT / "scripts/run_full40_fno_retrain_spark.sh"
    text = path.read_text()
    assert "--min-free-gib 20" in text
    assert 'allocator="0.20"' in text
    assert 'allocator="0.15"' in text
    assert "--network none" in text
    assert "EXECUTE_REVIEWED_FULL40_FNO_RETRAIN" in text
    assert "frozen_test" not in text
    assert '[[ ! -e "$output" ]]' in text
    assert "retry with a new independent run-id" in text
    assert "training_history.json" not in text
    assert "requested_run_id=\"${3:-}\"" in text
    assert "^[a-z0-9][a-z0-9_-]{0,63}$" in text
    assert "FULL40_ONESTEP_RUN_ID" in text
    assert 'training.initial_checkpoint=$onestep_parent/best' in text
