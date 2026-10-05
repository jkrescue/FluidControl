from pathlib import Path

import pytest

from fluid_control.dual_fno import require_dual_training_config


def test_separate_config_required_only_for_dual():
    require_dual_training_config(False, None)
    require_dual_training_config(True, Path("training.yaml"))
    with pytest.raises(ValueError):
        require_dual_training_config(True, None)
    with pytest.raises(ValueError):
        require_dual_training_config(False, Path("training.yaml"))


def test_both_evaluators_preserve_evaluation_config_and_bind_training_file():
    root = Path(__file__).resolve().parents[1]
    for name in ("evaluate_tandem_fno.py", "diagnose_fno_force_window.py"):
        source = (root / "scripts" / name).read_text()
        assert "cfg = load_composed_config(args.config)" in source
        assert "config_path=args.dual_training_config" in source
        assert "require_dual_training_config(use_dual_fno, args.dual_training_config)" in source
