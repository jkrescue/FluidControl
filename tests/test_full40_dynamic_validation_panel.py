from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "cfd/tandem_cylinders/make_full40_dynamic_validation_panel.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("dynamic_panel", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_obey_canonical_action_and_rate_bounds() -> None:
    module = load_generator()
    assert set(module.PROFILES) == {"minus", "zero", "plus"}
    for points in module.PROFILES.values():
        metrics = module.validate_points([list(row) for row in points])
        assert metrics["max_abs_omega"] <= 0.75
        assert metrics["max_abs_delta_omega_per_0p1"] <= 0.1
        assert points[0] == (0.0, 0.0)
        assert points[-1] == (20.0, 0.0)
    assert module.PROFILES["minus"] == tuple(
        (time, -omega) for time, omega in module.PROFILES["plus"]
    )


def test_predeclaration_is_validation_only_and_counterfactual_scope_is_strict() -> None:
    module = load_generator()
    full40, cases = module.load_cases()
    payload = module.predeclaration(full40, cases)
    assert payload["frozen_test_accessed"] is False
    assert payload["matrix"] == {
        "phase_bins": [1, 5],
        "profiles": ["minus", "zero", "plus"],
        "case_count": 6,
        "duration_D_over_U": 20.0,
    }
    assert all(row["split"] == "validation" for row in payload["cases"].values())
    contract = payload["fno_validation_contract"]
    assert contract["horizons_frames"] == [1, 10, 50, 100]
    assert contract["strict_counterfactual_pairs"] == "only start_frame=0 within each phase"
    assert contract["terminal_total_cd_pooled_nrmse_max"] == 0.10
    assert contract["strict_start0_delta_total_cd_mae_max"] == 0.023


def test_execution_and_qc_are_fail_closed() -> None:
    runner = (ROOT / "cfd/tandem_cylinders/run_full40_dynamic_validation_case.sh").read_text(
        encoding="utf-8"
    )
    audit = (ROOT / "scripts/audit_full40_dynamic_validation_panel.py").read_text(
        encoding="utf-8"
    )
    assert "--preflight-only" in runner
    assert "FULL40_DYNAMIC_VALIDATION_APPROVAL_TOKEN" in runner
    assert "MemAvailable is below 40 GiB" in runner
    assert "Curator/training activity detected" in runner
    assert "expected_repo=\"/workspace/fluid_control\"" in runner
    assert "Worker temporary copies require a separate reviewed transfer protocol" in runner
    assert "[p]impleFoam" in runner
    assert "refusing existing solver output" in runner
    assert "dst=/workspace/frozen" not in runner
    assert "frozen_test_accessed\") is not False" in runner
    assert "write_exclusive" in audit
    assert "Only frame 0 within each phase is a strict counterfactual" in audit
    assert "not closed-loop PPO" in audit
