from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_fc_p002_phase_action_horizon import aggregate


def test_grouped_macro_and_pooled_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    rows = []
    for predicted, target in ((2.0, 1.0), (4.0, 2.0)):
        rows.append({
            "case": "full40_dynamic_validation_b01_plus", "horizon": 100,
            "state_mae_physical_units": 0.2, "total_drag_absolute_error": abs(predicted - target),
            "rear_cl_mae": 0.3, "predicted_total_drag": predicted, "target_total_drag": target,
            "mean_abs_omega": 0.5, "mean_abs_domega_dt": 0.4,
        })
    path.write_text(json.dumps({"segments": rows}))
    result = aggregate(path)[0]
    assert result["segments"] == 2
    assert result["aggregation"]["total_cd_mae_macro_over_segments"] == pytest.approx(1.5)
    assert result["aggregation"]["total_cd_nrmse_pooled_over_endpoints"] == pytest.approx(1.0)
