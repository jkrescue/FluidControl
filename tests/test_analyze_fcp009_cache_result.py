import numpy as np
import pytest

from analyze_fcp009_cache_result import physical_metrics


def test_physical_metrics_reports_rear_and_total_cd():
    error = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    result = physical_metrics(error, np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([True, True]))
    assert result["rear_cd"]["mae"] == pytest.approx(0.9)
    assert result["rear_cd"]["rmse"] == pytest.approx(0.9)
    assert result["rear_cl"]["mae"] == pytest.approx(1.6)
    assert result["total_cd"]["mae"] == pytest.approx(1.0)
    assert all(result[name]["count"] == 2 and result[name]["bias"] == pytest.approx(0.0) for name in result)
