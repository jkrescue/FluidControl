import numpy as np
import pytest
import json
import hashlib

from analyze_fcp009_cache_result import (
    physical_metrics,
    validate_cache_schema,
    validate_normalization_binding,
)


def test_physical_metrics_reports_rear_and_total_cd():
    error = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    result = physical_metrics(error, np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([True, True]))
    assert result["rear_cd"]["mae"] == pytest.approx(0.9)
    assert result["rear_cd"]["rmse"] == pytest.approx(0.9)
    assert result["rear_cl"]["mae"] == pytest.approx(1.6)
    assert result["total_cd"]["mae"] == pytest.approx(1.0)
    assert all(result[name]["count"] == 2 and result[name]["bias"] == pytest.approx(0.0) for name in result)


def test_normalization_binding_rejects_tamper_and_nonpositive(tmp_path):
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps({"all_force_std": [1.0, 2.0, 3.0, 4.0]}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"input_sha256": {"normalization": digest}}
    np.testing.assert_array_equal(
        validate_normalization_binding(result, path, expected_sha=digest),
        [1.0, 2.0, 3.0, 4.0],
    )
    result["input_sha256"]["normalization"] = "0" * 64
    with pytest.raises(ValueError, match="normalization contract"):
        validate_normalization_binding(result, path, expected_sha=digest)
    path.write_text(json.dumps({"all_force_std": [1.0, 2.0, 0.0, 4.0]}))
    bad_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="normalization contract"):
        validate_normalization_binding(
            {"input_sha256": {"normalization": bad_digest}},
            path,
            expected_sha=bad_digest,
        )


def test_cache_schema_rejects_truncated_and_nonfinite():
    truncated = {
        "features": np.zeros((2, 128)),
        "matched_weight_h1_features": np.zeros((2, 128)),
        "targets_normalized": np.zeros((2, 4)),
        "phases": np.array(["b00", "b02"]),
        "relative_steps": np.array([1, 2]),
    }
    with pytest.raises(ValueError, match="cache schema"):
        validate_cache_schema(truncated)
    truncated["features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="cache schema"):
        validate_cache_schema(truncated)
