import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_fcp008_force_readout_candidate.py"
spec = importlib.util.spec_from_file_location("fcp008", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def synthetic_rows(seed=3):
    rng = np.random.default_rng(seed)
    families = np.concatenate(
        [np.repeat(family, count) for family, count in module.FAMILY_ENDPOINTS.items()]
    )
    phases = np.tile(np.array(module.PHASES), len(families) // 4)
    features = rng.normal(size=(len(families), 128))
    beta = rng.normal(size=(129, 4))
    targets = features @ beta[:128] + beta[128]
    return features, targets, phases, families


def test_global_family_weights_are_fixed_exposure_shares():
    _, _, _, families = synthetic_rows()
    weights = module.endpoint_weights(families)
    for family, exposure in module.FAMILY_EXPOSURES.items():
        np.testing.assert_allclose(
            weights[families == family].sum(), exposure / sum(module.FAMILY_EXPOSURES.values()), rtol=0, atol=2e-15
        )
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=2e-15)


def test_weighted_ridge_fold_statistics_use_only_fold_rows():
    features, targets, phases, families = synthetic_rows()
    weights = module.endpoint_weights(families)
    train = phases != "b06"
    coefficients, report = module.fit_weighted_ridge(features[train], targets[train], weights[train], 1e-6)
    local = weights[train] / weights[train].sum()
    expected_mean = local @ features[train]
    assert coefficients.shape == (129, 4)
    assert report["normalized_input_weight_sum"] == 1.0
    # Exact recovery on held phase proves the synthetic mapping is shared, without using held statistics.
    np.testing.assert_allclose(
        module.predict(features[~train], coefficients),
        targets[~train],
        rtol=0,
        atol=1e-4,
    )
    assert not np.allclose(expected_mean, weights @ features)


def test_phase_cv_uses_global_endpoint_weights_and_no_fold_rebalancing():
    features, targets, phases, families = synthetic_rows()
    weights = module.endpoint_weights(families)
    selected, report = module.phase_blocked_cv(features, targets, phases, families, weights)
    assert selected in module.ALPHAS
    assert set(report["fold_fits"]) == set(module.PHASES)
    for phase in module.PHASES:
        assert set(report["selected_oof_by_phase_family"][phase]) == set(module.FAMILY_EXPOSURES)
    assert "late" not in repr(report).lower()


def test_oof_pool_keeps_global_weights_when_a_fold_lacks_train16():
    rng = np.random.default_rng(31)
    family_counts = {"base20": 80, "train8": 40, "train16": 24}
    families = np.concatenate(
        [np.repeat(family, count) for family, count in family_counts.items()]
    )
    phases = np.concatenate(
        [
            np.tile(np.array(module.PHASES), 20),
            np.tile(np.array(module.PHASES), 10),
            np.tile(np.array(("b00", "b02")), 12),
        ]
    )
    features = rng.normal(size=(len(families), 128))
    targets = rng.normal(size=(len(families), 4))
    weights = np.empty(len(families), dtype=np.float64)
    for family, exposure in module.FAMILY_EXPOSURES.items():
        mask = families == family
        weights[mask] = exposure / sum(module.FAMILY_EXPOSURES.values()) / mask.sum()
    selected, report = module.phase_blocked_cv(
        features, targets, phases, families, weights
    )
    assert report["selected_oof_by_phase_family"]["b04"]["train16"] is None
    oof = np.empty_like(targets)
    for phase in module.PHASES:
        train = phases != phase
        coefficients, _ = module.fit_weighted_ridge(
            features[train], targets[train], weights[train], selected
        )
        oof[~train] = module.predict(features[~train], coefficients)
    expected = np.mean(weights @ np.square(oof - targets))
    assert report["aggregate_global_endpoint_weighted_equal_channel_mse"][str(selected)] == expected


def test_coefficients_are_returned_in_original_feature_coordinates():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(400, 128)) * np.linspace(0.2, 9.0, 128) + 7.0
    beta = rng.normal(size=(129, 4))
    y = x @ beta[:128] + beta[128]
    weights = rng.random(400)
    coefficients, _ = module.fit_weighted_ridge(x, y, weights, 0.0)
    np.testing.assert_allclose(module.predict(x, coefficients), y, rtol=0, atol=1e-9)


def test_inventory_mode_does_not_require_execution_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--base", str(tmp_path),
            "--train8", str(tmp_path),
            "--train16", str(tmp_path),
            "--normalization", str(tmp_path / "norm.json"),
            "--config", str(tmp_path / "config.yaml"),
            "--checkpoint-dir", str(tmp_path),
            "--source-phase-mapping", str(tmp_path / "mapping.json"),
            "--output", str(tmp_path / "inventory.json"),
            "--inventory-only",
        ],
    )
    assert module.parse_args().approval is None


def test_source_phase_comes_from_time_and_state_not_misleading_name(monkeypatch, tmp_path):
    states = {
        phase: np.full((3, 2, 2), index, dtype=np.float32)
        for index, phase in enumerate(module.PHASES)
    }
    times = {"b00": 148.0, "b02": 106.0, "b04": 120.0, "b06": 134.0}
    records = []
    lookup = {}
    for phase in module.PHASES:
        path = tmp_path / f"matched_start_acquisition_train_{phase}_zero.h5"
        records.append(module.Trajectory("base20", path.stem, path, "0" * 64, 801))
        lookup[path] = (times[phase], states[phase])
    misleading = tmp_path / "direct_cfd_episode_b00_but_physical_b02.h5"
    records.append(module.Trajectory("train16", misleading.stem, misleading, "1" * 64, 129))
    lookup[misleading] = (times["b02"], states["b02"])
    monkeypatch.setattr(module, "_initial_state", lambda path, _frames: lookup[path])
    assigned = module.assign_source_phases(records)
    assert assigned[-1].name.endswith("b00_but_physical_b02")
    assert assigned[-1].source_phase == "b02"


def test_exact_score_tie_prefers_larger_alpha():
    assert module.choose_alpha({alpha: 1.0 for alpha in module.ALPHAS}) == 1.0


def test_force_row_only_verifier_rejects_any_state_or_other_tensor_change():
    rng = np.random.default_rng(8)
    before = {
        "decoder_net.final_layer.linear.weight": rng.normal(size=(7, 128)).astype(np.float32),
        "decoder_net.final_layer.linear.bias": rng.normal(size=(7,)).astype(np.float32),
        "spec_encoder.layer.weight": rng.normal(size=(3, 3)).astype(np.float32),
    }
    coefficients = rng.normal(size=(129, 4)).astype(np.float32)
    after = {key: value.copy() for key, value in before.items()}
    after["decoder_net.final_layer.linear.weight"][3:7] = coefficients[:128].T
    after["decoder_net.final_layer.linear.bias"][3:7] = coefficients[128]
    changed = module.verify_force_row_only(
        before,
        after,
        "decoder_net.final_layer.linear.weight",
        "decoder_net.final_layer.linear.bias",
        coefficients,
    )
    assert changed == [
        "decoder_net.final_layer.linear.weight",
        "decoder_net.final_layer.linear.bias",
    ]
    after["spec_encoder.layer.weight"][0, 0] += 1
    try:
        module.verify_force_row_only(
            before,
            after,
            "decoder_net.final_layer.linear.weight",
            "decoder_net.final_layer.linear.bias",
            coefficients,
        )
    except ValueError as error:
        assert "non-force tensor" in str(error)
    else:
        raise AssertionError("non-force mutation must fail")
