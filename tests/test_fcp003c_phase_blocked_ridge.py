import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_fcp003c_phase_blocked_ridge.py"
spec = importlib.util.spec_from_file_location("ridge_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeCache(dict):
    pass


def make_cache():
    rng = np.random.default_rng(12)
    cache = FakeCache()
    beta = rng.normal(size=(129, 4))
    for panel_offset, panel in enumerate((module.PREFIX, module.LATE)):
        for phase_index, phase in enumerate(module.PHASES):
            zero_x = rng.normal(loc=phase_index * 3 + panel_offset, size=(100, 128))
            zero_y = zero_x @ beta[:128] + beta[128]
            cache[module.cache_key(panel, "zero", phase, "features")] = zero_x
            cache[module.cache_key(panel, "zero", phase, "targets_normalized")] = zero_y
            for profile in module.PROFILES:
                identity = f"{phase}:{profile}"
                action_x = rng.normal(loc=phase_index * 3 + panel_offset, size=(100, 128))
                action_y = action_x @ beta[:128] + beta[128]
                cache[module.cache_key(panel, "action", identity, "features")] = action_x
                cache[module.cache_key(panel, "action", identity, "targets_normalized")] = action_y
    return cache


def test_symmetric_zero_weighting_counts_each_phase_zero_twice():
    cache = make_cache()
    x, _ = module.symmetric_rows(cache, module.PREFIX, ("b00",))
    zero = cache[module.cache_key(module.PREFIX, "zero", "b00", "features")]
    assert x.shape == (400, 128)
    assert np.array_equal(x[100:200], zero)
    assert np.array_equal(x[300:400], zero)


def test_fold_standardization_excludes_held_phase():
    cache = make_cache()
    train_phases = ("b00", "b02", "b04")
    x, y = module.symmetric_rows(cache, module.PREFIX, train_phases)
    _, fit = module.fit_standardized_ridge(x, y, 1e-4)
    np.testing.assert_allclose(fit["feature_mean"], x.mean(axis=0), rtol=0, atol=0)
    global_x, _ = module.symmetric_rows(cache, module.PREFIX, module.PHASES)
    assert not np.allclose(fit["feature_mean"], global_x.mean(axis=0))


def test_alpha_zero_matches_direct_ols_predictions():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 128))
    beta = rng.normal(size=(129, 4))
    y = x @ beta[:128] + beta[128]
    fitted, _ = module.fit_standardized_ridge(x, y, 0.0)
    direct, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=1e-10)
    np.testing.assert_allclose(module.predict(x, fitted), module.predict(x, direct), rtol=0, atol=1e-11)


def test_constant_feature_is_zeroed_and_intercept_unpenalized():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 128))
    x[:, 7] = 9.0
    y = rng.normal(size=(300, 4)) + 5.0
    fitted, report = module.fit_standardized_ridge(x, y, 1.0)
    assert 7 in report["constant_feature_indices"]
    assert np.array_equal(fitted[7], np.zeros(4))
    np.testing.assert_allclose(module.predict(x, fitted).mean(0), y.mean(0), rtol=0, atol=1e-12)


def test_selection_uses_only_supplied_prefix_scores_and_ties_choose_larger():
    scores = {alpha: 3.0 for alpha in module.ALPHAS}
    assert module.select_alpha(scores) == 1.0
    scores[1e-4] = 1.0
    assert module.select_alpha(scores) == 1e-4
    # There is deliberately no late input to select_alpha.


def test_held_phase_metrics_do_not_mix_other_phases():
    cache = make_cache()
    x, y = module.symmetric_rows(cache, module.PREFIX, ("b00", "b02", "b04"))
    coefficients, _ = module.fit_standardized_ridge(x, y, 1e-4)
    report = module.evaluate_panel(cache, module.PREFIX, coefficients, ("b06",))
    assert report["endpoint_counts"] == {"action": 200, "unique_zero": 100, "delta": 200}
    assert set(report["per_pair"]) == {"b06:multisine", "b06:prbs"}
