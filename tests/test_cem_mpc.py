"""CPU unit tests for the model-agnostic CEM-MPC optimizer."""

from __future__ import annotations

import numpy as np

from fluid_control.cem_mpc import CEMConfig, CEMMPCOptimizer


def test_all_objective_candidates_respect_hard_constraints() -> None:
    config = CEMConfig(
        horizon=7,
        population_size=128,
        iterations=4,
        omega_abs_max=0.8,
        max_delta_omega=0.12,
        initial_std=4.0,
        seed=7,
    )
    optimizer = CEMMPCOptimizer(config)
    observed: list[np.ndarray] = []

    def objective(actions: np.ndarray) -> np.ndarray:
        observed.append(actions.copy())
        return np.square(actions - 0.4).sum(axis=1)

    result = optimizer.optimize(objective, current_omega=-0.3)
    candidates = np.concatenate(observed, axis=0)
    preceding = np.concatenate(
        (np.full((candidates.shape[0], 1), -0.3), candidates[:, :-1]), axis=1
    )

    assert np.max(np.abs(candidates)) <= config.omega_abs_max
    assert np.max(np.abs(candidates - preceding)) <= config.max_delta_omega + 1e-12
    assert np.max(np.abs(result.actions)) <= config.omega_abs_max


def test_cem_converges_on_feasible_quadratic_objective() -> None:
    target = np.array([0.2, 0.4, 0.6, 0.6, 0.35])
    optimizer = CEMMPCOptimizer(
        CEMConfig(
            horizon=target.size,
            population_size=512,
            elite_fraction=0.08,
            iterations=10,
            omega_abs_max=0.75,
            max_delta_omega=0.25,
            initial_std=0.6,
            min_std=1e-5,
            seed=19,
        )
    )

    def objective(actions: np.ndarray) -> np.ndarray:
        return np.square(actions - target).sum(axis=1)

    result = optimizer.optimize(objective, current_omega=0.0)

    assert result.cost < 1e-4
    np.testing.assert_allclose(result.actions, target, atol=1e-2)
    assert all(
        later <= earlier
        for earlier, later in zip(
            result.best_cost_history, result.best_cost_history[1:]
        )
    )


def test_seed_makes_repeated_searches_reproducible() -> None:
    optimizer = CEMMPCOptimizer(
        CEMConfig(horizon=4, population_size=64, iterations=3, seed=1234)
    )

    def objective(actions: np.ndarray) -> np.ndarray:
        return np.square(actions + 0.25).sum(axis=1)

    first = optimizer.optimize(objective, current_omega=0.1)
    second = optimizer.optimize(objective, current_omega=0.1)

    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.std, second.std)
    assert first.cost == second.cost
    assert first.best_cost_history == second.best_cost_history
