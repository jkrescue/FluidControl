"""CPU-only cross-entropy optimizer for scalar MPC action sequences.

The optimizer is deliberately independent of a dynamics model and reward design.
Callers provide a vectorized objective that maps a population of feasible action
sequences with shape ``(population, horizon)`` to one cost per sequence. Lower
costs are better.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True, slots=True)
class CEMConfig:
    """Configuration for cross-entropy action-sequence optimization."""

    horizon: int
    population_size: int = 512
    elite_fraction: float = 0.1
    iterations: int = 8
    omega_abs_max: float = 1.0
    max_delta_omega: float | None = None
    initial_std: float = 0.5
    min_std: float = 1e-3
    update_rate: float = 0.7
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject configurations that cannot define a valid CEM search."""
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not np.isfinite(self.omega_abs_max) or self.omega_abs_max <= 0.0:
            raise ValueError("omega_abs_max must be finite and positive")
        if self.max_delta_omega is not None and (
            not np.isfinite(self.max_delta_omega) or self.max_delta_omega <= 0.0
        ):
            raise ValueError("max_delta_omega must be finite and positive")
        if not np.isfinite(self.initial_std) or self.initial_std <= 0.0:
            raise ValueError("initial_std must be finite and positive")
        if not np.isfinite(self.min_std) or self.min_std <= 0.0:
            raise ValueError("min_std must be finite and positive")
        if self.min_std > self.initial_std:
            raise ValueError("min_std cannot exceed initial_std")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CEMResult:
    """Best feasible sequence and diagnostics from a CEM search."""

    actions: FloatArray
    cost: float
    mean: FloatArray
    std: FloatArray
    best_cost_history: tuple[float, ...]
    evaluations: int


class CEMMPCOptimizer:
    """Optimize scalar action sequences using the cross-entropy method."""

    def __init__(self, config: CEMConfig) -> None:
        self.config = config

    def optimize(
        self,
        objective: Objective,
        *,
        current_omega: float,
        initial_mean: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> CEMResult:
        """Minimize ``objective`` over hard-constrained omega sequences.

        A fresh random generator is created from ``config.seed`` for each call.
        Consequently, repeated calls with identical inputs and a deterministic
        objective return identical results.
        """
        current = self._validate_current_omega(current_omega)
        mean = self._initial_mean(current, initial_mean)
        std = np.full(self.config.horizon, self.config.initial_std, dtype=np.float64)
        rng = np.random.default_rng(self.config.seed)
        elite_count = max(
            2, int(np.ceil(self.config.elite_fraction * self.config.population_size))
        )

        best_actions: FloatArray | None = None
        best_cost = np.inf
        history: list[float] = []
        evaluations = 0

        for _ in range(self.config.iterations):
            population = rng.normal(
                loc=mean,
                scale=std,
                size=(self.config.population_size, self.config.horizon),
            )
            population[0] = mean
            population = self.project(population, current_omega=current)
            costs = self._evaluate(objective, population)
            evaluations += population.shape[0]

            iteration_best = int(np.argmin(costs))
            if costs[iteration_best] < best_cost:
                best_cost = float(costs[iteration_best])
                best_actions = population[iteration_best].copy()
            history.append(best_cost)

            elite_indices = np.argpartition(costs, elite_count - 1)[:elite_count]
            elites = population[elite_indices]
            elite_mean = elites.mean(axis=0)
            elite_std = elites.std(axis=0)
            rate = self.config.update_rate
            mean = self.project(
                ((1.0 - rate) * mean + rate * elite_mean)[None, :],
                current_omega=current,
            )[0]
            std = np.maximum(
                (1.0 - rate) * std + rate * elite_std,
                self.config.min_std,
            )

        mean_cost = float(self._evaluate(objective, mean[None, :])[0])
        evaluations += 1
        if best_actions is None or mean_cost < best_cost:
            best_actions = mean.copy()
            best_cost = mean_cost

        return CEMResult(
            actions=best_actions,
            cost=best_cost,
            mean=mean.copy(),
            std=std.copy(),
            best_cost_history=tuple(history),
            evaluations=evaluations,
        )

    def project(self, actions: ArrayLike, *, current_omega: float) -> FloatArray:
        """Project one or more sequences onto magnitude and slew-rate limits."""
        current = self._validate_current_omega(current_omega)
        projected = np.asarray(actions, dtype=np.float64).copy()
        squeeze = projected.ndim == 1
        if squeeze:
            projected = projected[None, :]
        if projected.ndim != 2 or projected.shape[1] != self.config.horizon:
            raise ValueError(
                f"actions must have shape ({self.config.horizon},) or "
                f"(population, {self.config.horizon})"
            )
        if not np.isfinite(projected).all():
            raise ValueError("actions must contain only finite values")

        limit = self.config.omega_abs_max
        previous = np.full(projected.shape[0], current, dtype=np.float64)
        for step in range(self.config.horizon):
            lower = np.full(projected.shape[0], -limit, dtype=np.float64)
            upper = np.full(projected.shape[0], limit, dtype=np.float64)
            if self.config.max_delta_omega is not None:
                lower = np.maximum(lower, previous - self.config.max_delta_omega)
                upper = np.minimum(upper, previous + self.config.max_delta_omega)
            projected[:, step] = np.clip(projected[:, step], lower, upper)
            previous = projected[:, step]
        return projected[0] if squeeze else projected

    def _validate_current_omega(self, current_omega: float) -> float:
        current = float(current_omega)
        if not np.isfinite(current):
            raise ValueError("current_omega must be finite")
        if abs(current) > self.config.omega_abs_max:
            raise ValueError("current_omega exceeds omega_abs_max")
        return current

    def _initial_mean(
        self,
        current_omega: float,
        initial_mean: Sequence[float] | NDArray[np.floating] | None,
    ) -> FloatArray:
        if initial_mean is None:
            return np.full(self.config.horizon, current_omega, dtype=np.float64)
        candidate = np.asarray(initial_mean, dtype=np.float64)
        if candidate.shape != (self.config.horizon,):
            raise ValueError(f"initial_mean must have shape ({self.config.horizon},)")
        return self.project(candidate, current_omega=current_omega)

    @staticmethod
    def _evaluate(objective: Objective, population: FloatArray) -> FloatArray:
        costs = np.asarray(objective(population), dtype=np.float64)
        if costs.shape != (population.shape[0],):
            raise ValueError(
                "objective must return one cost per sequence with shape "
                f"({population.shape[0]},), got {costs.shape}"
            )
        costs = np.where(np.isfinite(costs), costs, np.inf)
        if np.isinf(costs).all():
            raise ValueError("objective returned no finite costs")
        return costs
