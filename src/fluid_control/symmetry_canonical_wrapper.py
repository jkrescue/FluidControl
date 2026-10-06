"""Prospective policy-independent symmetry adapter for the physical69 contract.

This source is an engineering feasibility artifact.  It does not train or load a
policy, model, CFD case, or dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


ACTION_LIMIT = 0.75


def physical69(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (69,) or not np.isfinite(result).all():
        raise ValueError("physical69 observation must be finite with shape (69,)")
    return result


def reflect_physical69(value: np.ndarray) -> np.ndarray:
    """Exact y=7.5 reflection already used by the audited deployment driver."""
    source = physical69(value)
    result = source.copy()
    probes = source[:64].reshape(32, 2)
    result[:64] = (probes[::-1] * np.asarray([1.0, -1.0], np.float32)).reshape(-1)
    result[64] = source[64]
    result[65] = -source[65]
    result[66] = source[66]
    result[67] = -source[67]
    result[68] = -source[68]
    if not np.array_equal(reflect_physical69_unchecked(result), source):
        raise AssertionError("physical69 reflection must be involutive")
    return result


def reflect_physical69_unchecked(source: np.ndarray) -> np.ndarray:
    result = source.copy()
    probes = source[:64].reshape(32, 2)
    result[:64] = (probes[::-1] * np.asarray([1.0, -1.0], np.float32)).reshape(-1)
    result[64] = source[64]
    result[65] = -source[65]
    result[66] = source[66]
    result[67] = -source[67]
    result[68] = -source[68]
    return result


@dataclass(frozen=True)
class CanonicalObservation:
    value: np.ndarray
    orientation: int
    pivot_index: int
    odd_margin: float
    reflection_fixed: bool


def canonicalize_physical69(value: np.ndarray) -> CanonicalObservation:
    """Choose one representative of {o, R(o)} without consulting a policy.

    For a non-fixed orbit, the first largest-magnitude component of o-R(o)
    defines orientation.  The absolute vector is identical after reflection,
    while its sign reverses, so both orbit members map to the same representative.
    Exact fixed points use orientation +1 and are explicitly flagged: no
    invertible sign map can force a nonzero stochastic action to be odd there.
    """
    source = physical69(value)
    mirrored = reflect_physical69(source)
    odd = source - mirrored
    pivot = int(np.argmax(np.abs(odd)))
    margin = float(abs(odd[pivot]))
    fixed = bool(margin == 0.0)
    orientation = 1 if fixed or float(odd[pivot]) > 0.0 else -1
    canonical = source.copy() if orientation == 1 else mirrored
    return CanonicalObservation(canonical, orientation, pivot, margin, fixed)


def restore_physical_action(canonical_action: np.ndarray, orientation: int) -> np.ndarray:
    action = np.asarray(canonical_action, dtype=np.float32)
    if action.shape != (1,) or not np.isfinite(action).all():
        raise ValueError("canonical action must be finite with shape (1,)")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    if abs(float(action[0])) > ACTION_LIMIT + 1e-7:
        raise ValueError("canonical action outside existing support")
    return np.asarray([orientation * float(action[0])], dtype=np.float32)


class CanonicalSymmetryWrapper(gym.Wrapper):
    """Expose canonical observations/actions while the base env stays physical.

    The orientation cached from the current observation transforms the current
    sampled action.  The next orientation is computed only after env.step.
    Hence the existing base-environment rate/amplitude filter remains the single
    physical filter.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.observation_space.shape != (69,) or env.action_space.shape != (1,):
            raise ValueError("wrapper requires physical69 and scalar action spaces")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("wrapper requires a Box action space")
        if not (
            np.array_equal(env.action_space.low, np.asarray([-ACTION_LIMIT]))
            and np.array_equal(env.action_space.high, np.asarray([ACTION_LIMIT]))
        ):
            raise ValueError("wrapper requires the existing [-.75,.75] action contract")
        self._orientation: int | None = None

    @staticmethod
    def _info(info: dict, item: CanonicalObservation) -> dict:
        result = dict(info)
        result["symmetry_orientation"] = item.orientation
        result["symmetry_pivot_index"] = item.pivot_index
        result["symmetry_odd_margin"] = item.odd_margin
        result["symmetry_reflection_fixed"] = item.reflection_fixed
        return result

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        item = canonicalize_physical69(observation)
        self._orientation = item.orientation
        return item.value, self._info(info, item)

    def step(self, action):
        if self._orientation is None:
            raise RuntimeError("reset must precede step")
        applied_orientation = self._orientation
        canonical_action = np.asarray(action, dtype=np.float32).copy()
        physical_action = restore_physical_action(canonical_action, applied_orientation)
        observation, reward, terminated, truncated, info = self.env.step(physical_action)
        item = canonicalize_physical69(observation)
        self._orientation = item.orientation
        result_info = self._info(info, item)
        result_info.update(
            {
                "symmetry_applied_orientation": applied_orientation,
                # SB3 clips before env.step; this is the exact latent action the
                # wrapper receives, not a claim about the pre-clip Gaussian draw.
                "symmetry_canonical_action_received": canonical_action.tolist(),
                "symmetry_physical_requested_action": physical_action.tolist(),
                "symmetry_next_orientation": item.orientation,
            }
        )
        return item.value, reward, terminated, truncated, result_info
