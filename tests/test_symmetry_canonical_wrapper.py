import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from symmetry_canonical_wrapper import (
    CanonicalSymmetryWrapper,
    canonicalize_physical69,
    reflect_physical69,
    restore_physical_action,
)


def observation(seed=7):
    return np.random.default_rng(seed).normal(size=69).astype(np.float32)


def test_reflection_involution_and_orbit_canonicalization():
    original = observation()
    mirrored = reflect_physical69(original)
    assert np.array_equal(reflect_physical69(mirrored), original)
    left = canonicalize_physical69(original)
    right = canonicalize_physical69(mirrored)
    assert np.array_equal(left.value, right.value)
    assert left.pivot_index == right.pivot_index
    assert left.odd_margin == right.odd_margin > 0.0
    assert left.orientation == -right.orientation


def test_action_restore_is_invertible_odd_and_bound_preserving():
    obs = observation(9)
    mirrored = reflect_physical69(obs)
    left = canonicalize_physical69(obs)
    right = canonicalize_physical69(mirrored)
    canonical_action = np.asarray([0.37], dtype=np.float32)
    physical_left = restore_physical_action(canonical_action, left.orientation)
    physical_right = restore_physical_action(canonical_action, right.orientation)
    assert np.array_equal(physical_left, -physical_right)
    assert np.array_equal(
        restore_physical_action(physical_left, left.orientation), canonical_action
    )
    assert abs(float(physical_left[0])) <= 0.75


def test_tie_is_deterministic_and_exact_fixed_state_is_explicit_limit():
    tied = np.zeros(69, dtype=np.float32)
    tied[65] = 0.25
    tied[67] = 0.25
    item = canonicalize_physical69(tied)
    assert item.pivot_index == 65
    fixed = canonicalize_physical69(np.zeros(69, dtype=np.float32))
    assert fixed.reflection_fixed
    assert fixed.odd_margin == 0.0
    assert fixed.orientation == 1
    # The adapter deliberately does not pretend a nonzero action is odd at a
    # reflection-fixed observation; forcing zero would be non-invertible.
    assert restore_physical_action(np.asarray([0.2], np.float32), 1)[0] == np.float32(0.2)


class RecordingEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (69,), np.float32)
    action_space = gym.spaces.Box(-0.75, 0.75, (1,), np.float32)

    def __init__(self, initial, following):
        self.initial = initial
        self.following = following
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.initial.copy(), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.following.copy(), 1.0, False, False, {}


def test_actual_gym_wrapper_uses_current_orientation_once_then_updates_next():
    initial = observation(11)
    following = reflect_physical69(observation(13))
    base = RecordingEnv(initial, following)
    wrapped = CanonicalSymmetryWrapper(base)
    canonical_initial, initial_info = wrapped.reset()
    sampled = np.asarray([0.42], np.float32)
    expected_physical = restore_physical_action(
        sampled, initial_info["symmetry_orientation"]
    )
    canonical_next, _, _, _, next_info = wrapped.step(sampled)
    assert np.array_equal(base.actions[0], expected_physical)
    assert np.array_equal(canonical_initial, canonicalize_physical69(initial).value)
    assert np.array_equal(canonical_next, canonicalize_physical69(following).value)
    assert next_info["symmetry_orientation"] == canonicalize_physical69(following).orientation
    assert next_info["symmetry_applied_orientation"] == initial_info["symmetry_orientation"]
    assert next_info["symmetry_next_orientation"] == canonicalize_physical69(following).orientation
    assert next_info["symmetry_canonical_action_received"] == sampled.tolist()
    assert next_info["symmetry_physical_requested_action"] == expected_physical.tolist()


def test_reward_invariant_terms_under_global_sign_reversal():
    omega, delta = 0.41, -0.07
    rear_cl = np.asarray([-0.4, 0.7, -0.2, 0.1], dtype=np.float64)
    assert np.mean(rear_cl) ** 2 == np.mean(-rear_cl) ** 2
    assert np.std(rear_cl) == np.std(-rear_cl)
    assert omega**2 == (-omega) ** 2
    assert delta**2 == (-delta) ** 2


class TruncatingRecordingEnv(RecordingEnv):
    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.following.copy(), 1.0, False, True, {}


class AlwaysContinue(BaseCallback):
    def _on_step(self):
        return True


def test_real_sb3_dummyvec_keeps_latent_action_logprob_and_canonical_terminal():
    initial = observation(21)
    terminal = reflect_physical69(observation(22))
    bases = []

    def factory():
        base = TruncatingRecordingEnv(initial, terminal)
        bases.append(base)
        return CanonicalSymmetryWrapper(base)

    vec = DummyVecEnv([factory])
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=20261007,
        device="cpu",
        policy_kwargs={"net_arch": [8]},
    )
    _, callback = model._setup_learn(total_timesteps=2)
    assert isinstance(callback, BaseCallback)
    assert model.collect_rollouts(vec, callback, model.rollout_buffer, n_rollout_steps=2)
    latent_actions = model.rollout_buffer.actions[:, 0, :]
    orientation = canonicalize_physical69(initial).orientation
    for latent, physical in zip(latent_actions, bases[0].actions, strict=True):
        clipped = np.clip(latent, -0.75, 0.75).astype(np.float32)
        assert np.array_equal(physical, restore_physical_action(clipped, orientation))
    observations = torch.as_tensor(
        model.rollout_buffer.observations[:, 0, :], dtype=torch.float32
    )
    actions = torch.as_tensor(latent_actions, dtype=torch.float32)
    with torch.no_grad():
        _, current_log_prob, _ = model.policy.evaluate_actions(observations, actions)
    np.testing.assert_allclose(
        current_log_prob.numpy(), model.rollout_buffer.log_probs[:, 0], rtol=1e-6, atol=1e-6
    )
    # DummyVecEnv autoresets after truncation, but exposes the terminal observation
    # produced by the wrapper rather than the reset observation.
    vec.reset()
    _, _, _, infos = vec.step(np.asarray([[0.1]], dtype=np.float32))
    assert np.array_equal(
        infos[0]["terminal_observation"], canonicalize_physical69(terminal).value
    )
    assert infos[0]["TimeLimit.truncated"] is True


def test_wrong_action_bounds_are_rejected():
    base = RecordingEnv(observation(31), observation(32))
    base.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    try:
        CanonicalSymmetryWrapper(base)
    except ValueError as error:
        assert "[-.75,.75]" in str(error)
    else:
        raise AssertionError("wrong action bounds must be rejected")
