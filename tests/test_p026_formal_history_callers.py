from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    import pytest
except ModuleNotFoundError:  # The pinned official image intentionally omits pytest.
    @contextmanager
    def _raises(error, match=None):
        try:
            yield
        except error as caught:
            if match is not None and re.search(match, str(caught)) is None:
                raise AssertionError(f"exception did not match {match!r}: {caught}")
        else:
            raise AssertionError(f"expected {error.__name__}")

    class _PytestFallback:
        raises = staticmethod(_raises)

    pytest = _PytestFallback()

import evaluate_tandem_fno as evaluation
from fluid_control.dual_fno import combine_dual_raw, make_dual_fno_adapter
from p026_history_inference import (
    make_history_dual_fno_adapter,
    reset_history,
    trajectory_history,
)


class Toy(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(7 * channels, dtype=torch.float32).reshape(7, channels)
            / (7 * channels * 10)
        )
        self.seen = []

    def forward(self, value):
        self.seen.append(value.detach().clone())
        return torch.einsum("oc,bchw->bohw", self.weight, value)


def legacy_rollout(network, states, actions, starts, horizon, mask):
    predicted = states[starts].clone()
    raw = None
    for offset in range(horizon):
        indices = starts + offset
        now = actions[indices, None, None, None].expand(-1, 1, *mask.shape[-2:])
        following = actions[indices + 1, None, None, None].expand_as(now)
        raw = network(torch.cat((predicted, mask, now, following), dim=1))
        predicted = (predicted + raw[:, :3]) * mask
    return predicted, raw


def history_rollout(network, states, actions, starts, horizon, mask, k):
    buffer = trajectory_history(states, actions, starts, k=k)
    raw = None
    for offset in range(horizon):
        raw, predicted, buffer = evaluation.history_dual_step(
            network, buffer, mask, actions[starts + offset + 1]
        )
    return predicted, raw, buffer


def test_k1_actual_caller_matches_legacy_all_formal_horizons_and_starts():
    torch.manual_seed(12)
    states = torch.randn(126, 3, 3, 4)
    actions = torch.linspace(-0.5, 0.5, len(states))
    mask_all = torch.ones(len(states), 1, 3, 4)
    for horizon in (1, 10, 50, 100):
        starts = torch.arange(0, len(states) - horizon, 25)
        mask = mask_all[starts]
        old_flow, old_aero = Toy(6), Toy(6)
        new_flow, new_aero = copy.deepcopy(old_flow), copy.deepcopy(old_aero)
        old = make_dual_fno_adapter(old_flow, old_aero)
        new = make_history_dual_fno_adapter(new_flow, new_aero, k=1)
        legacy_state, legacy_raw = legacy_rollout(
            old, states, actions, starts, horizon, mask
        )
        history_state, history_raw, _ = history_rollout(
            new, states, actions, starts, horizon, mask, 1
        )
        torch.testing.assert_close(history_state, legacy_state, rtol=0, atol=0)
        torch.testing.assert_close(history_raw, legacy_raw, rtol=0, atol=0)
        target = states[starts + horizon]
        legacy_sums = evaluation.field_error_sums(
            legacy_state - target, target, mask
        )
        history_sums = evaluation.field_error_sums(
            history_state - target, target, mask
        )
        assert evaluation.relative_field_metrics(
            history_sums.cpu().numpy()
        ) == evaluation.relative_field_metrics(legacy_sums.cpu().numpy())


def test_k4_caller_routes_18_channels_and_never_reads_future_observation():
    torch.manual_seed(4)
    states = torch.randn(15, 3, 2, 3)
    actions = torch.linspace(-0.3, 0.3, len(states))
    starts = torch.tensor([2, 5])
    mask = torch.ones(2, 1, 2, 3)
    flow, aero = Toy(6), Toy(18)
    expected, raw, buffer = history_rollout(
        make_history_dual_fno_adapter(flow, aero, k=4),
        states,
        actions,
        starts,
        4,
        mask,
        4,
    )
    poisoned = states.clone()
    poisoned[6:] = 1.0e6
    flow2, aero2 = copy.deepcopy(flow), copy.deepcopy(aero)
    actual, raw2, buffer2 = history_rollout(
        make_history_dual_fno_adapter(flow2, aero2, k=4),
        poisoned,
        actions,
        starts,
        4,
        mask,
        4,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(raw2, raw, rtol=0, atol=0)
    assert aero.seen and all(value.shape[1] == 18 for value in aero.seen)
    assert flow.seen and all(value.shape[1] == 6 for value in flow.seen)
    assert buffer.source == buffer2.source == "autoregressive_prediction"


def test_force_window_reset_uses_frame0_padding_and_k1_matches_legacy():
    torch.manual_seed(8)
    q0 = torch.randn(1, 3, 2, 2)
    actions = torch.linspace(0.0, 0.5, 6)
    mask = torch.ones(1, 1, 2, 2)
    old_flow, old_aero = Toy(6), Toy(6)
    new_flow, new_aero = copy.deepcopy(old_flow), copy.deepcopy(old_aero)
    old = make_dual_fno_adapter(old_flow, old_aero)
    new = make_history_dual_fno_adapter(new_flow, new_aero, k=1)
    repeated = q0.expand(len(actions), -1, -1, -1).clone()
    starts = torch.tensor([0])
    legacy_state, legacy_raw = legacy_rollout(old, repeated, actions, starts, 5, mask)
    buffer = reset_history(q0, actions[:1], k=1)
    raw = None
    for step in range(5):
        raw, state, buffer = evaluation.history_dual_step(
            new, buffer, mask, actions[step + 1 : step + 2]
        )
    torch.testing.assert_close(state, legacy_state, rtol=0, atol=0)
    torch.testing.assert_close(raw, legacy_raw, rtol=0, atol=0)
    k4 = reset_history(q0, actions[:1], k=4)
    assert k4.padding_mask.tolist() == [[True, True, True, False]]
    assert torch.equal(k4.states, q0[:, None].expand_as(k4.states))


def test_explicit_profile_and_engineering_manifest_contract_fail_closed():
    k4 = make_history_dual_fno_adapter(Toy(6), Toy(18), k=4)
    with pytest.raises(ValueError, match="explicit profile"):
        evaluation.history_profile_length(
            "legacy_k1", use_dual_fno=True, network=k4, manifest_payload={}
        )
    with pytest.raises(ValueError, match="explicit dual"):
        evaluation.history_profile_length(
            "p026_k4", use_dual_fno=False, network=k4, manifest_payload={}
        )
    with pytest.raises(ValueError, match="manifest history"):
        evaluation.history_profile_length(
            "p026_k4", use_dual_fno=True, network=k4, manifest_payload={}
        )
    payload = {"history_input": evaluation.HISTORY_PROFILES["p026_k4"]}
    assert (
        evaluation.history_profile_length(
            "p026_k4", use_dual_fno=True, network=k4, manifest_payload=payload
        )
        == 4
    )
    k1 = make_history_dual_fno_adapter(Toy(6), Toy(6), k=1)
    with pytest.raises(ValueError, match="explicit profile"):
        evaluation.history_profile_length(
            "legacy_k1", use_dual_fno=True, network=k1, manifest_payload={}
        )
    with pytest.raises(ValueError, match="history length"):
        evaluation.history_profile_length(
            "p026_k4", use_dual_fno=True, network=k1, manifest_payload=payload
        )


def test_raw_channel_combination_is_unchanged():
    flow = torch.randn(2, 7, 3, 4)
    aero = torch.randn(2, 7, 3, 4)
    result = combine_dual_raw(flow, aero)
    assert torch.equal(result[:, :3], flow[:, :3])
    assert torch.equal(result[:, 3:], aero[:, 3:])


def test_diagnostic_copy_uses_reset_history_and_shared_step():
    source = (Path(__file__).parents[1] / "scripts/diagnose_fno_force_window.py").read_text()
    assert "reset_history(" in source
    assert "history_dual_step(" in source
    assert 'choices=("legacy_k1", "p026_k1", "p026_k4")' in source


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().copy().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} staged formal-caller tests passed")
