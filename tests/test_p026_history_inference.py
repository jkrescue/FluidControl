from __future__ import annotations

import copy

import pytest
import torch

import p026_history_inference as inference
from fluid_control.dual_fno import combine_dual_raw


class Toy(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, value):
        base = value.mean(dim=1, keepdim=True) * self.scale
        return torch.cat([base + channel for channel in range(7)], dim=1)


def packed(buffer, mask, following):
    return inference.pack_history_input(buffer, mask, following)


def test_k1_adapter_is_legacy_raw_equivalent():
    torch.manual_seed(2)
    flow, aero = Toy(0.2), Toy(-0.3)
    flow_reference, aero_reference = copy.deepcopy(flow), copy.deepcopy(aero)
    value = torch.randn(3, 6, 4, 5)
    expected = combine_dual_raw(flow_reference(value), aero_reference(value))
    adapter = inference.make_history_dual_fno_adapter(flow, aero, k=1)
    torch.testing.assert_close(adapter(value), expected, rtol=0, atol=0)
    torch.testing.assert_close(adapter(value, value.clone()), expected, rtol=0, atol=0)


def test_k4_requires_explicit_well_formed_second_input():
    adapter = inference.make_history_dual_fno_adapter(Toy(0.1), Toy(0.2), k=4)
    flow = torch.zeros(2, 6, 3, 4)
    with pytest.raises(ValueError, match="explicit 18-channel"):
        adapter(flow)
    with pytest.raises(ValueError, match="shape differs"):
        adapter(flow, torch.zeros(2, 17, 3, 4))
    with pytest.raises(ValueError, match="shape differs"):
        adapter(flow, torch.zeros(1, 18, 3, 4))


@pytest.mark.parametrize("channel", [9, 12, 16, 17])
def test_k4_rejects_current_state_mask_or_action_mismatch(channel):
    q0 = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4)
    omega0 = torch.tensor([0.1, -0.2])
    history = inference.reset_history(q0, omega0, k=4)
    mask = torch.ones(2, 1, 2, 4)
    following = torch.tensor([0.2, -0.1])
    aero = packed(history, mask, following)
    flow = packed(
        inference.HistoryBuffer(
            history.states[:, -1:],
            history.actions[:, -1:],
            history.padding_mask[:, -1:],
            history.source,
        ),
        mask,
        following,
    )
    changed = aero.clone()
    changed[:, channel] += 1
    adapter = inference.make_history_dual_fno_adapter(Toy(0.1), Toy(0.2), k=4)
    with pytest.raises(ValueError, match="current history"):
        adapter(flow, changed)
    assert adapter(flow, aero).shape == (2, 7, 2, 4)


def test_trajectory_history_retains_real_past_and_only_left_pads():
    states = torch.arange(6 * 3 * 2 * 2, dtype=torch.float32).reshape(6, 3, 2, 2)
    actions = torch.arange(6, dtype=torch.float32) / 10
    starts = torch.tensor([0, 1, 2, 4])
    result = inference.trajectory_history(states, actions, starts, k=4)
    expected = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2], [1, 2, 3, 4]]
    for row, indices in enumerate(expected):
        assert torch.equal(result.states[row], states[indices])
        assert torch.equal(result.actions[row], actions[indices])
    assert result.padding_mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
        [False, False, False, False],
    ]
    states[5].fill_(9999)
    assert not torch.any(result.states == 9999)


def test_causal_shift_uses_prediction_and_preserves_gradient():
    q0 = torch.zeros(1, 3, 2, 2)
    initial = inference.reset_history(q0, torch.tensor([0.0]), k=4)
    predicted = torch.full((1, 3, 2, 2), 3.0, requires_grad=True)
    shifted = inference.advance_history(initial, predicted, torch.tensor([0.25]))
    assert torch.equal(shifted.states[0, -1], predicted[0])
    assert shifted.actions.tolist() == [[0.0, 0.0, 0.0, 0.25]]
    assert shifted.padding_mask.tolist() == [[True, True, False, False]]
    assert shifted.source == "autoregressive_prediction"
    shifted.states.sum().backward()
    assert torch.equal(predicted.grad, torch.ones_like(predicted))


def test_reset_copy_restore_and_independent_env_buffers_do_not_alias():
    first = inference.reset_history(
        torch.ones(1, 3, 2, 2), torch.tensor([0.1]), k=4
    )
    second = inference.reset_history(
        torch.full((1, 3, 2, 2), 2.0), torch.tensor([-0.2]), k=4
    )
    saved = inference.clone_history(first)
    first.states[0, 0, 0, 0, 0] = 99
    first.actions[0, 0] = 99
    assert saved.states[0, 0, 0, 0, 0] == 1
    assert saved.actions[0, 0] == pytest.approx(0.1)
    assert torch.all(second.states == 2) and torch.all(second.actions == -0.2)
    restored = inference.clone_history(saved)
    restored.states.add_(5)
    assert torch.all(saved.states == 1)


def test_adapter_has_no_cross_environment_history_state():
    adapter = inference.make_history_dual_fno_adapter(Toy(0.1), Toy(0.2), k=4)
    mask = torch.ones(1, 1, 2, 2)

    def inputs(value, action):
        buffer = inference.reset_history(
            torch.full((1, 3, 2, 2), value), torch.tensor([action]), k=4
        )
        aero = packed(buffer, mask, torch.tensor([action + 0.1]))
        flow_buffer = inference.HistoryBuffer(
            buffer.states[:, -1:],
            buffer.actions[:, -1:],
            buffer.padding_mask[:, -1:],
            buffer.source,
        )
        return packed(flow_buffer, mask, torch.tensor([action + 0.1])), aero

    flow_a, aero_a = inputs(1.0, 0.0)
    flow_b, aero_b = inputs(2.0, -0.2)
    first = adapter(flow_a, aero_a)
    _ = adapter(flow_b, aero_b)
    torch.testing.assert_close(adapter(flow_a, aero_a), first, rtol=0, atol=0)
    assert not hasattr(adapter, "history") and not hasattr(adapter, "buffer")
