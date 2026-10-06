from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/p028_flow_h10_objective.py"
SPEC = importlib.util.spec_from_file_location("p028_objective", SCRIPT)
objective = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(objective)


def make_inputs(state, mask, now, nxt):
    height, width = state.shape[-2:]
    return torch.cat(
        (state, mask, now.reshape(-1, 1, 1, 1).expand(-1, 1, height, width),
         nxt.reshape(-1, 1, 1, 1).expand(-1, 1, height, width)),
        dim=1,
    )


class ToyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(6, 7, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


def predict(model, inputs, mask):
    raw = model(inputs)
    force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    return raw[:, :3], force


def batch(height=2, width=2):
    return {
        "state": torch.zeros(1, 3, height, width),
        "target": torch.zeros(1, 100, 3, height, width),
        "mask": torch.ones(1, 1, height, width),
        "actions": torch.arange(101, dtype=torch.float32).reshape(1, 101, 1) / 100,
    }


def run(model, values, *, backward):
    return objective.flow_rollout_objective(
        model, values["state"], values["target"], values["mask"], values["actions"],
        predict, make_inputs, backward=backward,
    )


def test_exact_ten_action_pairs_and_target_alignment():
    model = ToyFlow()
    values = batch()
    values["target"][:, :10] = torch.arange(1, 11).reshape(1, 10, 1, 1, 1)
    observed = []

    def recording_inputs(state, mask, now, nxt):
        observed.append((now.detach().clone(), nxt.detach().clone()))
        return make_inputs(state, mask, now, nxt)

    result = objective.flow_rollout_objective(
        model, values["state"], values["target"], values["mask"], values["actions"],
        predict, recording_inputs, backward=False,
    )
    assert len(observed) == 10
    for step, (now, nxt) in enumerate(observed):
        torch.testing.assert_close(now, values["actions"][:, step])
        torch.testing.assert_close(nxt, values["actions"][:, step + 1])
    assert result["per_step"] == pytest.approx([float((step + 1) ** 2) for step in range(10)])


def test_late_losses_backpropagate_through_early_recurrent_update():
    class FirstActionOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, inputs):
            delta = self.weight * inputs[:, 4:5]
            return torch.cat((delta.expand(-1, 3, -1, -1),
                              torch.zeros_like(inputs[:, :4])), dim=1)

    model = FirstActionOnly()
    values = batch(1, 1)
    values["actions"].zero_()
    values["actions"][:, 0] = 1
    values["target"][:, 0] = model.weight.detach()
    result = run(model, values, backward=True)
    assert result["per_step"][0] == pytest.approx(0.0)
    assert model.weight.grad is not None and abs(float(model.weight.grad)) > 0


def test_mask_is_applied_to_every_recurrent_state_and_loss():
    model = ToyFlow()
    with torch.no_grad():
        model.linear.bias[:3].fill_(1)
    values = batch()
    values["mask"][:, :, 0, 1:] = 0
    values["mask"][:, :, 1] = 0
    values["target"][:, :10, :, 0, 1:] = 1e6
    values["target"][:, :10, :, 1] = -1e6
    seen_states = []

    def recording_inputs(state, mask, now, nxt):
        seen_states.append(state.detach().clone())
        return make_inputs(state, mask, now, nxt)

    result = objective.flow_rollout_objective(
        model, values["state"], values["target"], values["mask"], values["actions"],
        predict, recording_inputs, backward=True,
    )
    assert len(seen_states) == 10
    assert all(torch.count_nonzero(state[..., 0, 1:]) == 0 for state in seen_states)
    assert all(torch.count_nonzero(state[..., 1, :]) == 0 for state in seen_states)
    assert result["per_step"] == pytest.approx([float((step + 1) ** 2) for step in range(10)])


def test_unused_flow_force_rows_receive_zero_gradient():
    model = ToyFlow()
    values = batch()
    values["target"][:, :10] = 1
    run(model, values, backward=True)
    assert torch.count_nonzero(model.linear.weight.grad[3:]) == 0
    assert torch.count_nonzero(model.linear.bias.grad[3:]) == 0
    assert torch.count_nonzero(model.linear.bias.grad[:3]) > 0


def test_eight_window_gradient_is_raw_mean_without_step():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    values = [torch.tensor([[float(index + 1) / 100]]) for index in range(8)]

    def one_window(value):
        loss = model(value).square().mean()
        loss.backward()
        return {"total": float(loss.detach()), "rollout_steps": 10}

    result = objective.accumulate_eight_window_gradients(
        model, optimizer, values, one_window
    )
    expected = 2 * 0.25 * sum(float(value.square()) for value in values) / 8
    assert float(model.weight.grad) == pytest.approx(expected)
    assert result["windows"] == 8 and result["optimizer_step_performed"] is False
    assert float(model.weight.detach()) == pytest.approx(0.25)


@pytest.mark.parametrize("count", [7, 9])
def test_accumulation_rejects_non_eight_groups(count):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def one_window(value):
        loss = model(value).square().mean()
        loss.backward()
        return {"total": float(loss.detach()), "rollout_steps": 10}

    with pytest.raises(ValueError, match="incomplete|too many"):
        objective.accumulate_eight_window_gradients(
            model, optimizer, [torch.ones(1, 1)] * count, one_window
        )


def test_original_h100_shapes_are_fail_closed():
    model = ToyFlow()
    values = batch()
    with pytest.raises(ValueError, match="100"):
        objective.flow_rollout_objective(
            model, values["state"], values["target"][:, :10], values["mask"],
            values["actions"], predict, make_inputs, backward=False,
        )


def test_exact_four_unused_channels_and_boolean_backward_required():
    model = ToyFlow()
    values = batch()

    def wrong_force(_model, inputs, _mask):
        raw = _model(inputs)
        return raw[:, :3], raw[:, 3:6].mean((-2, -1))

    with pytest.raises(ValueError, match="shape differs"):
        objective.flow_rollout_objective(
            model, values["state"], values["target"], values["mask"], values["actions"],
            wrong_force, make_inputs, backward=False,
        )
    with pytest.raises(ValueError, match="exact bool"):
        objective.flow_rollout_objective(
            model, values["state"], values["target"], values["mask"], values["actions"],
            predict, make_inputs, backward=1,
        )


def test_accumulation_norm_preserves_complex_gradient_magnitude():
    class ComplexParameter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(1 + 2j))

    model = ComplexParameter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def one_window(_value):
        loss = model.value.abs().square()
        loss.backward()
        return {"total": float(loss.detach()), "rollout_steps": 10}

    result = objective.accumulate_eight_window_gradients(
        model, optimizer, range(8), one_window
    )
    assert result["mean_gradient_norm_before_scope_mask_and_clip"] == pytest.approx(
        abs(complex(model.value.grad))
    )
