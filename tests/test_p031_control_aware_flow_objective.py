import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/p029_control_aware_flow_objective.py"
spec = importlib.util.spec_from_file_location("p029_objective", SCRIPT)
objective = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = objective
spec.loader.exec_module(objective)

CANONICAL_P029_SCRIPT = Path.cwd() / "scripts/p029_control_aware_flow_objective.py"
assert CANONICAL_P029_SCRIPT.is_file(), "run tests from the canonical repository root"
canonical_spec = importlib.util.spec_from_file_location(
    "canonical_p029_objective", CANONICAL_P029_SCRIPT
)
canonical = importlib.util.module_from_spec(canonical_spec)
sys.modules[canonical_spec.name] = canonical
canonical_spec.loader.exec_module(canonical)


P028_SCRIPT = Path.cwd() / "scripts/p028_flow_h10_objective.py"
assert P028_SCRIPT.is_file(), "run tests from the canonical repository root"
p028_spec = importlib.util.spec_from_file_location("canonical_p028_objective", P028_SCRIPT)
p028 = importlib.util.module_from_spec(p028_spec)
sys.modules[p028_spec.name] = p028
p028_spec.loader.exec_module(p028)


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        raw = inputs.new_zeros((inputs.shape[0], 7, *inputs.shape[-2:]))
        raw[:, :3] = self.weight * inputs[:, :3]
        return raw


class Aero(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, inputs):
        raw = inputs.new_zeros((inputs.shape[0], 7, *inputs.shape[-2:]))
        raw[:, 3:] = self.weight * inputs[:, :1]
        return raw


def make_inputs(state, mask, now, nxt):
    return torch.cat((state, mask, now[:, :, None, None].expand_as(mask),
                      nxt[:, :, None, None].expand_as(mask)), dim=1)


def predict(model, inputs, mask):
    raw = model(inputs)
    force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1))
    return raw[:, :3], force


def inputs():
    initial = torch.ones(1, 3, 2, 2)
    target = torch.zeros(1, 100, 3, 2, 2)
    mask = torch.ones(1, 1, 2, 2)
    actions = torch.arange(101, dtype=torch.float32).reshape(1, 101, 1) / 100
    force = torch.zeros(1, 100, 4)
    return initial, target, mask, actions, force


def losses(flow=None):
    flow = Flow() if flow is None else flow
    aero = Aero().eval()
    value = objective.control_aware_rollout_losses(
        flow, aero, *inputs(), predict, make_inputs
    )
    return flow, aero, value


def test_first_force_has_no_flow_dependency_later_force_does():
    flow, _, value = losses()
    first = torch.autograd.grad(
        value["force_per_step"][:, 0].sum(), flow.weight,
        retain_graph=True, allow_unused=True,
    )[0]
    later = torch.autograd.grad(value["force_per_step"][:, 1:].sum(), flow.weight)[0]
    assert first is None or torch.equal(first, torch.zeros_like(flow.weight))
    assert later is not None and later.abs() > 0


def test_default_h10_is_exactly_backward_compatible():
    args = (Flow(), Aero().eval(), *inputs(), predict, make_inputs)
    old = canonical.field_force_rollout_objective(
        *args, field_scale=2.0, force_scale=3.0, backward=False,
    )
    default = objective.field_force_rollout_objective(
        *args, field_scale=2.0, force_scale=3.0, backward=False,
    )
    explicit = objective.field_force_rollout_objective(
        *args, field_scale=2.0, force_scale=3.0, backward=False, horizon=10,
    )
    assert default == old == explicit


def test_h25_has_exact_steps_and_full_terminal_gradient():
    flow = Flow()
    aero = Aero().eval()
    value = objective.control_aware_rollout_losses(
        flow, aero, *inputs(), predict, make_inputs, horizon=25,
    )
    assert value["field_per_step"].shape == (1, 25)
    assert value["force_per_step"].shape == (1, 25)
    terminal = value["field_per_step"][:, -1].sum()
    gradient = torch.autograd.grad(terminal, flow.weight, retain_graph=True)[0]
    expected = 50 * (1 + flow.weight.detach()) ** 49
    torch.testing.assert_close(gradient, expected)
    first = torch.autograd.grad(
        value["force_per_step"][:, 0].sum(), flow.weight,
        retain_graph=True, allow_unused=True,
    )[0]
    later = torch.autograd.grad(value["force_per_step"][:, 1:].sum(), flow.weight)[0]
    assert first is None or torch.equal(first, torch.zeros_like(flow.weight))
    assert later is not None and later.abs() > 0


@pytest.mark.parametrize("horizon", [True, 9, 11, 24, 26, 100])
def test_horizon_is_exactly_predeclared(horizon):
    with pytest.raises(ValueError, match="exactly 10 or 25"):
        objective.control_aware_rollout_losses(
            Flow(), Aero().eval(), *inputs(), predict, make_inputs, horizon=horizon,
        )


def test_frozen_aero_has_no_grad_but_flow_receives_joint_gradient():
    flow = Flow()
    aero = Aero().eval()
    record = objective.field_force_rollout_objective(
        flow, aero, *inputs(), predict, make_inputs,
        field_scale=2.0, force_scale=3.0, backward=True,
    )
    assert flow.weight.grad is not None and torch.isfinite(flow.weight.grad)
    assert aero.weight.grad is None and not aero.weight.requires_grad
    assert record["force_terms_with_flow_gradient"] == 9
    assert record["terminal_state_force_supervised"] is False


def test_field_term_matches_existing_p028_arithmetic():
    flow, _, value = losses()
    initial, target, mask, actions, _ = inputs()
    expected = p028.flow_rollout_objective(
        flow, initial, target, mask, actions, predict, make_inputs, backward=False,
    )
    torch.testing.assert_close(
        value["field_per_step"].mean(), torch.tensor(expected["total"])
    )
    torch.testing.assert_close(
        value["field_per_step"].mean(0), torch.tensor(expected["per_step"])
    )


def test_terminal_field_step_has_full_flow_gradient():
    flow, _, value = losses()
    gradient = torch.autograd.grad(value["field_per_step"][:, -1].sum(), flow.weight)[0]
    assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0


def test_action_and_target_force_timing_is_current_to_next():
    class ActionAero(Aero):
        def forward(self, x):
            raw = x.new_zeros((x.shape[0], 7, *x.shape[-2:]))
            raw[:, 3:] = x[:, 5:6]
            return raw

    initial, target, mask, actions, force = inputs()
    force[:, :, :] = actions[:, 1:, :]
    value = objective.control_aware_rollout_losses(
        Flow(), ActionAero().eval(), initial, target, mask, actions, force,
        predict, make_inputs,
    )
    torch.testing.assert_close(value["force_per_step"], torch.zeros_like(value["force_per_step"]))


@pytest.mark.parametrize("field_scale,force_scale", [(0, 1), (1, 0), (-1, 1),
                                                       (float("nan"), 1), (1, float("inf"))])
def test_scales_fail_closed(field_scale, force_scale):
    with pytest.raises(ValueError):
        objective.field_force_rollout_objective(
            Flow(), Aero().eval(), *inputs(), predict, make_inputs,
            field_scale=field_scale, force_scale=force_scale, backward=False,
        )


def test_mask_excludes_solid_cells_and_normalization_is_exact():
    initial, target, mask, actions, force = inputs()
    mask[:, :, 1, :] = 0
    target[:, :, :, 1, :] = 1e6
    value = objective.field_force_rollout_objective(
        Flow(), Aero().eval(), initial, target, mask, actions, force,
        predict, make_inputs, field_scale=2, force_scale=4, backward=False,
    )
    assert value["total"] == pytest.approx(
        0.5*value["raw_field"]/2 + 0.5*value["raw_force"]/4
    )


def test_every_batch_member_requires_nonempty_mask():
    values = [value.repeat(2, *([1] * (value.ndim - 1))) for value in inputs()]
    values[2][1] = 0
    with pytest.raises(ValueError, match="every batch member"):
        objective.control_aware_rollout_losses(
            Flow(), Aero().eval(), *values, predict, make_inputs,
        )


def test_rejects_trainable_aero_wrong_shapes_and_nonbool_backward():
    aero = Aero().eval()
    aero.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        objective.control_aware_rollout_losses(Flow(), aero, *inputs(), predict, make_inputs)
    aero.weight.requires_grad_(False)
    aero.weight.grad = torch.ones_like(aero.weight)
    with pytest.raises(ValueError, match="no gradients"):
        objective.control_aware_rollout_losses(Flow(), aero, *inputs(), predict, make_inputs)
    initial, target, mask, actions, force = inputs()
    with pytest.raises(ValueError, match="target_force"):
        objective.control_aware_rollout_losses(
            Flow(), Aero().eval(), initial, target, mask, actions, force[:, :-1],
            predict, make_inputs,
        )
    with pytest.raises(ValueError, match="eval"):
        objective.control_aware_rollout_losses(
            Flow(), Aero().train(), *inputs(), predict, make_inputs,
        )
    with pytest.raises(ValueError, match="exact bool"):
        objective.field_force_rollout_objective(
            Flow(), Aero().eval(), *inputs(), predict, make_inputs,
            field_scale=1, force_scale=1, backward=1,
        )
    scale = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="fixed finite positive"):
        objective.field_force_rollout_objective(
            Flow(), Aero().eval(), *inputs(), predict, make_inputs,
            field_scale=scale, force_scale=1, backward=False,
        )
