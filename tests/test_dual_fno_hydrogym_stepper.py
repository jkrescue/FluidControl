"""Software-only routing fixtures: no CFD data, learned accuracy or PPO claim."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluid_control.dual_fno import make_dual_fno_adapter
from fluid_control.tandem_hydrogym import TandemFNOStepper


class FixtureNetwork(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.seen = []

    def forward(self, value):
        self.seen.append(value.clone())
        return self.values[None, :, None, None].expand(value.shape[0], -1, *value.shape[-2:])


def fixture_flow(force_values=(1., 2., 3., 4.)):
    field = FixtureNetwork([0.1, 0.2, 0.3, 999., 999., 999., 999.])
    forces = FixtureNetwork([999., 999., 999., *force_values])
    flow = SimpleNamespace(
        network=make_dual_fno_adapter(field, forces), device=torch.device('cpu'),
        q=torch.zeros(3, 2, 3), mask=torch.ones(1, 2, 3), omega=0., t=0.,
        MAX_CONTROL=0.75, max_delta_omega=0.1,
        force_channels=('front_cd', 'front_cl', 'rear_cd', 'rear_cl'),
        force_mean=np.array([10., 20., 30., 40.], dtype=np.float32),
        force_std=np.array([2., 3., 4., 5.], dtype=np.float32),
        reward_samples=0,
    )
    flow.set_control = lambda action: setattr(flow, 'last_set_control', action)
    flow.record_reward_sample = lambda: setattr(flow, 'reward_samples', flow.reward_samples + 1)
    return flow, field, forces


def test_actual_hydrogym_step_uses_flow_delta_and_separate_four_forces():
    flow, field, forces = fixture_flow()
    stepper = SimpleNamespace(flow=flow, dt=0.1)
    for iteration in range(2):
        returned = TandemFNOStepper.step(stepper, iteration, control=[0.75])
        assert returned is flow
        expected = torch.tensor([0.1, 0.2, 0.3])[:, None, None].expand_as(flow.q) * (iteration + 1)
        torch.testing.assert_close(flow.q, expected)
        np.testing.assert_array_equal(flow.force, [12., 26., 42., 60.])
        assert flow.omega == pytest.approx((iteration + 1) * 0.1)
        assert flow.applied_delta == pytest.approx(0.1)
        assert flow.rate_limited and flow.reward_samples == iteration + 1
        torch.testing.assert_close(field.seen[-1], forces.seen[-1])
        torch.testing.assert_close(field.seen[-1][:, -1], torch.full((1, 2, 3), flow.omega / 0.75))
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in flow.network.parameters())


def test_nonfinite_force_blocks_actual_step_before_state_update():
    flow, _, _ = fixture_flow((1., 2., 3., float('nan')))
    original = flow.q.clone()
    with pytest.raises(FloatingPointError):
        TandemFNOStepper.step(SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.1])
    torch.testing.assert_close(flow.q, original)
    assert flow.t == 0 and flow.reward_samples == 0
