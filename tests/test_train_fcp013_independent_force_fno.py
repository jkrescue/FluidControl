from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fcp013", ROOT / "scripts/train_fcp013_independent_force_fno.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class Tiny(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(float(scale)))


def predict(model, x, mask):
    delta = x[:, :3] * model.scale
    force = x[:, :4].mean(dim=(-2, -1)) * model.scale
    return delta, force


def batch():
    torch.manual_seed(4)
    return {
        "state": torch.randn(1, 3, 2, 2),
        "target_state": torch.randn(1, 100, 3, 2, 2),
        "target_force": torch.randn(1, 100, 4),
        "mask": torch.ones(1, 1, 2, 2),
        "omega": torch.linspace(-0.75, 0.75, 101)[None],
    }


def test_flow_residual_update_and_separate_model_instances():
    flow, aero = Tiny(0.1), Tiny(0.1)
    assert flow is not aero and flow.scale is not aero.scale
    flow.scale.requires_grad_(False)
    b = batch()
    states = M.frozen_flow_states(flow, b["state"], b["mask"], b["omega"], predict)
    assert torch.allclose(states[:, 0], b["state"])
    assert torch.allclose(states[:, 1], b["state"] * 1.1)
    assert states.grad_fn is None


def test_true_state_alignment_is_t_to_t_plus_one():
    b = batch(); values = M.true_state_inputs(b["state"], b["target_state"])
    assert torch.equal(values[:, 0], b["state"])
    assert torch.equal(values[:, 1:], b["target_state"][:, :-1])


def test_inputs_preserve_mask_and_both_action_endpoints():
    states = torch.arange(12.0).reshape(1, 3, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    inputs = M.make_inputs(states, mask, torch.tensor([-0.25]), torch.tensor([0.5]))
    assert torch.equal(inputs[:, :3], states)
    assert torch.equal(inputs[:, 3:4], mask)
    assert torch.equal(inputs[:, 4], torch.full((1, 2, 2), -0.25))
    assert torch.equal(inputs[:, 5], torch.full((1, 2, 2), 0.5))


def test_chunked_loss_and_gradient_equal_monolithic():
    b = batch(); flow = Tiny(0.02); flow.scale.requires_grad_(False)
    states = M.frozen_flow_states(flow, b["state"], b["mask"], b["omega"], predict)
    h1 = M.true_state_inputs(b["state"], b["target_state"])
    a, z = Tiny(0.3), Tiny(0.3)
    out_a = M.chunk_force_objective(a, states, h1, b["mask"], b["omega"], b["target_force"], predict, chunk_size=10, backward=True)
    grad_a = a.scale.grad.detach().clone()
    out_z = M.chunk_force_objective(z, states, h1, b["mask"], b["omega"], b["target_force"], predict, chunk_size=100, backward=True)
    assert out_a["total"] == pytest.approx(out_z["total"], rel=2e-6, abs=2e-7)
    assert grad_a.item() == pytest.approx(z.scale.grad.item(), rel=2e-6, abs=2e-7)


def test_balanced_force_formula():
    pred = torch.zeros(1, 2, 4); target = torch.tensor([[[1.,2.,3.,4.],[1.,2.,3.,4.]]])
    out = M.balanced_force_objective(pred, target)
    channel = torch.tensor([1.,4.,9.,16.])
    assert torch.equal(out["channel_mse"], channel)
    assert out["balanced"] == pytest.approx(0.5 * channel.mean() + 0.5 * 16)


def test_one_window_performs_exactly_one_optimizer_step():
    class CountingAdamW(torch.optim.AdamW):
        def __init__(self, parameters):
            super().__init__(parameters, lr=1e-5, weight_decay=1e-4)
            self.calls = 0

        def step(self, closure=None):
            self.calls += 1
            return super().step(closure)

    b = batch()
    flow, aero = Tiny(0.02), Tiny(0.3)
    flow.scale.requires_grad_(False)
    optimizer = CountingAdamW(aero.parameters())
    result = M.optimizer_window_step(flow, aero, b, predict, optimizer)
    assert optimizer.calls == 1
    assert result["optimizer_steps"] == 1
    assert torch.isfinite(torch.tensor(result["total"]))


def test_wrong_checkpoint_pair_rejected(tmp_path):
    (tmp_path / "FNO.0.0.mdlus").write_bytes(b"wrong")
    (tmp_path / "checkpoint.0.0.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA"):
        M.checkpoint_pair(tmp_path)


def test_source_has_probe_without_optimizer_step_or_save():
    text = (ROOT / "scripts/train_fcp013_independent_force_fno.py").read_text()
    probe = text.split("if args.resource_probe:", 1)[1].split("optimizer =", 1)[0]
    assert "optimizer.step" not in probe
    assert "save_checkpoint" not in probe
    assert '"optimizer_steps": 0' in probe
