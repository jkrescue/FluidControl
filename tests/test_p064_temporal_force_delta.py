from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p064_temporal_force_delta as residual


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 7)


class _Final(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = _Linear()


class TinyOfficialShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_net = _Final()

    def forward(self, value):
        return self.decoder_net.final_layer.linear(value)


def _make_inputs(states, mask, now, nxt):
    return torch.cat((states[:, :, 0, 0], mask[:, :, 0, 0], now, nxt), dim=1)


def _predict(model, inputs, mask):
    output = model(inputs)
    return output[:, :3], output[:, 3:]


def _objective(prediction, target):
    mse = (prediction - target).square().mean()
    return {"balanced": mse}


def _fixture(model, batch=1):
    torch.manual_seed(4)
    flow = torch.randn(batch, 100, 2, 1, 1)
    h1 = torch.randn(batch, 100, 2, 1, 1)
    mask = torch.ones(batch, 1, 1, 1)
    omega = torch.linspace(-0.4, 0.6, 101).reshape(1, 101, 1).expand(batch, -1, -1).clone()
    target = torch.randn(batch, 100, 4)
    initial = torch.randn(batch, 4)
    return dict(
        aerodynamic_model=model,
        flow_states=flow,
        h1_states=h1,
        mask=mask,
        omega=omega,
        target_force=target,
        initial_force=initial,
        make_inputs=_make_inputs,
        predict_fn=_predict,
        force_objective=_objective,
    )


def test_zero_rows_preserve_field_and_produce_exact_persistence():
    model = TinyOfficialShape()
    before_w = model.decoder_net.final_layer.linear.weight[:3].detach().clone()
    before_b = model.decoder_net.final_layer.linear.bias[:3].detach().clone()
    receipt = residual.zero_force_output_rows(model)
    assert receipt["zero_force_rows"] == [3, 4, 5, 6]
    assert torch.equal(before_w, model.decoder_net.final_layer.linear.weight[:3])
    assert torch.equal(before_b, model.decoder_net.final_layer.linear.bias[:3])
    args = _fixture(model)
    out = residual.temporal_residual_objective(**args, backward=False)
    current = residual.causal_current_force(args["target_force"], args["initial_force"])
    assert torch.equal(out["h1"], current)
    assert torch.equal(
        out["ar"], args["initial_force"][:, None].expand(-1, 100, -1)
    )


def test_truth_timing_has_no_future_leak_and_ar_never_reads_targets():
    model = TinyOfficialShape()
    residual.zero_force_output_rows(model)
    args = _fixture(model)
    first = residual.temporal_residual_objective(**args, backward=False)
    changed = copy.copy(args)
    changed["target_force"] = args["target_force"].clone()
    changed["target_force"][:, 70] += 99
    second = residual.temporal_residual_objective(**changed, backward=False)
    assert torch.equal(first["h1"][:, :71], second["h1"][:, :71])
    assert not torch.equal(first["h1"][:, 71], second["h1"][:, 71])
    assert torch.equal(first["ar"], second["ar"])


def test_zero_head_hidden_gradient_is_zero_but_force_rows_learn():
    model = TinyOfficialShape()
    residual.zero_force_output_rows(model)
    args = _fixture(model)
    residual.temporal_residual_objective(**args, backward=True)
    weight = model.decoder_net.final_layer.linear.weight
    bias = model.decoder_net.final_layer.linear.bias
    assert torch.count_nonzero(weight.grad[:3]) == 0
    assert torch.count_nonzero(bias.grad[:3]) == 0
    assert torch.count_nonzero(weight.grad[3:]) > 0
    assert torch.count_nonzero(bias.grad[3:]) > 0


def test_full_h100_autograd_matches_explicit_cumulative_recurrence():
    first = TinyOfficialShape()
    second = copy.deepcopy(first)
    args = _fixture(first)
    out = residual.temporal_residual_objective(**args, backward=True)
    observed = first.decoder_net.final_layer.linear.weight.grad.detach().clone()

    # Independent full-graph expression: each delta contributes to every later AR loss.
    manual_args = _fixture(second)
    h1_delta, ar_delta = [], []
    for step in range(100):
        states = torch.cat((manual_args["h1_states"][:, step], manual_args["flow_states"][:, step]))
        masks = torch.cat((manual_args["mask"], manual_args["mask"]))
        now = torch.cat((manual_args["omega"][:, step], manual_args["omega"][:, step]))
        nxt = torch.cat((manual_args["omega"][:, step + 1], manual_args["omega"][:, step + 1]))
        _, delta = _predict(second, _make_inputs(states, masks, now, nxt), masks)
        h1_delta.append(delta[:1])
        ar_delta.append(delta[1:])
    h1_delta = torch.stack(h1_delta, 1)
    ar_delta = torch.stack(ar_delta, 1)
    current = residual.causal_current_force(manual_args["target_force"], manual_args["initial_force"])
    h1 = current + h1_delta
    ar = manual_args["initial_force"][:, None] + torch.cumsum(ar_delta, dim=1)
    loss = 0.5 * _objective(h1, manual_args["target_force"])["balanced"] + 0.5 * _objective(ar, manual_args["target_force"])["balanced"]
    loss.backward()
    expected = second.decoder_net.final_layer.linear.weight.grad
    assert torch.allclose(out["total"], loss.detach(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(observed, expected, rtol=2e-5, atol=2e-6)


def test_chunk10_exact_recompute_vjp_matches_full_h100_graph():
    full = TinyOfficialShape()
    replay = copy.deepcopy(full)
    full_args = _fixture(full)
    replay_args = _fixture(replay)
    full_out = residual.temporal_residual_objective(**full_args, backward=True)
    replay_out = residual.exact_recompute_vjp_objective(
        **replay_args, chunk_size=10, backward=True
    )
    assert replay_out["gradient_method"].startswith("exact_full_h100")
    assert replay_out["maximum_model_batch"] == 20
    assert replay_out["force_recurrence_detaches"] == 0
    assert torch.allclose(full_out["total"], replay_out["total"], rtol=1e-6, atol=1e-7)
    for (full_name, full_parameter), (replay_name, replay_parameter) in zip(
        full.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert full_name == replay_name
        assert torch.allclose(
            full_parameter.grad, replay_parameter.grad, rtol=2e-5, atol=2e-6
        ), full_name


def test_recompute_vjp_multiple_batches_and_chunk_sizes_match_oracle():
    parent = TinyOfficialShape()
    for chunk_size in (1, 10, 20):
        full = copy.deepcopy(parent).train()
        replay = copy.deepcopy(parent).train()
        full_args = _fixture(full, batch=3)
        replay_args = _fixture(replay, batch=3)
        full_out = residual.temporal_residual_objective(**full_args, backward=True)
        replay_out = residual.exact_recompute_vjp_objective(
            **replay_args, chunk_size=chunk_size, backward=True
        )
        assert replay_out["maximum_model_batch"] == 2 * 3 * chunk_size
        assert torch.equal(full_out["h1"], replay_out["h1"])
        assert torch.allclose(full_out["ar"], replay_out["ar"], rtol=2e-6, atol=2e-6)
        assert torch.allclose(full_out["total"], replay_out["total"], rtol=1e-6, atol=1e-7)
        for left, right in zip(full.parameters(), replay.parameters(), strict=True):
            assert torch.allclose(left.grad, right.grad, rtol=3e-5, atol=3e-6)


def test_recompute_rejects_eval_mode_because_mode_must_be_frozen():
    model = TinyOfficialShape().eval()
    with torch.no_grad():
        pass
    import pytest
    with pytest.raises(ValueError, match="training-mode"):
        residual.exact_recompute_vjp_objective(**_fixture(model), chunk_size=10, backward=True)


def test_one_adam_step_changes_only_force_rows_from_zero_head():
    model = TinyOfficialShape()
    residual.zero_force_output_rows(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    args = _fixture(model)
    residual.temporal_residual_objective(**args, backward=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    after = dict(model.named_parameters())
    assert torch.equal(before[residual.FINAL_WEIGHT][:3], after[residual.FINAL_WEIGHT][:3])
    assert torch.equal(before[residual.FINAL_BIAS][:3], after[residual.FINAL_BIAS][:3])
    assert not torch.equal(before[residual.FINAL_WEIGHT][3:], after[residual.FINAL_WEIGHT][3:])
    assert not torch.equal(before[residual.FINAL_BIAS][3:], after[residual.FINAL_BIAS][3:])
