import copy
from pathlib import Path

import pytest
import torch

import p026_history_objective as history
import p026_state_history as mapping


@pytest.fixture
def original():
    return history.load_original_objective(
        Path(
            "/workspace/fluid_control/artifacts/fcp013_training_source_1634c05_immutable/scripts/train_fcp013_independent_force_fno.py"
        )
    )


def data(k=1, batch=1):
    torch.manual_seed(53)
    return (
        torch.randn(batch, 100, 3, 2, 2),
        torch.randn(batch, 100, 3, 2, 2),
        torch.ones(batch, 1, 2, 2),
        torch.randn(batch, 101, 1),
        torch.randn(batch, 100, 4),
        torch.randn(batch, k - 1, 3, 2, 2),
        torch.randn(batch, k - 1, 1),
    )


def predict(model, inputs, mask):
    output = model(inputs)
    return output[:, :3], output[:, 3:].mean((-2, -1))


@pytest.mark.parametrize("batch", [1, 2])
def test_k1_exact_p013_outputs_loss_and_gradient(original, batch):
    args = data(batch=batch)
    a = torch.nn.Conv2d(6, 7, 1)
    b = copy.deepcopy(a)
    capture = []

    def old_predict(*items):
        result = predict(*items)
        capture.append(result[1].detach())
        return result

    reference = original.chunk_force_objective(a, *args[:5], old_predict, backward=True)
    actual = history.chunk_force_objective(b, *args, predict, original, backward=True)
    assert {
        k: v for k, v in actual.items() if k != "normalized_predictions"
    } == reference
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    for domain, offset in [("h1", 0), ("ar", batch * 10)]:
        expected = torch.cat(
            [x[offset : offset + batch * 10].reshape(batch, 10, 4) for x in capture],
            dim=1,
        )
        torch.testing.assert_close(
            actual["normalized_predictions"][domain], expected, rtol=0, atol=0
        )


def test_k4_order_padding_final_transition():
    states = torch.arange(100, dtype=torch.float32)[None, :, None, None, None].expand(
        1, 100, 3, 1, 1
    )
    mask = torch.ones(1, 1, 1, 1)
    omega = torch.arange(101, dtype=torch.float32).reshape(1, 101, 1)
    prefix = torch.zeros(1, 3, 3, 1, 1)
    actions = torch.zeros(1, 3, 1)
    inputs, _ = history.chunk_inputs(states, prefix, mask, omega, actions, 0, 10)
    assert inputs[0, :12, 0, 0].tolist() == [0.0] * 12
    assert inputs[1, :12, 0, 0].tolist() == [0.0] * 9 + [1.0] * 3
    assert (
        inputs[3, :12, 0, 0].tolist() == [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [3.0] * 3
    )
    final, _ = history.chunk_inputs(states, prefix, mask, omega, actions, 90, 100)
    assert (
        final[-1, :12, 0, 0].tolist()
        == [96.0] * 3 + [97.0] * 3 + [98.0] * 3 + [99.0] * 3
    )
    assert final[-1, 13:, 0, 0].tolist() == [96.0, 97.0, 98.0, 99.0, 100.0]


def test_ar_never_uses_true_future_or_aerodynamic_delta(original):
    args = list(data(4))
    model = torch.nn.Conv2d(18, 7, 1)
    first = history.chunk_force_objective(
        model, *args, predict, original, backward=False
    )
    args[1] = args[1] + 100
    args[4] = args[4] - 100

    def changed_delta(*a):
        _, forces = predict(*a)
        return torch.full((20, 3, 2, 2), float("nan")), forces

    second = history.chunk_force_objective(
        model, *args, changed_delta, original, backward=False
    )
    torch.testing.assert_close(
        first["normalized_predictions"]["ar"],
        second["normalized_predictions"]["ar"],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        first["normalized_predictions"]["h1"], second["normalized_predictions"]["h1"]
    )


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = torch.nn.Conv2d(20, 24, 1)
        self.decoder = torch.nn.Conv2d(24, 7, 1)

    def forward(self, x):
        coords = torch.zeros(x.shape[0], 2, *x.shape[-2:])
        return self.decoder(self.lift(torch.cat((x, coords), 1)).tanh())


def test_k4_288_history_gradients_and_chunk_vs_monolithic(original):
    args = data(4)
    model = Toy()
    full = copy.deepcopy(model)
    actual = history.chunk_force_objective(
        model, *args, predict, original, backward=True
    )
    flow, h1, mask, omega, target, pre, actions = args
    hi, m = history.chunk_inputs(h1, pre, mask, omega, actions, 0, 100)
    ar, am = history.chunk_inputs(flow, pre, mask, omega, actions, 0, 100)
    _, forces = predict(full, torch.cat((hi, ar)), torch.cat((m, am)))
    hl = original.balanced_force_objective(forces[:100, None], target[0, :, None])
    al = original.balanced_force_objective(forces[100:, None], target[0, :, None])
    loss = 0.5 * (hl["balanced"] + al["balanced"])
    loss.backward()
    assert abs(actual["total"] - float(loss.detach())) < 1e-6
    for p, q in zip(model.parameters(), full.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
    gradient = torch.cat(
        (model.lift.weight.grad[:, :9], model.lift.weight.grad[:, 13:16]), 1
    )
    assert (
        gradient.numel() == 288
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient) == 288
    )


def test_target_last_index_and_no_trainable_history(original):
    args = list(data())
    args[4].zero_()
    args[4][:, -1, 3] = 100
    model = torch.nn.Conv2d(6, 7, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    result = history.chunk_force_objective(
        model, *args, predict, original, backward=False
    )
    # 100^2/100 * rearCl weight (.5+.5/4) = 62.5, both domains same.
    assert result["total"] == 62.5
    args[0].requires_grad_(True)
    with pytest.raises(ValueError, match="gradients"):
        history.chunk_force_objective(model, *args, predict, original, backward=True)


def test_nonfinite_rejected(original):
    args = list(data(4))
    args[5][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        history.chunk_force_objective(
            torch.nn.Conv2d(18, 7, 1), *args, predict, original, backward=False
        )
