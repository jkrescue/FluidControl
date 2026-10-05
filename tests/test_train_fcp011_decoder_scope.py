from __future__ import annotations

import copy

import pytest
import torch

from train_fcp011_decoder_scope import (
    FINAL_BIAS,
    FINAL_WEIGHT,
    HIDDEN_BIAS,
    HIDDEN_WEIGHT,
    assert_scope_confinement,
    balanced_decoder_objective,
    capture_frozen_final_rows,
    configure_trainable_scope,
    evaluate_diagnostics,
    mask_and_step,
    metadata_rows,
    run_resource_probe,
    sequence_sha,
    training_identity,
    validate_data_contract,
)


class LinearBox(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([LinearBox(2, 4), LinearBox(4, 4)])
        self.final_layer = LinearBox(4, 7)


class TinyFNO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder_net = Decoder()
        self.register_buffer("fixed_buffer", torch.arange(3.0))

    def forward(self, value):
        hidden = self.decoder_net.layers[1].linear(value)
        return self.decoder_net.final_layer.linear(hidden)


def state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_balanced_loss_exact_arithmetic_and_channel_shares():
    predicted_state = torch.ones(1, 2, 3, 1, 1)
    target_state = torch.zeros_like(predicted_state)
    predicted_force = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
    target_force = torch.zeros_like(predicted_force)
    mask = torch.ones(1, 1, 1, 1)
    result = balanced_decoder_objective(
        predicted_state, predicted_force, target_state, target_force, mask
    )
    channel = torch.tensor([1.0, 4.0, 9.0, 16.0])
    assert torch.equal(result["force_channel_mse"], channel)
    assert result["field"].item() == pytest.approx(1.0)
    assert result["equal_four_force"].item() == pytest.approx(7.5)
    assert result["rear_cl"].item() == pytest.approx(16.0)
    assert result["balanced_force"].item() == pytest.approx(11.75)
    assert result["total"].item() == pytest.approx(3.35)


@pytest.mark.parametrize(
    ("scope", "expected", "effective"),
    [
        ("head_only", {FINAL_WEIGHT, FINAL_BIAS}, 129),
        (
            "decoder_tail",
            {FINAL_WEIGHT, FINAL_BIAS, HIDDEN_WEIGHT, HIDDEN_BIAS},
            129 + 4 * 4 + 4,
        ),
    ],
)
def test_scope_is_exact(scope, expected, effective):
    model = TinyFNO()
    result = configure_trainable_scope(model, scope)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert actual == expected
    assert result["effective_trainable_coefficients"] == effective


@pytest.mark.parametrize("scope", ["head_only", "decoder_tail"])
def test_masked_adamw_preserves_every_frozen_value(scope):
    torch.manual_seed(7)
    model = TinyFNO()
    configure_trainable_scope(model, scope)
    before = state(model)
    frozen_rows = capture_frozen_final_rows(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
        weight_decay=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(5, 4)).square().mean()
    loss.backward()
    mask_and_step(model, optimizer)
    after = state(model)
    changed = assert_scope_confinement(before, after, scope)
    assert FINAL_WEIGHT in changed and FINAL_BIAS in changed
    assert torch.equal(model.decoder_net.final_layer.linear.weight[:6], frozen_rows["weight"])
    assert torch.equal(model.decoder_net.final_layer.linear.bias[:6], frozen_rows["bias"])
    assert torch.equal(before["fixed_buffer"], after["fixed_buffer"])
    if scope == "head_only":
        assert HIDDEN_WEIGHT not in changed and HIDDEN_BIAS not in changed
    else:
        assert HIDDEN_WEIGHT in changed and HIDDEN_BIAS in changed


def test_confinement_rejects_unapproved_tensor_change():
    model = TinyFNO()
    before = state(model)
    after = copy.deepcopy(before)
    after["encoder.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="outside approved scope"):
        assert_scope_confinement(before, after, "decoder_tail")


def test_metadata_normalization_and_identity_are_fail_closed():
    row = {
        "case": "train_b00",
        "start": 3,
        "dataset_index": 1,
        "split": "train",
        "rollout_steps": 100,
    }
    assert metadata_rows([row]) == [row]
    assert training_identity({key: [value] for key, value in row.items()}) == row
    with pytest.raises(ValueError):
        training_identity({"case": ["train_b00"], "start": [-1]})


@pytest.mark.parametrize(
    "change",
    [
        {"split": "validation"},
        {"rollout_steps": 99},
        {"dataset_index": 3},
    ],
)
def test_training_identity_rejects_wrong_split_rollout_or_family(change):
    row = {
        "case": "train_b00",
        "start": 3,
        "dataset_index": 1,
        "split": "train",
        "rollout_steps": 100,
        **change,
    }
    with pytest.raises(ValueError):
        training_identity([row])


def test_sequence_sha_is_order_sensitive():
    assert sequence_sha([0, 1, 2]) != sequence_sha([0, 2, 1])


def test_data_contract_rejects_wrong_manifest_or_exposed_validation(tmp_path):
    class Cfg:
        pass

    cfg = Cfg()
    cfg.data = Cfg()
    roots = [tmp_path / name for name in ("base", "train8", "train16")]
    for root in roots:
        root.mkdir()
        (root / "manifest.json").write_text("{}")
        (root / "normalization.json").write_text("{}")
    cfg.data.root = str(roots[0])
    cfg.data.additional_train_roots = [str(root) for root in roots[1:]]
    with pytest.raises(ValueError, match="manifest differs"):
        validate_data_contract(cfg)


def test_two_arms_start_identically_and_only_scope_differs():
    torch.manual_seed(11)
    parent = TinyFNO()
    arm_a, arm_b = copy.deepcopy(parent), copy.deepcopy(parent)
    assert all(torch.equal(left, right) for left, right in zip(arm_a.state_dict().values(), arm_b.state_dict().values()))
    configure_trainable_scope(arm_a, "head_only")
    configure_trainable_scope(arm_b, "decoder_tail")
    assert {name for name, p in arm_a.named_parameters() if p.requires_grad} == {FINAL_WEIGHT, FINAL_BIAS}
    assert {name for name, p in arm_b.named_parameters() if p.requires_grad} - {FINAL_WEIGHT, FINAL_BIAS} == {HIDDEN_WEIGHT, HIDDEN_BIAS}


@pytest.mark.parametrize("scope", ["head_only", "decoder_tail"])
def test_resource_probe_has_gradients_but_no_optimizer_step_or_state_change(scope):
    torch.manual_seed(19)
    model = TinyFNO()
    configure_trainable_scope(model, scope)
    steps = 2
    batch = {
        "state": torch.zeros(1, 3, 1, 1),
        "target_state": torch.zeros(1, steps, 3, 1, 1),
        "omega": torch.zeros(1, steps + 1, 1),
        "target_force": torch.zeros(1, steps, 4),
        "mask": torch.ones(1, 1, 1, 1),
    }

    def fake_rollout(network, state, mask, omega, target_state, teacher_forcing):
        del state, mask, omega, teacher_forcing
        raw = network(torch.ones(1, 4))
        predicted_state = raw[:, :3, None, None][:, None].expand_as(target_state)
        predicted_force = raw[:, 3:][:, None].expand(1, steps, 4)
        return predicted_state, predicted_force

    before = state(model)
    result = run_resource_probe(
        model,
        batch,
        {"case": "synthetic_train", "start": 0},
        scope,
        torch.device("cpu"),
        fake_rollout,
    )
    after = state(model)
    assert result["optimizer_created"] is False
    assert result["optimizer_steps"] == 0
    assert result["model_saved"] is False
    assert result["model_tensor_sha256_before"] == result["model_tensor_sha256_after"]
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_true_state_h1_diagnostic_consumes_later_recorded_states():
    model = TinyFNO()
    steps = 100
    sample = {
        "state": torch.zeros(3, 1, 1),
        "target_state": torch.zeros(steps, 3, 1, 1),
        "omega": torch.zeros(steps + 1, 1),
        "target_force": torch.zeros(steps, 4),
        "mask": torch.ones(1, 1, 1),
    }
    item = {
        "global_index": 0,
        "family": "synthetic",
        "identity": {"case": "train", "start": 0},
        "sample": sample,
    }

    def fake_rollout(network, state, mask, omega):
        del network, state, mask, omega
        return torch.zeros(1, steps, 3, 1, 1), torch.zeros(1, steps, 4)

    def fake_predict(network, inputs, masks):
        del network, masks
        signal = inputs[:, 0].mean(dim=(-2, -1), keepdim=False)[:, None]
        return torch.zeros(inputs.shape[0], 3, 1, 1), signal.expand(-1, 4)

    common = (
        model,
        [item],
        torch.device("cpu"),
        torch.zeros(3, 1, 1),
        torch.ones(3, 1, 1),
        torch.zeros(4),
        torch.ones(4),
        fake_rollout,
        fake_predict,
    )
    baseline = evaluate_diagnostics(*common)["windows"][0]
    sample["target_state"][49, 0] = 2.0
    altered = evaluate_diagnostics(*common)["windows"][0]
    assert baseline["true_state_h1_force_mae"] == [0.0] * 4
    assert altered["true_state_h1_force_mae"][3] > 0.0
    assert altered["true_state_h1_field_relative_l2_uvp"][0] != baseline[
        "true_state_h1_field_relative_l2_uvp"
    ][0]
