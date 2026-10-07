import torch
from physicsnemo.models.fno import FNO
from pathlib import Path
import p026_history_objective as history

import p064_total_pressure_aux as m


def tiny(out_channels):
    return FNO(in_channels=6, out_channels=out_channels, dimension=2,
               latent_channels=4, num_fno_layers=2, num_fno_modes=[2, 2],
               decoder_layers=2, decoder_layer_size=8, padding=1,
               coord_features=True).float()


def test_out11_preserves_old7_and_standardized_copy():
    torch.manual_seed(7)
    old = tiny(7)
    with torch.random.fork_rng(devices=[]):
        new = tiny(11)
    m.initialize_out11_from_out7(old, new)
    x = torch.randn(2, 6, 8, 10)
    mask = torch.ones(2, 1, 8, 10); mask[:, :, :2] = 0
    y7, y11 = old(x), new(x)
    torch.testing.assert_close(y11[:, :7], y7)
    torch.testing.assert_close(y11[:, 7:11], y7[:, 3:7])
    _, total, pressure = m.extract_total_pressure(y11, mask)
    torch.testing.assert_close(total, pressure)


def test_aux_zero_old_gradient_and_fixed_lambda_gradient():
    torch.manual_seed(11)
    old, new = tiny(7), tiny(11)
    m.initialize_out11_from_out7(old, new)
    x = torch.randn(2, 6, 8, 10)
    mask = torch.ones(2, 1, 8, 10)
    total_target = torch.randn(2, 4)
    pressure_physical = m.PRESSURE_MEAN + m.PRESSURE_STD * torch.randn(2, 4)
    raw7, raw11 = old(x), new(x)
    total7 = (raw7[:, 3:7] * mask).mean((-2, -1))
    _, total11, pressure11 = m.extract_total_pressure(raw11, mask)
    loss7 = (total7 - total_target).square().mean()
    loss11 = (total11 - total_target).square().mean()
    loss7.backward(); loss11.backward(retain_graph=True)
    oldp, newp = dict(old.named_parameters()), dict(new.named_parameters())
    for name in oldp:
        if name in (m.FINAL_WEIGHT, m.FINAL_BIAS):
            torch.testing.assert_close(newp[name].grad[:7], oldp[name].grad)
            assert torch.count_nonzero(newp[name].grad[7:]) == 0
        else:
            torch.testing.assert_close(newp[name].grad, oldp[name].grad)
    new.zero_grad(set_to_none=True)
    aux = m.pressure_h1_mse(pressure11, pressure_physical)
    objective = loss11 + m.AUX_WEIGHT * aux
    objective.backward()
    assert torch.isfinite(objective) and m.AUX_WEIGHT == 0.1
    assert torch.count_nonzero(newp[m.FINAL_WEIGHT].grad[7:]) > 0
    trainable = [p for p in new.parameters() if p.grad is not None]
    preclip = torch.linalg.vector_norm(torch.stack([p.grad.double().norm() for p in trainable]))
    assert torch.isfinite(preclip) and preclip > 0


def test_pressure_normalization_mean_and_std_contract():
    values = torch.stack((m.PRESSURE_MEAN, m.PRESSURE_MEAN + m.PRESSURE_STD))
    normalized = m.normalize_pressure(values)
    torch.testing.assert_close(normalized[0], torch.zeros(4))
    torch.testing.assert_close(normalized[1], torch.ones(4))


def test_actual_sidecar_identity_and_h100_time_contract():
    root = Path("/workspace/fluid_control/artifacts/p064_force_component_sidecars_20261007")
    lookup = m.PressureSidecars(root, root / "manifest.json")
    assert len(lookup.data) == 45
    assert "b00_projected_ppo_train" in lookup.data
    cases = [
        "matched_start_acquisition_train_b00_m0375",
        "dynamic_train8_b00_multisine",
        "direct_cfd_directppo2048_v1_env0_ep0002_b00",
        "b00_projected_ppo_train",
    ]
    for dataset_index, case in enumerate(cases):
        row = lookup.data[case]
        target = lookup.target(
            {"case": case, "start": 0, "dataset_index": dataset_index},
            row["time"][0],
        )
        assert torch.equal(target, row["pressure"][1:101])


def test_ten_chunk_objective_keeps_old_total_and_adds_h1_pressure():
    class Objective:
        @staticmethod
        def balanced_force_objective(prediction, target):
            channel = (prediction[:, 0] - target[:, 0]).square().mean(0)
            return {"channel_mse": channel,
                    "balanced": 0.5 * channel.mean() + 0.5 * channel[3]}
    torch.manual_seed(19)
    model = torch.nn.Conv2d(6, 11, 1)
    states = torch.randn(1, 100, 3, 4, 5)
    h1 = torch.randn_like(states)
    mask = torch.ones(1, 1, 4, 5); mask[:, :, 0, 0] = 0
    omega = torch.randn(1, 101, 1)
    force = torch.randn(1, 100, 4)
    pressure = m.PRESSURE_MEAN + m.PRESSURE_STD * torch.randn(1, 100, 4)
    pre = torch.empty(1, 0, 3, 4, 5); actions = torch.empty(1, 0, 1)
    calls = []
    def forward(net, inputs):
        calls.append(inputs.shape)
        return net(inputs)
    result = m.chunk_total_pressure_objective(
        model, states, h1, mask, omega, force, pressure, pre, actions,
        forward, history, Objective, backward=True)
    assert len(calls) == 10 and all(shape == (20, 6, 4, 5) for shape in calls)
    assert abs(result["training_objective"] -
               result["original_total"] - m.AUX_WEIGHT * result["pressure_h1"]) < 1e-6
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
