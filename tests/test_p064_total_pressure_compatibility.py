"""Synthetic CPU-only parameterization fixture; not CFD/model-quality evidence.

Prespecified float32 tolerances: forward/gradient atol=2e-6, rtol=2e-5;
AdamW old-parameter update atol=2e-7, rtol=2e-5. Official spectral weights
internally use complex64. No checkpoint, real dataset, or candidate is loaded.
"""
import hashlib
import inspect
import json
from pathlib import Path

import torch
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected

FROZEN = {"spec_encoder.lift_network.0.conv.bias",
          "spec_encoder.lift_network.2.conv.bias"}
WEIGHT = "decoder_net.final_layer.linear.weight"
BIAS = "decoder_net.final_layer.linear.bias"


def test_official_tiny_total_pressure_aux_zero_compatibility():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), "CUDA must be hidden"
    for cls, expected in (
        (FNO, "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"),
        (FullyConnected, "2a5abce0334b04c0eeecc5c8757f238499348a07bf2ff8351f2afdda30abcf82"),
    ):
        assert hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() == expected
    torch.manual_seed(20261007)
    cfg = dict(in_channels=6, dimension=2, latent_channels=4,
               num_fno_layers=2, num_fno_modes=[2, 2], decoder_layers=2,
               decoder_layer_size=8, padding=1, coord_features=True)
    old = FNO(out_channels=7, **cfg).float()
    rng_before = torch.get_rng_state().clone()
    # New-head construction must not perturb subsequent synthetic data RNG.
    with torch.random.fork_rng(devices=[]):
        new = FNO(out_channels=11, **cfg).float()
    assert torch.equal(rng_before, torch.get_rng_state())
    old_params, new_params = dict(old.named_parameters()), dict(new.named_parameters())
    assert old_params.keys() == new_params.keys()
    assert FROZEN <= old_params.keys()
    with torch.no_grad():
        for name, value in old_params.items():
            if name in (WEIGHT, BIAS):
                new_params[name][:7].copy_(value)
                new_params[name][7:].copy_(value[3:7] * 0.5)
            else:
                new_params[name].copy_(value)
    for model in (old, new):
        for name, param in model.named_parameters():
            param.requires_grad_(name not in FROZEN)
    before = {n: p.detach().clone() for n, p in old_params.items()}
    x = torch.randn(2, 6, 8, 10, dtype=torch.float32)
    target = torch.randn(2, 4, dtype=torch.float32)
    mask = torch.ones(2, 1, 8, 10, dtype=torch.float32)
    mask[:, :, :2, :2] = 0
    data_rng = torch.get_rng_state().clone()
    def old_total_loss(raw):
        # Aux=0 explicitly reads only the old total channels, not pressure.
        force = (raw[:, 3:7] * mask).sum((-2, -1)) / mask.sum((-2, -1))
        mse = (force - target).square().mean(0)
        return 0.5 * mse.mean() + 0.5 * mse[3]
    y_old, y_new = old(x), new(x)
    torch.testing.assert_close(y_new[:, :7], y_old, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(y_new[:, 7:], y_old[:, 3:7] * 0.5, atol=2e-6, rtol=2e-5)
    loss_old, loss_new = old_total_loss(y_old), old_total_loss(y_new)
    loss_old.backward(); loss_new.backward()
    max_grad_difference = 0.0
    for name, param in old_params.items():
        other = new_params[name]
        if name in FROZEN:
            assert param.grad is None and other.grad is None
            continue
        grad = other.grad[:7] if name in (WEIGHT, BIAS) else other.grad
        torch.testing.assert_close(grad, param.grad, atol=2e-6, rtol=2e-5)
        max_grad_difference = max(max_grad_difference, float((grad-param.grad).abs().max()))
    assert torch.count_nonzero(new_params[WEIGHT].grad[7:]) == 0
    assert torch.count_nonzero(new_params[BIAS].grad[7:]) == 0
    old_norm = torch.nn.utils.clip_grad_norm_([p for p in old.parameters() if p.requires_grad], 1.0)
    new_norm = torch.nn.utils.clip_grad_norm_([p for p in new.parameters() if p.requires_grad], 1.0)
    torch.testing.assert_close(old_norm, new_norm, atol=2e-6, rtol=2e-5)
    for model in (old, new):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1.5625e-7, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=1e-4)
        optimizer.step()
    max_update_difference = 0.0
    updated_old_parameters = 0
    for name, param in old_params.items():
        other = new_params[name][:7] if name in (WEIGHT, BIAS) else new_params[name]
        delta_old, delta_new = param.detach()-before[name], other.detach()-before[name]
        if name not in FROZEN and torch.count_nonzero(delta_old):
            updated_old_parameters += 1
        torch.testing.assert_close(delta_new, delta_old, atol=2e-7, rtol=2e-5)
        max_update_difference = max(max_update_difference, float((delta_new-delta_old).abs().max()))
        if name in FROZEN:
            assert torch.equal(param, before[name]) and torch.equal(other, before[name])
    assert updated_old_parameters > 0, "old optimizer step must perform a real update"
    assert torch.equal(data_rng, torch.get_rng_state()), "forward/backward/Adam changed data RNG"
    print(json.dumps({"fixture": "synthetic_tiny_official_FNO_aux_zero",
        "old_parameter_tensors": len(old_params), "max_forward_abs": float((y_new[:, :7]-y_old).detach().abs().max()),
        "max_gradient_abs": max_grad_difference, "max_update_abs": max_update_difference,
        "old_loss": float(loss_old.detach()), "new_loss": float(loss_new.detach()),
        "updated_old_parameter_tensors": updated_old_parameters,
        "rng_unchanged": True, "real_data_loaded": False, "candidate_saved": False}))
