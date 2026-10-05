"""Synthetic official-FNO CPU verification; Root approval required to execute.

This is an engineering fixture, not CFD data or scientific accuracy evidence.
Run only in the externally verified pinned official image with no GPU access.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from physicsnemo.models.fno import FNO

import p026_state_history as history

OFFICIAL_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
REQUIRED_IMAGE = (
    "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only execution required")
    if sha(history.__file__) != args.adapter_sha256:
        raise ValueError("reviewed adapter SHA differs")
    official_source = inspect.getfile(FNO)
    if sha(official_source) != OFFICIAL_SHA:
        raise ValueError("pinned official FNO source differs")
    torch.set_num_threads(2)
    torch.manual_seed(41)
    options = dict(
        out_channels=7,
        dimension=2,
        latent_channels=48,
        num_fno_layers=5,
        num_fno_modes=2,
        padding=0,
        decoder_layers=2,
        decoder_layer_size=8,
        coord_features=True,
    )
    old = FNO(in_channels=6, **options)
    k4 = FNO(in_channels=18, **options)
    k1 = FNO(in_channels=6, **options)
    k4.load_state_dict(
        history.history_warmstart_state(old.state_dict(), k4.state_dict(), 4),
        strict=True,
    )
    k1.load_state_dict(
        history.history_warmstart_state(old.state_dict(), k1.state_dict(), 1),
        strict=True,
    )
    for model in (old, k1, k4):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    old_digest = state_digest(old)
    states = torch.randn(4, 3, 4, 4) * 0.1
    mask = torch.ones(1, 4, 4)
    actions = torch.tensor([[-0.3], [-0.2], [-0.1], [0.0]])
    selected = torch.tensor(0.1)
    x1 = history.build_input(states[-1:], mask, actions[-1:], selected)[None]
    x4 = history.build_input(states, mask, actions, selected)[None]
    with torch.no_grad():
        reference, y1, y4 = old(x1), k1(x1), k4(x4)
    torch.testing.assert_close(y1, reference, rtol=0, atol=0)
    torch.testing.assert_close(y4, reference, rtol=1e-5, atol=1e-6)
    zero_mapping = dict(
        k1_max_abs=float((y1 - reference).abs().max()),
        k4_max_abs=float((y4 - reference).abs().max()),
        output_rtol=1e-5,
        output_atol=1e-6,
    )

    # Synthetic nonzero history columns, not a learned model or checkpoint.
    weight = dict(k4.named_parameters())[history.LIFT]
    with torch.no_grad():
        weight[:, :9] = torch.linspace(-0.02, 0.03, 24 * 9).reshape(24, 9, 1, 1)
        weight[:, 13:16] = 0.01
    nonzero_digest = state_digest(k4)
    observed_history = states.clone().requires_grad_(True)
    first = k4(history.build_input(observed_history, mask, actions, selected)[None])
    history_gradient = torch.autograd.grad(
        first.square().mean(), observed_history, retain_graph=True
    )[0]
    if (
        not torch.isfinite(history_gradient).all()
        or history_gradient[:3].norm().item() == 0
    ):
        raise RuntimeError("historical-state sensitivity absent/nonfinite")

    predicted_next_state = first[0, :3]
    shifted_states, shifted_actions = history.shift_history(
        observed_history, actions, predicted_next_state, selected
    )
    second = k4(history.build_input(shifted_states, mask, shifted_actions, 0.2)[None])
    predicted_state_gradient, initial_history_gradient = torch.autograd.grad(
        second.square().mean(), (predicted_next_state, observed_history)
    )
    for gradient in (predicted_state_gradient, initial_history_gradient):
        if not torch.isfinite(gradient).all() or gradient.norm().item() == 0:
            raise RuntimeError("AR-shift gradient path absent/nonfinite")
    if (
        state_digest(old) != old_digest
        or state_digest(k1) != old_digest
        or state_digest(k4) != nonzero_digest
    ):
        raise RuntimeError("CPU fixture forward mutated model tensors")
    if any(
        parameter.grad is not None
        for model in (old, k1, k4)
        for parameter in model.parameters()
    ):
        raise RuntimeError("frozen model accumulated parameter gradients")
    print(
        json.dumps(
            dict(
                status="FC_P026_SYNTHETIC_OFFICIAL_CPU_ENGINEERING_NOT_ADMISSION",
                required_official_image_id=REQUIRED_IMAGE,
                image_identity_note="required identity only; actual container image must be independently recorded",
                official_source=dict(path=official_source, sha256=sha(official_source)),
                adapter_sha256=sha(history.__file__),
                script_sha256=sha(__file__),
                cuda_available=False,
                synthetic_fixture=True,
                scientific_admission=False,
                zero_history_mapping=zero_mapping,
                mapping=dict(
                    current_state=[9, 12],
                    mask=12,
                    current_next_actions=[16, 18],
                    coordinates=[18, 20],
                ),
                historical_state_gradient_norm=float(history_gradient[:3].norm()),
                predicted_state_gradient_norm=float(predicted_state_gradient.norm()),
                initial_history_gradient_norm=float(initial_history_gradient.norm()),
                all_gradients_finite=True,
                all_model_tensors_unchanged_during_forwards=True,
                parameter_gradients_absent=True,
                optimizer_created=False,
                candidate_saved=False,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
