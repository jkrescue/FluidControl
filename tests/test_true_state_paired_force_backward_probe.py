from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

from fluid_control import paired_step_force

dynamic = types.ModuleType("fluid_control.dynamic_pair_stat_datapipe")
dynamic.DynamicMatchedPairStatDataset = object
datapipes = types.ModuleType("physicsnemo.datapipes")
datapipes.DataLoader = object
distributed = types.ModuleType("physicsnemo.distributed")
distributed.DistributedManager = object
utils = types.ModuleType("physicsnemo.utils")
utils.load_checkpoint = object
physicsnemo = types.ModuleType("physicsnemo")
trainer = types.ModuleType("train_tandem_fno")
trainer.build_model = trainer.configured_force_indices = trainer.predict = object
STUBS = {
    dynamic.__name__: dynamic,
    "physicsnemo": physicsnemo,
    "physicsnemo.datapipes": datapipes,
    "physicsnemo.distributed": distributed,
    "physicsnemo.utils": utils,
    "train_tandem_fno": trainer,
    "fluid_control.paired_step_force": paired_step_force,
}


SCRIPT = Path(__file__).parents[1] / "scripts/probe_true_state_paired_force_backward.py"
SPEC = importlib.util.spec_from_file_location("true_state_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
with mock.patch.dict(sys.modules, STUBS):
    SPEC.loader.exec_module(MODULE)


def test_chunk_scaling_matches_monolithic_loss_and_gradient() -> None:
    torch.manual_seed(7)
    feature = torch.randn(1, 20, 4)
    target_a = torch.randn(1, 20, 4)
    target_z = torch.randn(1, 20, 4)
    weights = torch.tensor(MODULE.WEIGHTS)

    full_parameter = torch.nn.Parameter(torch.randn(4, 4))
    full_prediction = feature @ full_parameter
    full_loss = MODULE.scaled_loss(
        full_prediction,
        torch.zeros_like(full_prediction),
        target_a,
        target_z,
        weights,
        20,
    )
    full_loss.backward()

    chunk_parameter = torch.nn.Parameter(full_parameter.detach().clone())
    chunk_loss = torch.zeros(())
    for start in (0, 10):
        prediction = feature[:, start : start + 10] @ chunk_parameter
        loss = MODULE.scaled_loss(
            prediction,
            torch.zeros_like(prediction),
            target_a[:, start : start + 10],
            target_z[:, start : start + 10],
            weights,
            20,
        )
        loss.backward()
        chunk_loss += loss.detach()

    assert torch.allclose(chunk_loss, full_loss.detach(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        chunk_parameter.grad, full_parameter.grad, rtol=1e-6, atol=1e-7
    )


def test_scaled_loss_rejects_invalid_temporal_contract() -> None:
    values = torch.zeros(1, 2, 4)
    weights = torch.tensor(MODULE.WEIGHTS)
    with pytest.raises(ValueError, match="invalid chunk"):
        MODULE.scaled_loss(values, values, values, values, weights, 1)


def test_probe_source_has_no_optimizer_or_checkpoint_save() -> None:
    source = SCRIPT.read_text()
    assert "torch.optim" not in source
    assert "save_checkpoint" not in source
    assert "optimizer.step" not in source
    assert '"optimizer_constructed": False' in source
    assert '"candidate_weights_saved": False' in source
    assert '"validation_or_frozen_accessed": False' in source
