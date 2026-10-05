from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fcp012", ROOT / "scripts" / "diagnose_fcp012_decoder_gradient_alignment.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Trainer:
    HIDDEN_WEIGHT = "hidden.weight"
    HIDDEN_BIAS = "hidden.bias"
    FINAL_WEIGHT = "final.weight"
    FINAL_BIAS = "final.bias"
    REAR_CL_INDEX = 6


NAMES = (
    Trainer.HIDDEN_WEIGHT,
    Trainer.HIDDEN_BIAS,
    Trainer.FINAL_WEIGHT,
    Trainer.FINAL_BIAS,
)


def values():
    return [
        torch.ones(2, 2),
        torch.ones(2),
        torch.arange(14.0).reshape(7, 2),
        torch.arange(7.0),
    ]


def test_mask_matches_fcp011_final_row_contract():
    masked = MODULE.mask_component_gradients(values(), NAMES, Trainer)
    assert torch.count_nonzero(masked[2][:6]) == 0
    assert torch.equal(masked[2][6], torch.tensor([12.0, 13.0]))
    assert torch.count_nonzero(masked[3][:6]) == 0
    assert masked[3][6] == 6
    assert torch.equal(masked[0], values()[0])


@pytest.mark.parametrize("bad", [None, torch.tensor([float("nan")])])
def test_missing_or_nonfinite_gradient_rejected(bad):
    item = values()
    item[0] = bad
    with pytest.raises((RuntimeError, FloatingPointError)):
        MODULE.mask_component_gradients(item, NAMES, Trainer)


def test_zero_norm_cosine_is_null_not_zero():
    zero = [torch.zeros(3)]
    nonzero = [torch.ones(3)]
    result = MODULE.compare_vectors(zero, nonzero)
    assert result["cosine"] is None
    assert result["cosine_defined"] is False


def test_group_separation_and_component_residual():
    masked = MODULE.mask_component_gradients(values(), NAMES, Trainer)
    assert len(MODULE.select_group(masked, NAMES, "hidden", Trainer)) == 2
    assert len(MODULE.select_group(masked, NAMES, "rear_cl_row", Trainer)) == 2
    assert len(MODULE.select_group(masked, NAMES, "complete", Trainer)) == 4
    total = [2 * value for value in masked]
    summed = [value + value for value in masked]
    result = MODULE.residual_stats(total, summed)
    assert result["absolute_l2"] == 0
    assert result["relative_l2"] == 0
    assert result["interpretation"] == "observational_only_no_predeclared_tolerance"


def test_autograd_decomposition_masks_nonrear_rows_and_keeps_graph_strict():
    hidden_w = torch.nn.Parameter(torch.tensor([[1.0]]))
    hidden_b = torch.nn.Parameter(torch.tensor([0.5]))
    final_w = torch.nn.Parameter(torch.ones(7, 1))
    final_b = torch.nn.Parameter(torch.zeros(7))
    parameters = (hidden_w, hidden_b, final_w, final_b)
    hidden = hidden_w @ torch.tensor([[2.0]]) + hidden_b
    output = final_w @ hidden.flatten() + final_b
    losses = {
        "field": output[:3].square().mean(),
        "balanced_force": output[6].square(),
    }
    losses["total"] = losses["field"] + 0.2 * losses["balanced_force"]
    result = MODULE.decompose(losses, parameters, NAMES, Trainer)
    rear_field = result["rear_cl_row"]["field"]
    assert rear_field["l2"] == 0
    assert result["rear_cl_row"]["field_vs_weighted_force"]["cosine"] is None
    assert result["complete"]["total_direct_vs_component_sum"]["relative_l2"] < 1e-6


def test_source_declares_no_optimizer_or_checkpoint_save():
    text = (ROOT / "scripts" / "diagnose_fcp012_decoder_gradient_alignment.py").read_text()
    assert "torch.optim" not in text
    assert "save_checkpoint" not in text
    assert '"optimizer_steps": 0' in text
    assert '"validation_accessed": False' in text
    assert '"frozen_test_accessed": False' in text
    assert "gpu_memory_fraction) <= 0.45" in text
    assert "fcp012_gradient_window_complete" in text
