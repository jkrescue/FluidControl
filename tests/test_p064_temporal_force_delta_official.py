from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch


REPO = Path(os.environ.get("P064_REPO", "/workspace/fluid_control"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(Path(__file__).resolve().parents[1] / "scripts")]
import p064_temporal_force_delta as residual


def test_actual_official_fno_terminal_rows_are_exactly_supported():
    from omegaconf import OmegaConf
    from train_tandem_fno import build_model

    config = REPO / "artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml"
    model = build_model(OmegaConf.load(config))
    named = dict(model.named_parameters())
    assert named[residual.FINAL_WEIGHT].shape == (7, 128)
    assert named[residual.FINAL_BIAS].shape == (7,)
    field_weight = named[residual.FINAL_WEIGHT][:3].detach().clone()
    field_bias = named[residual.FINAL_BIAS][:3].detach().clone()
    receipt = residual.zero_force_output_rows(model)
    assert receipt["field_rows_preserved"] is True
    assert torch.equal(field_weight, named[residual.FINAL_WEIGHT][:3])
    assert torch.equal(field_bias, named[residual.FINAL_BIAS][:3])
    assert torch.count_nonzero(named[residual.FINAL_WEIGHT][3:]) == 0
    assert torch.count_nonzero(named[residual.FINAL_BIAS][3:]) == 0


def test_worker_is_no_save_train_only_and_fixed_first_b_schedule_slot():
    worker = Path(__file__).resolve().parents[1] / "scripts" / "probe_p064_temporal_force_delta_r3.py"
    text = worker.read_text()
    assert "save_checkpoint" not in text
    assert "FIXED_START = 0" in text
    assert "first controlled_b00 slot" in text
    assert '"dev_or_frozen_accessed": False' in text
    assert '"models_saved": 0' in text
    assert "COMPLETE_NOT_A_CANDIDATE" in text
    assert "two real optimizer steps on isolated arm copies" in text
    assert "LR = 1.5625e-7" in text


def test_worker_sets_actual_parent_training_precision_before_guard():
    import probe_p064_temporal_force_delta_r3 as worker
    from fluid_control.dual_fno import validate_runtime_precision

    worker.configure_parent_training_precision(torch)
    receipt = validate_runtime_precision()
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert receipt["float32_matmul_precision"] == "high"
