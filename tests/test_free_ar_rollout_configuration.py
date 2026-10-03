"""Focused regression checks for free-autoregressive rollout configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from hydra import compose, initialize_config_dir

import train_tandem_fno_rollout as rollout_module


ROOT = Path(__file__).resolve().parents[1]


class _DatasetArgumentsCaptured(RuntimeError):
    """Stop ``main`` after both dataset constructor calls are observed."""


def _compose(config_name: str):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        return compose(config_name=config_name)


def _dataset_horizons(config_name: str) -> tuple[int, int]:
    """Run the real horizon-selection branch and capture dataset arguments."""
    cfg = _compose(config_name)
    calls: list[tuple[object, ...]] = []

    def capture(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise _DatasetArgumentsCaptured
        return MagicMock()

    dist = SimpleNamespace(
        cuda=True,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        distributed=False,
    )
    properties = SimpleNamespace(name="cpu-test", total_memory=8 << 30)
    with tempfile.TemporaryDirectory() as temporary:
        cfg.output_dir = temporary
        with patch.object(
            rollout_module, "DistributedManager", return_value=dist
        ), patch.object(
            rollout_module.torch.cuda, "set_per_process_memory_fraction"
        ), patch.object(
            rollout_module.torch.cuda,
            "get_device_properties",
            return_value=properties,
        ), patch.object(
            rollout_module.torch.cuda, "manual_seed_all"
        ), patch.object(
            rollout_module, "PythonLogger", return_value=MagicMock()
        ), patch.object(
            rollout_module.LaunchLogger, "initialize"
        ), patch.object(
            rollout_module, "TandemRolloutDataset", side_effect=capture
        ):
            with unittest.TestCase().assertRaises(_DatasetArgumentsCaptured):
                rollout_module.main.__wrapped__(cfg)

    if len(calls) != 2:
        raise AssertionError(f"expected train and validation datasets, got {calls!r}")
    return int(calls[0][2]), int(calls[1][2])


class FreeAutoregressiveConfigurationTests(unittest.TestCase):
    def test_validation_horizon_defaults_to_training_horizon(self) -> None:
        self.assertEqual(_dataset_horizons("tandem_fno_full40_h20"), (20, 20))

    def test_validation_horizon_override_reaches_validation_dataset(self) -> None:
        self.assertEqual(
            _dataset_horizons("tandem_fno_full40_free_ar_h20"), (20, 100)
        )

    def test_h50_keeps_controlled_batch_and_h100_validation(self) -> None:
        cfg = _compose("tandem_fno_full40_free_ar_h50")
        self.assertEqual(int(cfg.training.batch_size), 4)
        self.assertEqual(
            _dataset_horizons("tandem_fno_full40_free_ar_h50"), (50, 100)
        )

    def test_free_ar_schedule_is_zero_for_every_epoch(self) -> None:
        cfg = _compose("tandem_fno_full40_free_ar_h20")
        self.assertEqual(
            [rollout_module.teacher_forcing_ratio(cfg, epoch) for epoch in (1, 2, 8, 100)],
            [0.0, 0.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
