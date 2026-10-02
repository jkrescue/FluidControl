"""Regression checks for objective-focused rollout force weighting."""

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from train_tandem_fno_rollout import force_channel_weights


class ForceChannelWeightTests(unittest.TestCase):
    def test_unspecified_preserves_equal_channel_loss(self) -> None:
        cfg = OmegaConf.create({"training": {}})
        actual = force_channel_weights(cfg, (0, 1, 2, 3), torch.device("cpu"))
        torch.testing.assert_close(actual, torch.full((4,), 0.25))

    def test_rear_drag_weight_is_normalized(self) -> None:
        cfg = OmegaConf.create(
            {"training": {"force_channel_weights": [1, 1, 4, 1]}}
        )
        actual = force_channel_weights(cfg, (0, 1, 2, 3), torch.device("cpu"))
        torch.testing.assert_close(actual, torch.tensor([1, 1, 4, 1]) / 7)

    def test_invalid_weights_fail_closed(self) -> None:
        for values in ([1, 1], [1, 0, 4, 1], [1, -1, 4, 1], [1, 1, float("nan"), 1]):
            cfg = OmegaConf.create({"training": {"force_channel_weights": values}})
            with self.subTest(values=values), self.assertRaises(ValueError):
                force_channel_weights(cfg, (0, 1, 2, 3), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
