import unittest

import torch

from fluid_control.paired_training import combine_paired_rollout_batch, paired_batch_indices


def fixture(batch=2):
    return {
        "action_state": torch.ones(batch, 101, 3, 4, 5),
        "zero_state": torch.zeros(batch, 101, 3, 4, 5),
        "action_omega": torch.ones(batch, 101, 1),
        "zero_omega": torch.zeros(batch, 101, 1),
        "action_force": torch.ones(batch, 101, 4),
        "zero_force": torch.zeros(batch, 101, 4),
        "mask": torch.ones(batch, 1, 4, 5),
        "paired_targets": torch.ones(batch, 3, 3),
    }


class PairedTrainingContractTest(unittest.TestCase):
    def test_frontloaded_schedule_is_backward_compatible(self):
        self.assertEqual(paired_batch_indices(1368, 16), tuple(range(16)))

    def test_interleaved_schedule_uses_fixed_floor_rule_and_endpoints(self):
        expected = tuple(i * 1367 // 15 for i in range(16))
        actual = paired_batch_indices(1368, 16, "interleaved")
        self.assertEqual(actual, expected)
        self.assertEqual((actual[0], actual[-1]), (0, 1367))
        self.assertEqual(len(actual), len(set(actual)))

    def test_invalid_schedule_fails_closed(self):
        with self.assertRaises(ValueError):
            paired_batch_indices(10, 11, "interleaved")
        with self.assertRaises(ValueError):
            paired_batch_indices(10, 2, "random")

    def test_action_then_zero_order_and_causal_targets(self):
        combined = combine_paired_rollout_batch(fixture())
        self.assertEqual(combined["state"].shape, (4, 3, 4, 5))
        self.assertEqual(combined["target_state"].shape, (4, 100, 3, 4, 5))
        self.assertTrue(torch.equal(combined["state"][:2], torch.ones(2, 3, 4, 5)))
        self.assertTrue(torch.equal(combined["state"][2:], torch.zeros(2, 3, 4, 5)))
        self.assertEqual(combined["action_force"].shape, (2, 100, 4))
        self.assertEqual(combined["zero_force"].shape, (2, 100, 4))

    def test_wrong_keys_shape_and_nonfinite_fail_closed(self):
        value = fixture()
        value.pop("paired_targets")
        with self.assertRaisesRegex(ValueError, "keys differ"):
            combine_paired_rollout_batch(value)
        value = fixture()
        value["action_state"] = value["action_state"][:, :-1]
        with self.assertRaisesRegex(ValueError, "identical"):
            combine_paired_rollout_batch(value)
        value = fixture()
        value["action_force"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            combine_paired_rollout_batch(value)


if __name__ == "__main__":
    unittest.main()
