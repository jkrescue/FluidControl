import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

P = Path(__file__).resolve().parents[1] / "scripts/run_fcp003c_train_fit_calibration.py"
S = importlib.util.spec_from_file_location("calibration", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_schedule_is_one_128_stream_with_64_positions(self):
        values = M.calibration_positions()
        self.assertEqual(len(values), 64)
        self.assertEqual(len(set(values)), 64)
        self.assertEqual((values[0], values[-1]), (0, 127))

    def test_pair_passes_require_eight_complete_sets(self):
        expected = {
            f"b{i:02d}:{p}" for i in (0, 2, 4, 6) for p in ("multisine", "prbs")
        }
        M.validate_pair_passes([sorted(expected) for _ in range(8)], expected)
        broken = [sorted(expected) for _ in range(8)]
        broken[3][-1] = broken[3][0]
        with self.assertRaises(RuntimeError):
            M.validate_pair_passes(broken, expected)
        with self.assertRaises(RuntimeError):
            M.validate_pair_passes([sorted(expected) for _ in range(7)], expected)

    def test_regular_step_is_exactly_one_clipped_optimizer_step(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.weight.detach().clone()
        loss = model(torch.ones(1, 2)).square().mean()
        result = M.regular_optimizer_step(model, optimizer, loss, 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(result["preclip_gradient_norm"])))
        self.assertFalse(torch.equal(before, model.weight.detach()))
        self.assertEqual(optimizer.state_dict()["state"][0]["step"], 1)

    def test_regular_step_rejects_nonfinite_without_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.weight.detach().clone()
        loss = model(torch.ones(1, 1)).sum() * torch.tensor(float("nan"))
        with self.assertRaises(FloatingPointError):
            M.regular_optimizer_step(model, optimizer, loss, 1.0)
        self.assertTrue(torch.equal(before, model.weight.detach()))
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_h1_branch_uses_qt_plus_delta_and_endpoint_actions(self):
        fake = types.ModuleType("fluid_control.paired_step_force")

        def step_input(initial, targets, mask, omega, step):
            state = initial if step == 0 else targets[:, step - 1]
            now = omega[:, step, :, None, None].expand(-1, 1, 2, 2)
            nxt = omega[:, step + 1, :, None, None].expand(-1, 1, 2, 2)
            return torch.cat((state, mask, now, nxt), 1)

        fake.true_state_step_input = step_input
        old = sys.modules.get("fluid_control.paired_step_force")
        sys.modules["fluid_control.paired_step_force"] = fake
        try:
            state = torch.zeros(1, 3, 3, 2, 2)
            state[:, 1] = 2
            state[:, 2] = 5
            pair = {
                "action_state": state,
                "action_omega": torch.tensor([[[0.0], [0.1], [0.2]]]),
                "mask": torch.ones(1, 1, 2, 2),
            }

            def predict(_model, inputs, _mask):
                delta = inputs[:, :3] * 0 + inputs[:, 5:6]
                return delta.expand(-1, 3, -1, -1), inputs[:, 5:6, 0, 0].repeat(1, 4)

            predicted, force = M._branch_h1(
                torch.nn.Identity(), pair, "action", predict
            )
        finally:
            if old is None:
                sys.modules.pop("fluid_control.paired_step_force", None)
            else:
                sys.modules["fluid_control.paired_step_force"] = old
        self.assertTrue(torch.allclose(predicted[:, 0], torch.full((1, 3, 2, 2), 0.1)))
        self.assertTrue(torch.allclose(predicted[:, 1], torch.full((1, 3, 2, 2), 2.2)))
        self.assertTrue(torch.allclose(force[0, :, 0], torch.tensor([0.1, 0.2])))

    def test_train_readout_flattens_time_and_keeps_four_force_channels(self):
        evaluator = types.ModuleType("evaluate_tandem_fno")

        def field_sums(error, target, mask):
            self.assertEqual(tuple(error.shape), (100, 3, 2, 2))
            self.assertEqual(tuple(target.shape), (100, 3, 2, 2))
            self.assertEqual(tuple(mask.shape), (100, 1, 2, 2))
            return torch.stack(
                [
                    (error.double().square() * mask).sum((0, 2, 3)),
                    (target.double().square() * mask).sum((0, 2, 3)),
                ]
            )

        evaluator.field_error_sums = field_sums
        evaluator.relative_field_metrics = lambda sums: {"sums": sums.tolist()}
        old = sys.modules.get("evaluate_tandem_fno")
        sys.modules["evaluate_tandem_fno"] = evaluator
        try:
            state = (
                torch.arange(101, dtype=torch.float32)[None, :, None, None, None]
                .expand(1, 101, 3, 2, 2)
                .clone()
            )
            action_force = (
                torch.tensor([1.0, 2.0, 3.0, 4.0])[None, None].expand(1, 101, 4).clone()
            )
            pair = {
                "action_state": state,
                "zero_state": state.clone(),
                "action_omega": torch.zeros(1, 101, 1),
                "zero_omega": torch.zeros(1, 101, 1),
                "action_force": action_force,
                "zero_force": torch.zeros(1, 101, 4),
                "mask": torch.ones(1, 1, 2, 2),
            }

            def predict(_model, inputs, _mask):
                return torch.ones_like(inputs[:, :3]), torch.zeros(inputs.shape[0], 4)

            result = M.train_h1_readout(
                torch.nn.Identity(),
                [
                    {
                        "panel": "train_paired_window",
                        "identity": "b00:prbs",
                        "pair": pair,
                    },
                    {
                        "panel": "train_paired_window",
                        "identity": "b00:multisine",
                        "pair": pair,
                    },
                ],
                torch.zeros(1, 1, 3, 1, 1),
                torch.ones(1, 1, 3, 1, 1),
                torch.ones(4),
                predict,
            )["train_paired_window"]
        finally:
            if old is None:
                sys.modules.pop("evaluate_tandem_fno", None)
            else:
                sys.modules["evaluate_tandem_fno"] = old
        self.assertEqual(
            result["per_pair"]["b00:prbs"]["absolute"]["rear_cl"]["mae"], 4.0
        )
        self.assertEqual(
            result["per_pair"]["b00:prbs"]["action_minus_zero"]["front_cd"]["mae"], 1.0
        )
        self.assertEqual(result["force_absolute"]["rear_cl"]["count"], 300)
        self.assertEqual(result["force_action_absolute"]["rear_cl"]["count"], 200)
        self.assertEqual(result["force_action_absolute"]["rear_cl"]["mae"], 4.0)
        self.assertEqual(result["force_zero_absolute"]["rear_cl"]["count"], 100)
        self.assertEqual(result["force_zero_absolute"]["rear_cl"]["mae"], 0.0)
        self.assertEqual(
            result["force_zero_absolute_by_phase"]["b00"]["rear_cl"]["count"],
            100,
        )
        self.assertEqual(result["field"]["sums"][0], [0.0, 0.0, 0.0])
        self.assertEqual(
            result["per_pair"]["b00:prbs"]["field_action"]["sums"][0],
            [0.0, 0.0, 0.0],
        )

    def test_train_readout_rejects_broadcast_force_std(self):
        with self.assertRaises(ValueError):
            M.train_h1_readout(
                torch.nn.Identity(),
                [],
                torch.zeros(1, 1, 3, 1, 1),
                torch.ones(1, 1, 3, 1, 1),
                torch.ones(1, 1, 4),
                lambda *_: None,
            )

    def test_late_readout_has_101_targets_and_includes_target200(self):
        evaluator = types.ModuleType("evaluate_tandem_fno")
        evaluator.field_error_sums = lambda error, target, mask: torch.stack(
            [
                (error.double().square() * mask).sum((0, 2, 3)),
                (target.double().square() * mask).sum((0, 2, 3)),
            ]
        )
        evaluator.relative_field_metrics = lambda sums: {"sums": sums.tolist()}
        old = sys.modules.get("evaluate_tandem_fno")
        sys.modules["evaluate_tandem_fno"] = evaluator
        try:
            state = (
                torch.arange(102, dtype=torch.float32)[None, :, None, None, None]
                .expand(1, 102, 3, 2, 2)
                .clone()
            )
            action_force = torch.zeros(1, 102, 4)
            action_force[:, -1, 3] = 101.0
            pair = {
                "action_state": state,
                "zero_state": state.clone(),
                "action_omega": torch.zeros(1, 102, 1),
                "zero_omega": torch.zeros(1, 102, 1),
                "action_force": action_force,
                "zero_force": torch.zeros(1, 102, 4),
                "mask": torch.ones(1, 1, 2, 2),
            }

            def predict(_model, inputs, _mask):
                return torch.ones_like(inputs[:, :3]), torch.zeros(inputs.shape[0], 4)

            result = M.train_h1_readout(
                torch.nn.Identity(),
                [
                    {
                        "panel": "train_late_window",
                        "identity": "b00:prbs",
                        "pair": pair,
                    }
                ],
                torch.zeros(1, 1, 3, 1, 1),
                torch.ones(1, 1, 3, 1, 1),
                torch.ones(4),
                predict,
            )["train_late_window"]
        finally:
            if old is None:
                sys.modules.pop("evaluate_tandem_fno", None)
            else:
                sys.modules["evaluate_tandem_fno"] = old
        self.assertEqual(result["force_absolute"]["rear_cl"]["count"], 202)
        self.assertEqual(result["force_action_absolute"]["rear_cl"]["count"], 101)
        self.assertEqual(result["force_zero_absolute"]["rear_cl"]["count"], 101)
        self.assertEqual(result["force_action_minus_zero"]["rear_cl"]["count"], 101)
        self.assertEqual(
            result["per_pair"]["b00:prbs"]["absolute"]["rear_cl"]["mae"], 1.0
        )

    def test_tensor_state_hash_covers_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        one = M.tensor_state_sha256(model)
        model.running_mean[0] = 1
        self.assertNotEqual(one, M.tensor_state_sha256(model))


if __name__ == "__main__":
    unittest.main()
