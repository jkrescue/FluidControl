import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).parent
REPO = HERE.parent
HELPER_DIR = HERE if (HERE / 'p064_y_reflection.py').exists() else REPO / 'src/fluid_control'
sys.path.insert(0, str(HELPER_DIR))
import p064_y_reflection as r


STATS = r.ReflectionStats(
    state_mean=torch.tensor([0.9297897467, 4.374019421e-7, 3.13e-15]),
    state_std=torch.tensor([0.4, 0.3, 0.2]),
    force_mean=torch.tensor([1.39117904, -7.2559492e-5, 0.885191385, -0.00013177838]),
    force_std=torch.tensor([0.5, 0.25, 0.4, 0.2]),
)


def rollout(height=8, width=6, steps=5):
    generator = torch.Generator().manual_seed(7)
    mask = torch.ones(1, height, width)
    mask[:, 2:4, 1:3] = 0
    mask[:, height - 4 : height - 2, 1:3] = 0
    state = torch.randn(3, height, width, generator=generator) * mask
    target = torch.randn(steps, 3, height, width, generator=generator) * mask
    return {
        "state": state,
        "target_state": target,
        "omega": torch.linspace(-1, 1, steps + 1).reshape(-1, 1),
        "target_force": torch.randn(steps, 4, generator=generator),
        "mask": mask,
        "time": torch.tensor(0.0),
    }


def test_physical_involution_is_exact_and_normalized_roundtrip_is_bounded():
    sample = rollout()
    physical_state = sample["state"] * STATS.state_std[:, None, None] + STATS.state_mean[:, None, None]
    physical_force = sample["target_force"] * STATS.force_std + STATS.force_mean
    assert r.physical_roundtrip_report(physical_state, physical_force) == {
        "physical_max_abs_error": 0.0
    }
    report = r.normalized_roundtrip_report(sample, STATS)
    assert report["masked_cells_exact_zero"] is True
    assert report["normalized_max_abs_error"] <= 8 * torch.finfo(torch.float32).eps * max(
        1.0, float(sample["target_force"].abs().max())
    )


def test_physical_parity_nonzero_means_actions_and_mask():
    sample = rollout()
    reflected = r.reflect_normalized_rollout(sample, STATS)
    physical = sample["state"] * STATS.state_std[:, None, None] + STATS.state_mean[:, None, None]
    reflected_physical = reflected["state"] * STATS.state_std[:, None, None] + STATS.state_mean[:, None, None]
    active = reflected["mask"].expand_as(reflected_physical).bool()
    expected = r.reflect_physical_state(physical)
    assert torch.equal(reflected["omega"], -sample["omega"])
    assert torch.equal(reflected["mask"], sample["mask"].flip(-2))
    assert torch.allclose(reflected_physical[active], expected[active])
    direct_sign_flip_v = -sample["state"][1].flip(-2)
    assert not torch.equal(reflected["state"][1], direct_sign_flip_v)
    assert torch.all(reflected["state"].masked_select(~active) == 0)


def test_asymmetric_mask_fails_closed():
    sample = rollout()
    sample["mask"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="not exactly"):
        r.reflect_normalized_rollout(sample, STATS)


def test_missing_required_key_is_rejected_even_with_extra_time():
    sample = rollout()
    del sample["omega"]
    sample["extra"] = torch.tensor(1)
    with pytest.raises(ValueError, match="omega"):
        r.reflect_normalized_rollout(sample, STATS)


def test_unbatched_contract_rejects_implicit_batch_broadcast():
    sample = rollout()
    sample["state"] = sample["state"].unsqueeze(0)
    with pytest.raises(ValueError, match="unbatched"):
        r.reflect_normalized_rollout(sample, STATS)


def test_nonfinite_statistics_fail_closed():
    sample = rollout()
    bad = r.ReflectionStats(
        STATS.state_mean,
        torch.tensor([0.4, float("nan"), 0.2]),
        STATS.force_mean,
        STATS.force_std,
    )
    with pytest.raises(ValueError, match="finite"):
        r.reflect_normalized_rollout(sample, bad)


def test_history_including_empty_k1_reflects_without_original_reentry():
    sample = rollout()
    empty = torch.empty(1, 0, 3, 8, 6)
    actions = torch.empty(1, 0, 1)
    reflected, reflected_actions = r.reflect_normalized_history(
        empty, actions, sample["mask"], STATS
    )
    assert reflected.shape == empty.shape and reflected_actions.shape == actions.shape
    states = sample["state"].reshape(1, 1, 3, 8, 6)
    actions = torch.tensor([[[0.25]]])
    reflected, reflected_actions = r.reflect_normalized_history(
        states, actions, sample["mask"], STATS
    )
    assert torch.equal(reflected_actions, -actions)
    assert not torch.equal(reflected, states)


def test_serial_halves_independent_flow_and_one_divide_clip_step():
    accumulation_spec = importlib.util.spec_from_file_location(
        "p015_accumulation", (HERE / "train_fcp015_window_accumulation.py"
                            if (HERE / "train_fcp015_window_accumulation.py").exists()
                            else REPO / "scripts/train_fcp015_window_accumulation.py")
    )
    accumulation = importlib.util.module_from_spec(accumulation_spec)
    accumulation_spec.loader.exec_module(accumulation)
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    windows = [
        {"x": torch.tensor([[0.01 * (index + 1), 0.02]]), "target": torch.tensor([[0.005]])}
        for index in range(8)
    ]
    flow_calls = []

    def reflect(window):
        return {
            "x": torch.stack((-window["x"][:, 0], window["x"][:, 1]), dim=1),
            "target": -window["target"],
        }

    def branch(window, label):
        # This stands in for constructing frozen_flow_states independently from
        # the branch q0/actions.  A reused original flow would duplicate its id.
        flow_state = window["x"].clone()
        flow_calls.append((label, flow_state.data_ptr(), flow_state.clone()))
        loss = ((model(flow_state) - window["target"]) ** 2).mean()
        loss.backward()
        scalar = float(loss.detach())
        return {"h1_balanced": scalar, "ar_balanced": scalar, "total": scalar}

    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    expected_losses = []
    for window in windows:
        mirror = reflect(window)
        expected_losses.append(
            0.5 * ((reference(window["x"]) - window["target"]) ** 2).mean()
            + 0.5 * ((reference(mirror["x"]) - mirror["target"]) ** 2).mean()
        )
    expected = torch.stack(expected_losses).mean()
    expected.backward()
    expected_grad = reference.weight.grad.clone()
    expected_preclip = float(expected_grad.norm())
    clipped_expected_grad = expected_grad * min(1.0, 1.0 / expected_preclip)
    expected_weight = reference.weight.detach() - 0.1 * clipped_expected_grad

    def train_pair(window):
        result = r.serial_half_pair_backward(model, window, reflect(window), branch)
        return {
            key: 0.5 * result["original"][key] + 0.5 * result["reflected"][key]
            for key in ("h1_balanced", "ar_balanced", "total")
        }

    record = accumulation.accumulation_step(
        model,
        optimizer,
        iter(windows),
        train_pair,
        lambda current: {"grad": float(current.weight.grad.norm())},
    )
    assert record["windows"] == 8
    assert record["optimizer_steps"] == 1
    assert record["preclip_mean_gradient_norm"] == pytest.approx(expected_preclip, abs=1e-8)
    assert len(flow_calls) == 16
    assert [label for label, _, _ in flow_calls] == [x for _ in range(8) for x in ("original", "reflected")]
    assert all(flow_calls[i][1] != flow_calls[i + 1][1] for i in range(0, 16, 2))
    assert torch.allclose(model.weight, expected_weight, atol=1e-7, rtol=0)


def test_half_hooks_preserve_existing_gradient_and_are_removed_on_exception():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.tensor([[3.0]])

    def branch(sample, label):
        loss = model(sample["x"]).sum()
        loss.backward()
        return {"total": float(loss.detach())}

    original = {"x": torch.tensor([[2.0]])}
    mirror = {"x": torch.tensor([[4.0]])}
    r.serial_half_pair_backward(model, original, mirror, branch)
    assert torch.equal(model.weight.grad, torch.tensor([[6.0]]))  # 3 + .5*2 + .5*4

    model.weight.grad = None

    def failing(sample, label):
        model(sample["x"]).sum().backward()
        raise RuntimeError("deliberate")

    with pytest.raises(RuntimeError, match="deliberate"):
        r.serial_half_pair_backward(model, original, mirror, failing)
    model.weight.grad = None
    model(original["x"]).sum().backward()
    assert torch.equal(model.weight.grad, torch.tensor([[2.0]]))  # no stale .5 hook


def test_branch_result_must_not_retain_graph():
    model = torch.nn.Linear(1, 1, bias=False)
    original = {"x": torch.tensor([[2.0]])}

    def bad_result(sample, label):
        value = model(sample["x"])
        value.sum().backward()
        return {"prediction": value}

    with pytest.raises(ValueError, match="retains an autograd graph"):
        r.serial_half_pair_backward(model, original, original, bad_result)
    model.weight.grad = None
    model(original["x"]).sum().backward()
    assert torch.equal(model.weight.grad, torch.tensor([[2.0]]))

    def large_detached(sample, label):
        model(sample["x"]).sum().backward()
        return {"prediction": torch.zeros(65)}

    model.weight.grad = None
    with pytest.raises(ValueError, match="large tensor"):
        r.serial_half_pair_backward(model, original, original, large_detached)


def test_disabled_path_is_exact_original_behavior():
    model_a = torch.nn.Linear(1, 1, bias=False)
    model_b = torch.nn.Linear(1, 1, bias=False)
    model_b.load_state_dict(model_a.state_dict())
    x = torch.tensor([[2.0]])
    target = torch.tensor([[1.0]])
    ((model_a(x) - target) ** 2).backward()
    calls = []

    def branch(sample, label):
        calls.append((label, sample["x"].data_ptr()))
        loss = ((model_b(sample["x"]) - sample["target"]) ** 2).mean()
        loss.backward()
        return {"total": float(loss.detach())}

    def must_not_reflect(_):
        raise AssertionError("disabled wrapper called reflection")

    r.optional_reflection_backward(
        False, model_b, {"x": x, "target": target}, must_not_reflect, branch
    )
    assert calls == [("original", x.data_ptr())]
    assert torch.equal(model_a.weight.grad, model_b.weight.grad)
    assert model_a.state_dict().keys() == model_b.state_dict().keys()


def test_actual_project_dataset_official_reader_fixture(tmp_path):
    if importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("run with reviewed curator Python")
    import h5py
    import numpy as np

    repo = Path("/workspace/fluid_control")
    source = repo / "src"
    sys.path.insert(0, str(source))
    from fluid_control.tandem_datapipe import TandemRolloutDataset

    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    stats = {
        "state_mean": STATS.state_mean.tolist(),
        "state_std": STATS.state_std.tolist(),
        "all_force_mean": STATS.force_mean.tolist(),
        "all_force_std": STATS.force_std.tolist(),
        "all_force_channels": ["front_Cd", "front_Cl", "rear_Cd", "rear_Cl"],
        "force_mean": STATS.force_mean[2:].tolist(),
        "force_std": STATS.force_std[2:].tolist(),
        "force_channels": ["rear_Cd", "rear_Cl"],
    }
    (root / "normalization.json").write_text(json.dumps(stats))
    (root / "manifest.json").write_text(json.dumps({"max_abs_omega": 0.75}))
    sample = rollout(height=8, width=6, steps=5)
    state = sample["state"] * STATS.state_std[:, None, None] + STATS.state_mean[:, None, None]
    target = sample["target_state"] * STATS.state_std[:, None, None] + STATS.state_mean[:, None, None]
    states = torch.cat((state[None], target), dim=0).numpy().astype("float32")
    forces = torch.cat((torch.zeros(1, 4), sample["target_force"] * STATS.force_std + STATS.force_mean)).numpy().astype("float32")
    with h5py.File(root / "train" / "fixture.h5", "w") as handle:
        handle.create_dataset("state", data=states)
        handle.create_dataset("mask", data=np.repeat(sample["mask"][None].numpy(), 6, axis=0))
        handle.create_dataset("omega", data=(sample["omega"] * 0.75).numpy())
        handle.create_dataset("force", data=forces)
        handle.create_dataset("time", data=np.arange(6, dtype="float32").reshape(6, 1) / 10)
    dataset = TandemRolloutDataset(root, "train", 5, stride=1, num_workers=1, force_indices=(0, 1, 2, 3))
    try:
        loaded, metadata = dataset[0]
        assert metadata["step"] == 0
        reflected = r.reflect_normalized_rollout(loaded, STATS)
        state_physical = loaded["state"].numpy() * STATS.state_std.numpy()[:, None, None] + STATS.state_mean.numpy()[:, None, None]
        expected_state = state_physical[:, ::-1, :].copy()
        expected_state[1] *= -1
        expected_state = (expected_state - STATS.state_mean.numpy()[:, None, None]) / STATS.state_std.numpy()[:, None, None]
        expected_state *= loaded["mask"].numpy()[:, ::-1, :]
        target_physical = loaded["target_state"].numpy() * STATS.state_std.numpy()[None, :, None, None] + STATS.state_mean.numpy()[None, :, None, None]
        expected_target = target_physical[:, :, ::-1, :].copy()
        expected_target[:, 1] *= -1
        expected_target = (expected_target - STATS.state_mean.numpy()[None, :, None, None]) / STATS.state_std.numpy()[None, :, None, None]
        expected_target *= loaded["mask"].numpy()[None, :, ::-1, :]
        force_physical = loaded["target_force"].numpy() * STATS.force_std.numpy() + STATS.force_mean.numpy()
        expected_force = force_physical.copy()
        expected_force[:, [1, 3]] *= -1
        expected_force = (expected_force - STATS.force_mean.numpy()) / STATS.force_std.numpy()
        np.testing.assert_allclose(reflected["state"].numpy(), expected_state, rtol=0, atol=2e-6)
        np.testing.assert_allclose(reflected["target_state"].numpy(), expected_target, rtol=0, atol=2e-6)
        np.testing.assert_allclose(reflected["target_force"].numpy(), expected_force, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(reflected["omega"].numpy(), -loaded["omega"].numpy())
        report = r.normalized_roundtrip_report(loaded, STATS)
        assert report["masked_cells_exact_zero"]
        assert reflected["state"].shape == loaded["state"].shape
        assert reflected["target_force"].shape == (5, 4)
        assert type(dataset).__mro__[1].__module__ == "fluid_control.tandem_datapipe"
    finally:
        dataset.close()


def test_actual_train_hdf_one_window_through_official_reader():
    if importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("run with reviewed curator Python")
    repo = Path("/workspace/fluid_control")
    sys.path.insert(0, str(repo / "src"))
    from fluid_control.tandem_datapipe import TandemRolloutDataset

    root = repo / "artifacts/b00_controlled_train_dataset_view_20261006"
    dataset = TandemRolloutDataset(
        root, "train", 5, stride=1, num_workers=1, force_indices=(0, 1, 2, 3)
    )
    try:
        loaded, metadata = dataset[0]
        normalization = json.loads((root / "normalization.json").read_text())
        actual_stats = r.ReflectionStats(
            torch.tensor(normalization["state_mean"]),
            torch.tensor(normalization["state_std"]),
            torch.tensor(normalization["all_force_mean"]),
            torch.tensor(normalization["all_force_std"]),
        )
        assert metadata["split"] == "train" and metadata["step"] == 0
        reflected = r.reflect_normalized_rollout(loaded, actual_stats)
        assert torch.equal(reflected["mask"], loaded["mask"].flip(-2))
        assert torch.equal(reflected["omega"], -loaded["omega"])
        assert r.normalized_roundtrip_report(loaded, actual_stats)["masked_cells_exact_zero"]
    finally:
        dataset.close()
