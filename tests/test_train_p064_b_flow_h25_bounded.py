import ast
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_p064_b_flow_h25_bounded.py"


def load():
    spec = importlib.util.spec_from_file_location("bounded_h25", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resources():
    return {
        "container_memory_gib": 48,
        "allocator_bytes": 32 * 2**30,
        "startup_mem_available_gib": 80,
        "runtime_mem_available_gib": 22,
        "reserved_mem_available_gib": 20,
        "scales_deadline_seconds": 1800,
        "probe_deadline_seconds": 600,
        "train_deadline_seconds": 1800,
    }


def test_fixed_bounded_protocol_and_parent():
    module = load()
    p = module.protocol()
    assert p["horizon"] == 25 and p["scale_horizon"] == 10
    assert p["training_windows"] == 256 and p["optimizer_steps"] == 32
    assert p["accumulation_windows"] == 8 and p["learning_rate"] == 1e-5
    assert p["selected_training_order_sha256"] == module.SELECTED_ORDER_SHA
    assert module.PARENT_SHA == "92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891"


def test_scale_summary_requires_all1368_and_is_plain_parent_mean():
    module = load()
    rows = [{"raw_field": 2.0, "raw_force": 4.0} for _ in range(1368)]
    assert module.summarize_scales(rows) == {"field": 2.0, "force": 4.0}
    with pytest.raises(ValueError):
        module.summarize_scales(rows[:256])


def test_uma_resource_limits_ignore_cuda_cache_but_enforce_available_and_deadline():
    module = load()
    module.resource_limits({"MemAvailable": 81}, 0.0, 0, "resource-probe", startup=True)
    module.resource_limits({"MemAvailable": 22}, 0.0, 599, "resource-probe")
    with pytest.raises(RuntimeError):
        module.resource_limits({"MemAvailable": 21.99}, 100.0, 1, "train")
    with pytest.raises(RuntimeError):
        module.resource_limits({"MemAvailable": 100}, 100.0, 601, "resource-probe")


def test_actual_caller_freezes_aero_and_uses_h10_only_for_scales():
    tree = ast.parse(SCRIPT.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    objective_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                       and node.func.attr == "field_force_rollout_objective"]
    assert len(objective_calls) == 1
    keywords = {item.arg: ast.unparse(item.value) for item in objective_calls[0].keywords}
    assert keywords["horizon"] == "10 if mode == 'scales' else 25"
    assert keywords["backward"] == "mode != 'scales'"
    source = SCRIPT.read_text()
    assert "aero.to(dist.device).eval().requires_grad_(False)" in source
    assert "flow.to(dist.device).train().requires_grad_(True)" in source
    assert "for step in range(1, groups + 1)" in source
    assert "groups = 171 if mode == \"scales\" else 32" in source


def test_probe_performs_one_real_optimizer_update_without_save():
    source = SCRIPT.read_text()
    assert "sample, metadata = train[expected[0]]" in source
    assert "base.optimizer_update(flow, optimizer, saved, 1)" in source
    assert 'optimizer_steps=(32 if mode == "train" else 1 if mode == "resource-probe" else 0)' in source
    assert 'if mode == "train":' in source and "save_terminal(" in source


def test_validate_scales_receipt_is_b_parent_h10_not_h25(tmp_path):
    module = load()
    spec = {"parent_manifest": {"sha256": module.PARENT_SHA},
            "config": {"sha256": module.CONFIG_SHA},
            "train_audit": {"sha256": module.AUDIT_SHA}, "source_sha256": {"runner": "c" * 64}}
    receipt = {
        "status": "FC_P064_B_H10_PARENT_SCALES_COMPLETE_NOT_ADMISSION",
        "mode": "scales", "optimizer_steps": 0, "optimizer_created": False,
        "model_saved": False, "scientific_admission": False,
        "flow_initial_tensor_sha256": "a" * 64, "flow_terminal_tensor_sha256": "a" * 64,
        "frozen_aerodynamic_tensor_sha256": "b" * 64,
        "training_windows": 1368, "sampler_order_sha256": module.ORDER_SHA,
        "fixed_scales": {"field": 2.0, "force": 3.0}, "scale_horizon": 10,
        "protocol": module.protocol(), "source_spec": spec,
    }
    assert module.validate_scales_receipt(receipt, spec) == {"field": 2.0, "force": 3.0}
    receipt["scale_horizon"] = 25
    with pytest.raises(ValueError):
        module.validate_scales_receipt(receipt, spec)
    receipt["scale_horizon"] = 10
    receipt["source_spec"] = dict(spec, source_sha256={"runner": "d" * 64})
    with pytest.raises(ValueError):
        module.validate_scales_receipt(receipt, spec)


def test_actual_h25_eight_window_accumulation_and_adam_step():
    module = load()
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def run(value):
        loss = model(torch.tensor([[float(value)]])).square().mean()
        loss.backward()
        return {"total": float(loss.detach()), "rollout_steps": 25}

    before = model.weight.detach().clone()
    record = module.accumulate_eight_h25_gradients(
        torch, model, optimizer, range(1, 9), run)
    assert record["windows"] == 8
    assert record["optimizer_step_performed"] is False
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())

    with pytest.raises(FloatingPointError):
        module.accumulate_eight_h25_gradients(
            torch, model, optimizer, range(8),
            lambda _: {"total": 1.0, "rollout_steps": 10})


def test_v3_receipts_allow_only_reviewed_runner_repair():
    module = load()
    common = {
        "parent_manifest": {"sha256": module.PARENT_SHA},
        "config": {"sha256": module.CONFIG_SHA},
        "train_audit": {"sha256": module.AUDIT_SHA},
    }
    old = dict(common, source_sha256={
        module.RUNNER_RELATIVE: module.V3_RUNNER_SHA,
        "scripts/p029_control_aware_flow_objective.py": "a" * 64,
    })
    current = dict(common, source_sha256={
        module.RUNNER_RELATIVE: module.sha(SCRIPT),
        "scripts/p029_control_aware_flow_objective.py": "a" * 64,
    })
    assert module.receipt_source_matches_v4(old, current)
    current["source_sha256"] = dict(current["source_sha256"])
    current["source_sha256"]["scripts/p029_control_aware_flow_objective.py"] = "b" * 64
    assert not module.receipt_source_matches_v4(old, current)
    current["source_sha256"]["scripts/p029_control_aware_flow_objective.py"] = "a" * 64
    current["parent_manifest"] = {"sha256": "c" * 64}
    assert not module.receipt_source_matches_v4(old, current)


def test_real_h25_objective_feeds_eight_window_accumulator():
    module = load()
    torch = pytest.importorskip("torch")
    objective_path = SCRIPT.with_name("p029_control_aware_flow_objective.py")
    spec = importlib.util.spec_from_file_location("bounded_h25_objective", objective_path)
    objective = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objective)

    class Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))

    class Aero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    flow, aero = Flow(), Aero().eval()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-5)
    initial = torch.zeros(1, 3, 1, 1)
    target = torch.zeros(1, 100, 3, 1, 1)
    mask = torch.ones(1, 1, 1, 1)
    actions = torch.zeros(1, 101, 1)
    forces = torch.zeros(1, 100, 4)

    def predict(model, inputs, current_mask):
        if model is flow:
            return model.weight.expand_as(initial), torch.zeros(1, 4)
        return torch.zeros_like(initial), torch.zeros(1, 4)

    def run(_):
        return objective.field_force_rollout_objective(
            flow, aero, initial, target, mask, actions, forces,
            predict, lambda current, *_: current,
            field_scale=1.0, force_scale=1.0, backward=True, horizon=25)

    single_flow = Flow()
    single_flow.weight.data.copy_(flow.weight.data)
    flow = single_flow
    single = run(0)
    expected_gradient = flow.weight.grad.detach().clone()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-5)
    record = module.accumulate_eight_h25_gradients(
        torch, flow, optimizer, range(8), run)
    assert len(record["records"]) == 8
    assert all(row["rollout_steps"] == 25 for row in record["records"])
    assert torch.equal(flow.weight.grad, expected_gradient)
    before = flow.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, flow.weight.detach())
