"""Synthetic CPU engineering fixtures; no official model/HDF/GPU access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "train_p029_control_aware_flow.py"
if not SCRIPT.exists():
    SCRIPT = HERE.parent / "scripts/train_p029_control_aware_flow.py"
definition = importlib.util.spec_from_file_location("p029_test_runner", SCRIPT)
runner = importlib.util.module_from_spec(definition)
definition.loader.exec_module(runner)
SCRIPTS = HERE.parent / "scripts"
BASE = Path(os.environ.get("P028_RUNNER", SCRIPTS / "train_p028_flow_rollout.py"))
P029_OBJECTIVE = Path(os.environ.get("P029_OBJECTIVE", SCRIPTS / "p029_control_aware_flow_objective.py"))
P028_OBJECTIVE = Path(os.environ.get("P028_OBJECTIVE", SCRIPTS / "p028_flow_h10_objective.py"))
base = runner.load_module("p029_test_base", BASE)


def spec(mode="scales"):
    return dict(status="FC_P029_EXECUTION_APPROVED", mode=mode, protocol=runner.protocol(),
                parent_manifest=dict(path="parent.json", sha256=runner.PARENT_SHA),
                config=dict(path="config.yaml", sha256=runner.CONFIG_SHA),
                train_audit=dict(path="audit.json", sha256=runner.AUDIT_SHA),
                source_sha256={"toy": "a"*64}, resources=dict(allocator_fraction=.06),
                scales_receipt=dict(path="scales.json", sha256="b"*64),
                fixed_scales=dict(field=2., force=3.),
                resource_probe_receipt=dict(path="probe.json", sha256="c"*64))


def receipt(mode="scales"):
    value = dict(status="FC_P029_PARENT_SCALES_COMPLETE_NOT_ADMISSION", mode=mode,
                 protocol=runner.protocol(), optimizer_steps=0, optimizer_created=False,
                 model_saved=False, scientific_admission=False, flow_initial_tensor_sha256="d"*64,
                 flow_terminal_tensor_sha256="d"*64, frozen_aerodynamic_tensor_sha256="e"*64,
                 source_spec=spec(), training_windows=1368, sampler_order_sha256=runner.ORDER_SHA,
                 records=[dict(raw_field=2., raw_force=3.) for _ in range(1368)],
                 fixed_scales=dict(field=2., force=3.))
    if mode == "resource-probe":
        value.update(status="FC_P029_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION", training_windows=1,
                     source_spec=spec(mode), records=[])
    return value


@pytest.mark.parametrize("mode", runner.MODES)
def test_separate_mode_approval(mode):
    runner.validate_spec(spec(mode), mode)
    bad = spec(mode)
    bad["status"] = "FC_P028_EXECUTION_APPROVED"
    with pytest.raises(ValueError):
        runner.validate_spec(bad, mode)
    bad = spec(mode)
    bad["protocol"]["learning_rate"] *= 2
    with pytest.raises(ValueError):
        runner.validate_spec(bad, mode)


@pytest.mark.parametrize("mode,key", [("train", "resource_probe_receipt"),
                                       ("train", "scales_receipt"),
                                       ("resource-probe", "scales_receipt")])
def test_missing_actual_receipt_rejected(mode, key):
    value = spec(mode)
    del value[key]
    with pytest.raises(ValueError):
        runner.validate_spec(value, mode)


@pytest.mark.parametrize("values", [{"field": 0., "force": 1.}, {"field": True, "force": 1.},
                                      {"field": float("inf"), "force": 1.},
                                      {"field": 1.}, {"field": 1., "force": float("nan")}])
def test_invalid_scale(values):
    with pytest.raises(ValueError):
        runner.validate_scales(values)


def test_full_scales_recompute_and_receipt_binding():
    value = receipt()
    assert runner.validate_receipt(value, spec("train"), "scales") == dict(field=2., force=3.)
    for damage in ("count", "order", "source", "scale", "raw", "tensor", "optimizer", "parent"):
        bad = copy.deepcopy(value)
        if damage == "count":
            bad["records"].pop()
        elif damage == "order":
            bad["sampler_order_sha256"] = "0"*64
        elif damage == "source":
            bad["source_spec"]["source_sha256"] = {}
        elif damage == "scale":
            bad["fixed_scales"]["field"] += 1
        elif damage == "raw":
            bad["records"][0]["raw_force"] = float("nan")
        elif damage == "tensor":
            bad["flow_initial_tensor_sha256"] = bad["flow_terminal_tensor_sha256"] = None
        elif damage == "optimizer":
            bad["optimizer_created"] = True
        else:
            bad["source_spec"]["parent_manifest"]["sha256"] = "0"*64
        with pytest.raises(ValueError):
            runner.validate_receipt(bad, spec("train"), "scales")


def test_resource_binds_actual_scales_receipt_and_zero_updates():
    value = receipt("resource-probe")
    assert runner.validate_receipt(value, spec("train"), "resource-probe") == dict(field=2., force=3.)
    for key, replacement in (("optimizer_steps", 1), ("training_windows", 2),
                             ("mode", "train"), ("model_saved", True)):
        bad = copy.deepcopy(value)
        bad[key] = replacement
        with pytest.raises(ValueError):
            runner.validate_receipt(bad, spec("train"), "resource-probe")
    value["source_spec"]["scales_receipt"]["sha256"] = "f"*64
    with pytest.raises(ValueError):
        runner.validate_receipt(value, spec("train"), "resource-probe")


def test_original_parent_and_fixed_protocol_not_p028_terminal():
    assert runner.sha(BASE) == runner.BASE_SHA
    value = runner.protocol()
    assert value["training_windows"] == 1368 and value["optimizer_steps"] == 171
    assert value["field_weight"] == value["force_weight"] == .5
    assert value["sampler_order_sha256"] == base.ORDER_SHA
    assert runner.PARENT_SHA == base.PARENT_SHA
    assert value["experiment"] != base.protocol()["experiment"]


def test_resource_floors_and_deadlines():
    runner.resource_limits(dict(MemFree=30, MemAvailable=50), 30, 0, "scales", True)
    for mode in runner.MODES:
        with pytest.raises(RuntimeError):
            runner.resource_limits(dict(MemFree=19.99, MemAvailable=100), 30, 1, mode)
        with pytest.raises(RuntimeError):
            runner.resource_limits(dict(MemFree=30, MemAvailable=100), 19.99, 1, mode)
    for mode in ("scales", "resource-probe"):
        with pytest.raises(RuntimeError):
            runner.resource_limits(dict(MemFree=30, MemAvailable=100), 30, 901, mode)
    runner.resource_limits(dict(MemFree=30, MemAvailable=100), 30, 901, "train")


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_net = torch.nn.Module()
        self.decoder_net.final_layer = torch.nn.Module()
        self.decoder_net.final_layer.linear = torch.nn.Linear(3, 7)
        torch.nn.init.constant_(self.decoder_net.final_layer.linear.weight, .01)
        torch.nn.init.constant_(self.decoder_net.final_layer.linear.bias, .01)

    def forward(self, x):
        return self.decoder_net.final_layer.linear(x[:, :3].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def test_actual_objective_eight_mean_clip_step_and_frozen_aero():
    objective = runner.load_module("p029_test_objective", P029_OBJECTIVE)
    accumulation = runner.load_module("p029_test_accumulation", P028_OBJECTIVE)
    flow, aero = Flow(), Flow().eval().requires_grad_(False)
    before_aero = {k: v.clone() for k, v in aero.state_dict().items()}
    saved = base.preserve_rows(flow)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
    base.check_optimizer(flow, optimizer, 0)
    def predict(model, x, mask):
        out = model(x)
        return out[:, :3], (out[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1))
    def make(q, mask, a, b):
        return torch.cat((q, mask, a[:, :, None, None].expand_as(mask), b[:, :, None, None].expand_as(mask)), 1)
    def run(_):
        return objective.field_force_rollout_objective(flow, aero, torch.ones(1, 3, 2, 2),
            torch.zeros(1, 100, 3, 2, 2), torch.ones(1, 1, 2, 2), torch.zeros(1, 101, 1),
            torch.zeros(1, 100, 4), predict, make, field_scale=2., force_scale=3., backward=True)
    optimizer.zero_grad()
    one = run(None)
    expected = {name: p.grad.clone() for name, p in flow.named_parameters()}
    record = accumulation.accumulate_eight_window_gradients(flow, optimizer, range(8), run)
    for name, p in flow.named_parameters():
        torch.testing.assert_close(p.grad, expected[name])
    assert record["mean_total"] == pytest.approx(one["total"])
    update = base.optimizer_update(flow, optimizer, saved, 1)
    assert update["applied_clip_scale"] <= 1
    for name, value in saved.items():
        assert torch.equal(dict(flow.named_parameters())[name][3:], value)
        assert torch.count_nonzero(optimizer.state[dict(flow.named_parameters())[name]]["exp_avg"][3:]) == 0
    assert all(p.grad is None and p not in optimizer.state for p in aero.parameters())
    assert all(torch.equal(v, before_aero[k]) for k, v in aero.state_dict().items())


def test_scales_forward_only_preserves_tensors_and_has_no_grad(monkeypatch):
    objective = runner.load_module("p029_test_scales_objective", P029_OBJECTIVE)
    flow, aero = Flow(), Flow().eval().requires_grad_(False)
    before = [p.detach().clone() for model in (flow, aero) for p in model.parameters()]
    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("scales must not construct optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    def predict(model, x, mask):
        assert not torch.is_grad_enabled()
        out = model(x)
        return out[:, :3], (out[:, 3:]*mask).sum((-2, -1))/mask.sum((-2, -1))
    def make(q, mask, a, b):
        return torch.cat((q, mask, a[:, :, None, None].expand_as(mask), b[:, :, None, None].expand_as(mask)), 1)
    with torch.set_grad_enabled(False):
        value = objective.field_force_rollout_objective(flow, aero, torch.ones(1, 3, 2, 2),
            torch.zeros(1, 100, 3, 2, 2), torch.ones(1, 1, 2, 2), torch.zeros(1, 101, 1),
            torch.zeros(1, 100, 4), predict, make, field_scale=1., force_scale=1., backward=False)
    assert value["raw_field"] > 0 and value["raw_force"] > 0
    for p, original in zip((p for model in (flow, aero) for p in model.parameters()), before, strict=True):
        assert p.grad is None and torch.equal(p, original)
    scales = runner.summarize_scales([value] * 1368)
    assert scales == dict(field=value["raw_field"], force=value["raw_force"])


def test_source_identity_failure_precedes_import(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_p029_control_aware_flow.py").write_bytes(SCRIPT.read_bytes())
    objective = scripts / "p029_control_aware_flow_objective.py"
    objective.write_text("raise AssertionError('must not import unchecked closure')")
    bad_base = scripts / "train_p028_flow_rollout.py"
    bad_base.write_text("raise AssertionError('must reject bad source before import')")
    value = dict(source_root=str(tmp_path), source_sha256={
        "scripts/train_p029_control_aware_flow.py": runner.sha(SCRIPT),
        "scripts/p029_control_aware_flow_objective.py": runner.sha(objective)})
    with pytest.raises(ValueError, match="file identity"):
        runner.dependencies(value)


def test_explicit_p029_save_schema_with_synthetic_official_api(tmp_path):
    # Emulates signatures and bytes only, NOT official checkpoint compatibility evidence.
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "FNO.0.1.mdlus").write_bytes(b"synthetic aero")
    (parent / "checkpoint.0.1.pt").write_bytes(b"synthetic aero state")
    inherited = ("schema_version", "architecture", "flow_architecture", "aerodynamic_architecture",
                 "config_sha256", "normalization_sha256", "precision_protocol", "input_sha256",
                 "history_input", "history_inventory", "history_state_module_sha256", "history_inference_module_sha256",
                 "aerodynamic_initial_model_sha256", "aerodynamic_initial_state_sha256")
    payload = {key: key for key in inherited}
    for role in ("flow", "aerodynamic"):
        payload[role] = dict(directory=role, model_file="FNO.0.1.mdlus", state_file="checkpoint.0.1.pt",
                             model_sha256="a"*64, state_sha256="b"*64, metadata_kind="parent")
    identity = SimpleNamespace(payload=payload, aerodynamic=SimpleNamespace(directory=parent))
    flow, aero = Flow(), Flow().eval().requires_grad_(False)
    output = tmp_path / "candidate"
    output.mkdir()
    result = dict(fixed_scales=dict(field=2., force=3.), source_spec=spec("train"), flow_terminal_tensor_sha256="toy")
    saved_metadata = {}
    def save(path, *, models, optimizer, epoch, metadata):
        path.mkdir()
        (path / "FNO.0.1.mdlus").write_bytes(b"synthetic flow")
        (path / "checkpoint.0.1.pt").write_bytes(b"synthetic optimizer")
        saved_metadata.update(metadata)
    def load(path, *, models, metadata_dict, device):
        if path.name == "flow":
            metadata_dict.update(saved_metadata)
        return 1
    runner.save_terminal(output, flow, aero, None, {}, identity, result, lambda cfg: Flow(), save, load, lambda m: "toy")
    manifest = json.loads((output / "dual_model_manifest.json").read_text())
    protocol = json.loads((output / "training_protocol.json").read_text())
    assert manifest["kind"] == "FC_P029_CONTROL_AWARE_FLOW_REPAIR"
    assert manifest["status"] == "FC_P029_DUAL_FNO_MANIFEST_VERIFIED"
    assert saved_metadata["status"] == "FC_P029_CONTROL_AWARE_FLOW_CHECKPOINT"
    assert manifest["flow"]["frozen"] is False and manifest["aerodynamic"]["frozen"] is True
    assert protocol["fixed_scales"] == manifest["fixed_scales"] == saved_metadata["fixed_scales"]
    assert saved_metadata["scales_receipt_sha256"] == spec()["scales_receipt"]["sha256"]
    assert manifest["training_protocol_sha256"] == runner.sha(output / "training_protocol.json")
    assert (output / "aerodynamic/FNO.0.1.mdlus").read_bytes() == b"synthetic aero"
