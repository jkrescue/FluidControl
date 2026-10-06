"""Synthetic CPU engineering fixtures, not CFD/scientific evidence."""
import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "train_p028_flow_rollout.py"
if not SCRIPT.exists():
    SCRIPT = HERE.parent / "scripts/train_p028_flow_rollout.py"
spec = importlib.util.spec_from_file_location("p028_runner_tested", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_net = torch.nn.Module()
        self.decoder_net.final_layer = torch.nn.Module()
        self.decoder_net.final_layer.linear = torch.nn.Linear(3, 7)
        self.spectral = torch.nn.Parameter(torch.tensor([1 + 2j], dtype=torch.complex64))


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)


def gradients(model):
    for p in model.parameters():
        p.grad = torch.ones_like(p) * (2 + 3j if p.is_complex() else 2)


def execution_spec(mode="resource-probe"):
    return dict(status="FC_P028_EXECUTION_APPROVED", mode=mode, protocol=runner.protocol(),
                parent_manifest={"sha256": runner.PARENT_SHA}, config={"sha256": runner.CONFIG_SHA},
                train_audit={"sha256": runner.AUDIT_SHA}, resources={"allocator_fraction": .06})


def test_scope_mask_decay_moments_and_complex_norm():
    torch.manual_seed(4)
    model = Toy()
    opt = optimizer(model)
    saved = runner.preserve_rows(model)
    runner.check_optimizer(model, opt, 0)
    before = model.decoder_net.final_layer.linear.weight.detach().clone()
    gradients(model)
    audit = runner.optimizer_update(model, opt, saved, 1)
    assert audit["gradient_norms"]["spectral"] == pytest.approx(13**.5)
    assert audit["applied_clip_scale"] < 1
    for name, p in model.named_parameters():
        if name in runner.FINAL:
            assert torch.equal(p[3:], saved[name])
            assert torch.count_nonzero(p.grad[3:]) == 0
            assert torch.count_nonzero(opt.state[p]["exp_avg"][3:]) == 0
            assert torch.count_nonzero(opt.state[p]["exp_avg_sq"][3:]) == 0
    assert not torch.equal(model.decoder_net.final_layer.linear.weight[:3], before[:3])
    runner.check_optimizer(model, opt, 1)


def test_averaged_gradient_clipped_once_matches_independent_adam():
    torch.manual_seed(5)
    model = Toy()
    reference = copy.deepcopy(model)
    opt, refopt = optimizer(model), optimizer(reference)
    saved = runner.preserve_rows(model)
    for p in model.parameters():
        p.grad = sum(torch.ones_like(p) * (i + 1) for i in range(8)) / 8
    for p in reference.parameters():
        p.grad = torch.full_like(p, 4.5)
    runner.mask_rows(reference, refopt, runner.preserve_rows(reference))
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    refopt.step()
    runner.mask_rows(reference, refopt, saved, restore=True)
    runner.optimizer_update(model, opt, saved, 1)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["missing", "nan", "frozen"])
def test_finite_gradient_fail_closed(defect):
    model = Toy()
    gradients(model)
    parameter = next(model.parameters())
    if defect == "missing":
        parameter.grad = None
    elif defect == "nan":
        parameter.grad.fill_(float("nan"))
    else:
        parameter.requires_grad_(False)
    with pytest.raises(FloatingPointError):
        runner.finite_gradients(model)


def test_nonfinite_moments_and_wrong_optimizer_ownership():
    model = Toy()
    opt = optimizer(model)
    gradients(model)
    runner.optimizer_update(model, opt, runner.preserve_rows(model), 1)
    opt.state[next(model.parameters())]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        runner.check_optimizer(model, opt, 1)
    with pytest.raises(ValueError):
        runner.check_optimizer(Toy(), opt, 1)


def test_frozen_aero_proof():
    model = Toy().eval().requires_grad_(False)
    digest = lambda m: tuple(p.detach().numpy().tobytes() for p in m.parameters())
    expected = digest(model)
    runner.assert_frozen(model, expected, digest)
    next(model.parameters()).data.add_(1)
    with pytest.raises(RuntimeError):
        runner.assert_frozen(model, expected, digest)


def test_memory_estimate_complex_real_and_no_optimizer():
    model = Toy()
    estimate = runner.optimizer_memory_estimate(model)
    # Linear weight/bias float32 plus one complex64 coefficient.
    assert estimate["parameter_bytes"] == (7 * 3 + 7) * 4 + 8
    assert estimate["adam_first_second_moment_bytes"] == 2 * estimate["parameter_bytes"]
    assert estimate["optimizer_created"] is False
    assert estimate["optimizer_peak_measured"] is False


def test_precision_set_before_validation_and_parent_agreement():
    from types import SimpleNamespace
    calls = []
    fake = SimpleNamespace(set_float32_matmul_precision=lambda value: calls.append(value))
    def validate():
        assert calls == ["high"]
        calls.append("validated")
        return {"precision": "high"}
    assert runner.configure_precision(fake, validate, {"precision": "high"}) == {"precision": "high"}
    assert calls == ["high", "validated"]
    with pytest.raises(ValueError):
        runner.configure_precision(fake, lambda: {"precision": "high"}, {"precision": "highest"})


@pytest.mark.parametrize("defect", [None, "missing_hashes", "bad_hash", "mode", "windows", "parent", "source"])
def test_probe_receipt_semantics(defect):
    source = dict(parent_manifest={"sha256": runner.PARENT_SHA}, source_sha256={"a": "b"})
    probe = dict(status="FC_P028_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION", mode="resource-probe",
                 protocol=runner.protocol(), optimizer_steps=0, training_windows=1,
                 flow_initial_tensor_sha256="a" * 64, flow_terminal_tensor_sha256="a" * 64,
                 source_spec=copy.deepcopy(source))
    if defect == "missing_hashes":
        probe.pop("flow_initial_tensor_sha256")
        probe.pop("flow_terminal_tensor_sha256")
    elif defect == "bad_hash":
        probe["flow_initial_tensor_sha256"] = probe["flow_terminal_tensor_sha256"] = "x" * 64
    elif defect == "mode":
        probe["mode"] = "train"
    elif defect == "windows":
        probe["training_windows"] = 2
    elif defect == "parent":
        probe["source_spec"]["parent_manifest"]["sha256"] = "wrong"
    elif defect == "source":
        probe["source_spec"]["source_sha256"] = {}
    if defect is None:
        runner.validate_probe_receipt(probe, source)
    else:
        with pytest.raises(ValueError):
            runner.validate_probe_receipt(probe, source)


@pytest.mark.parametrize("memory,cuda,elapsed", [({"MemFree": 19.99, "MemAvailable": 70}, 40, 1),
                                                ({"MemFree": 40, "MemAvailable": 19}, 40, 1),
                                                ({"MemFree": 40, "MemAvailable": 70}, 19.99, 1),
                                                ({"MemFree": 40, "MemAvailable": 70}, 40, 901)])
def test_resource_guards(memory, cuda, elapsed):
    with pytest.raises(RuntimeError):
        runner.resource_limits(memory, cuda, elapsed, "resource-probe")


def test_protocol_and_actual_resource_receipt_required():
    valid = execution_spec()
    runner.validate_spec(valid, "resource-probe")
    invalid = copy.deepcopy(valid)
    invalid["protocol"]["learning_rate"] = 1.5625e-7
    with pytest.raises(ValueError):
        runner.validate_spec(invalid, "resource-probe")
    with pytest.raises(ValueError):
        runner.validate_spec(execution_spec("train"), "train")
    invalid = copy.deepcopy(valid)
    invalid["status"] = "PREPARATION_ONLY"
    with pytest.raises(ValueError):
        runner.validate_spec(invalid, "resource-probe")


def test_exclusive_file_and_symlink_check(tmp_path):
    p = tmp_path / "source"
    p.write_text("fixture")
    record = dict(path=str(p), sha256=runner.sha(p))
    assert runner.checked(record) == p
    alias = tmp_path / "alias"
    alias.symlink_to(p)
    with pytest.raises(ValueError):
        runner.checked(dict(path=str(alias), sha256=record["sha256"]))


def test_preparation_cli_does_not_read_spec(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["runner", "--spec", "/missing", "--spec-sha256", "not-a-sha",
                                    "--mode", "train", "--output", "/missing-output"])
    runner.main()
    assert "NO_MODEL_OR_DATA_ACCESS" in capsys.readouterr().out


def test_official_save_reload_adapter_contract(tmp_path):
    # Mock checkpoint APIs explicitly; this is not an official runtime smoke.
    model = Toy()
    aero = Toy().eval().requires_grad_(False)
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in ("FNO.0.1.mdlus", "checkpoint.0.1.pt"):
        (parent / name).write_text("fixture")
    from types import SimpleNamespace
    identity = SimpleNamespace(aerodynamic=SimpleNamespace(directory=parent), payload={
        "aerodynamic": {"model_file": "FNO.0.1.mdlus", "state_file": "checkpoint.0.1.pt", "model_sha256": "aero_model", "state_sha256": "aero_state"},
        "flow": {"model_file": "FNO.0.0.mdlus", "state_file": "checkpoint.0.0.pt", "model_sha256": "flow_model", "state_sha256": "flow_state"},
        "flow_architecture": {}, "aerodynamic_architecture": {}, "normalization_sha256": "fixture"})
    identity.payload.update(schema_version=1, architecture={}, config_sha256=runner.CONFIG_SHA,
                            precision_protocol={}, input_sha256={}, history_input={}, history_inventory={},
                            history_state_module_sha256="state_module", history_inference_module_sha256="inference",
                            aerodynamic_initial_model_sha256="P018_model", aerodynamic_initial_state_sha256="P018_state")
    digest = lambda m: str([(n, p.detach().tolist()) for n, p in m.named_parameters()])
    checkpoints = {}
    def save(path, *, models, optimizer, epoch, metadata):
        path.mkdir()
        checkpoints[str(path)] = (copy.deepcopy(models.state_dict()), metadata)
        for name in ("FNO.0.1.mdlus", "checkpoint.0.1.pt"):
            (path / name).write_text("mock-official")
    def load(path, *, models, metadata_dict, device):
        if Path(path).name == "aerodynamic":
            models.load_state_dict(aero.state_dict())
        else:
            state, meta = checkpoints[str(path)]
            models.load_state_dict(state)
            metadata_dict.update(meta)
        return 1
    result = dict(source_spec={"source_sha256": {}}, flow_terminal_tensor_sha256=digest(model),
                  frozen_aerodynamic_tensor_sha256=digest(aero), precision={})
    output = tmp_path / "output"
    output.mkdir()
    runner.save_terminal(output, model, aero, optimizer(model), None, identity, result,
                         lambda cfg: Toy(), save, load, digest)
    assert result["official_fresh_reload_verified"] is True
    assert (output / "dual_model_manifest.json").is_file()
    import json
    manifest = json.loads((output / "dual_model_manifest.json").read_text())
    assert "roles" not in manifest
    assert manifest["flow"]["frozen"] is False
    assert manifest["aerodynamic"]["frozen"] is True
    assert manifest["aerodynamic_parent_model_sha256"] == "aero_model"
    assert manifest["aerodynamic_initial_model_sha256"] == "P018_model"
    assert manifest["actual_learning_rate"] == 1e-5
    assert manifest["flow"]["metadata_kind"] == "FC_P028_FLOW_ROLLOUT_CHECKPOINT"


def test_actual_pinned_objective_average_and_step():
    path = Path(os.environ.get("P028_OBJECTIVE", str(HERE / "p028_flow_h10_objective.py")))
    if not path.exists():
        path = HERE.parent / "scripts/p028_flow_h10_objective.py"
    assert runner.sha(path) == "28a9a462f72a7ed5965fc97d22de47ef7fd12740edd66c3f505e822eb1ffd41a"
    module_spec = importlib.util.spec_from_file_location("actual_p028_objective", path)
    objective = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(objective)
    torch.manual_seed(6)
    model = Toy()
    reference = copy.deepcopy(model)
    opt, reference_opt = optimizer(model), optimizer(reference)
    saved = runner.preserve_rows(model)
    def make(q, mask, now, nxt):
        return q
    def predict(m, x, mask):
        # Deliberately nonzero real AND imaginary spectral derivative.
        z = (m.spectral * (1 + 2j)).real[0]
        raw = m.decoder_net.final_layer.linear(x[:, :, 0, 0]) * (.001 * z)
        return raw[:, :3, None, None], raw[:, 3:]
    def window(m, i):
        return objective.flow_rollout_objective(
            m, torch.ones(1, 3, 1, 1) * (i + 1) / 10,
            torch.zeros(1, 100, 3, 1, 1), torch.ones(1, 1, 1, 1),
            torch.zeros(1, 101, 1), predict, make, backward=True)
    result = objective.accumulate_eight_window_gradients(model, opt, range(8), lambda i: window(model, i))
    reference_opt.zero_grad(set_to_none=True)
    for i in range(8):
        window(reference, i)
    for p in reference.parameters():
        p.grad.div_(8)
    assert model.spectral.grad.imag.abs().item() > 0
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert result["windows"] == 8 and result["optimizer_step_performed"] is False
    runner.optimizer_update(model, opt, saved, 1)
    runner.optimizer_update(reference, reference_opt, saved, 1)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
