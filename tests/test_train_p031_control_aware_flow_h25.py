"""Synthetic CPU contract tests; no official model, HDF, or GPU access."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


STAGE = Path(__file__).resolve().parents[1]
SCRIPT = STAGE / "scripts/train_p031_control_aware_flow_h25.py"
definition = importlib.util.spec_from_file_location("p031_runner_tested", SCRIPT)
runner = importlib.util.module_from_spec(definition)
sys.modules[definition.name] = runner
definition.loader.exec_module(runner)


def spec(mode="resource-probe"):
    value = {
        "status": "FC_P031_EXECUTION_APPROVED",
        "mode": mode,
        "protocol": runner.protocol(),
        "parent_manifest": {"path": "parent.json", "sha256": runner.PARENT_SHA},
        "config": {"path": "config.yaml", "sha256": runner.CONFIG_SHA},
        "train_audit": {"path": "audit.json", "sha256": runner.AUDIT_SHA},
        "source_sha256": {"toy": "a" * 64},
        "resources": {"allocator_fraction": 0.06},
        "scales_receipt": {"path": "scales.json", "sha256": "b" * 64},
        "fixed_scales": dict(runner.FIXED_SCALES),
    }
    if mode == "train":
        value["resource_probe_receipt"] = {"path": "probe.json", "sha256": "c" * 64}
    return value


def scales_receipt():
    return {
        "status": "FC_P029_PARENT_SCALES_COMPLETE_NOT_ADMISSION",
        "mode": "scales", "optimizer_steps": 0, "optimizer_created": False,
        "model_saved": False, "scientific_admission": False,
        "flow_initial_tensor_sha256": "d" * 64,
        "flow_terminal_tensor_sha256": "d" * 64,
        "frozen_aerodynamic_tensor_sha256": "e" * 64,
        "training_windows": 1368, "sampler_order_sha256": runner.ORDER_SHA,
        "fixed_scales": dict(runner.FIXED_SCALES),
    }


def resource_receipt():
    return {
        "status": "FC_P031_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION",
        "mode": "resource-probe", "protocol": runner.protocol(),
        "optimizer_steps": 0, "optimizer_created": False,
        "model_saved": False, "scientific_admission": False,
        "flow_initial_tensor_sha256": "d" * 64,
        "flow_terminal_tensor_sha256": "d" * 64,
        "frozen_aerodynamic_tensor_sha256": "e" * 64,
        "training_windows": 1, "fixed_scales": dict(runner.FIXED_SCALES),
        "source_spec": spec("train"),
    }


@pytest.mark.parametrize("mode", runner.MODES)
def test_exact_h25_mode_contract(mode):
    runner.validate_spec(spec(mode), mode)
    bad = spec(mode)
    bad["protocol"]["horizon"] = 10
    with pytest.raises(ValueError, match="protocol"):
        runner.validate_spec(bad, mode)
    bad = spec(mode)
    bad["fixed_scales"]["field"] *= 2
    with pytest.raises(ValueError, match="scales"):
        runner.validate_spec(bad, mode)


def test_probe_needs_scale_receipt_and_train_needs_probe():
    value = spec("resource-probe")
    assert "resource_probe_receipt" not in value
    del value["scales_receipt"]
    with pytest.raises(ValueError, match="scales"):
        runner.validate_spec(value, "resource-probe")
    value = spec("train")
    del value["resource_probe_receipt"]
    with pytest.raises(ValueError, match="resource"):
        runner.validate_spec(value, "train")
    value = spec("resource-probe")
    value["resources"]["allocator_fraction"] = .45
    with pytest.raises(ValueError, match="allocator"):
        runner.validate_spec(value, "resource-probe")


def test_actual_p029_scale_receipt_is_fixed_and_fail_closed():
    value = scales_receipt()
    assert runner.validate_scales_receipt(value) == runner.FIXED_SCALES
    for key, replacement in (("training_windows", 1), ("optimizer_created", True),
                             ("sampler_order_sha256", "0" * 64)):
        bad = copy.deepcopy(value)
        bad[key] = replacement
        with pytest.raises(ValueError, match="P029 scales"):
            runner.validate_scales_receipt(bad)
    bad = copy.deepcopy(value)
    bad["fixed_scales"]["force"] *= 2
    with pytest.raises(ValueError, match="P029 scales"):
        runner.validate_scales_receipt(bad)


def test_p031_resource_receipt_binds_protocol_and_source():
    value = resource_receipt()
    runner.validate_resource_receipt(value, spec("train"))
    for damage in ("horizon", "source", "steps", "save"):
        bad = copy.deepcopy(value)
        if damage == "horizon":
            bad["protocol"]["horizon"] = 10
        elif damage == "source":
            bad["source_spec"]["source_sha256"] = {}
        elif damage == "steps":
            bad["optimizer_steps"] = 1
        else:
            bad["model_saved"] = True
        with pytest.raises(ValueError, match="resource receipt"):
            runner.validate_resource_receipt(bad, spec("train"))


def test_resource_limits_and_fixed_budget():
    protocol = runner.protocol()
    assert (protocol["horizon"], protocol["training_windows"],
            protocol["optimizer_steps"], protocol["accumulation_windows"]) == (25, 1368, 171, 8)
    runner.resource_limits({"MemFree": 30, "MemAvailable": 50}, 30, 0,
                           "resource-probe", True)
    with pytest.raises(RuntimeError):
        runner.resource_limits({"MemFree": 19.9, "MemAvailable": 50}, 30, 1,
                               "resource-probe")
    with pytest.raises(RuntimeError):
        runner.resource_limits({"MemFree": 30, "MemAvailable": 50}, 30, 901,
                               "resource-probe")
    runner.resource_limits({"MemFree": 30, "MemAvailable": 50}, 30, 901, "train")


def test_production_explicit_h25_and_probe_no_optimizer_or_save():
    source = SCRIPT.read_text()
    assert "backward=True, horizon=25" in source
    start = source.index('if mode == "resource-probe":', source.index("records, observed"))
    probe = source[start:source.index("else:", start)]
    assert "torch.optim.AdamW" not in probe and "save_terminal" not in probe


def test_source_identity_checks_new_runner_before_import(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    copied = scripts / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())
    objective = scripts / "p029_control_aware_flow_objective.py"
    objective.write_text("raise AssertionError('unchecked objective import')")
    value = {"source_root": str(tmp_path), "source_sha256": {
        "scripts/train_p031_control_aware_flow_h25.py": runner.sha(copied),
        "scripts/p029_control_aware_flow_objective.py": "0" * 64,
    }}
    with pytest.raises(ValueError, match="file identity"):
        runner.dependencies(value)
