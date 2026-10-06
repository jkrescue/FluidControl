"""Synthetic CPU identities only; no official model or research data access."""
import copy
import json
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import test_p028_dual_loader as legacy
from test_run_fcp028_posteval import MODULE as formal, BASE
from flow_repair_profiles import repair_profile, validate_p029_result_binding


def fixture(tmp_path, monkeypatch):
    path, payload = legacy.p028_fixture(tmp_path, monkeypatch)
    m = legacy.MODULE
    protocol = copy.deepcopy(payload["training_semantics"])
    protocol.update(
        experiment="FC-P029", force_loss=True,
        objective=repair_profile("FC-P029").objective,
        field_weight=.5, force_weight=.5, scale_windows=1368,
        force_timing="aero_current_state_and_current_next_action_predicts_next_force",
        fixed_scales={"field": .125, "force": .25}, scales_receipt_sha256="a" * 64,
    )
    payload.update(status=m.P029_MANIFEST_STATUS, kind=m.P029_SYSTEM_KIND,
                   training_experiment="FC-P029", training_semantics=protocol,
                   fixed_scales=protocol["fixed_scales"], scales_receipt_sha256="a" * 64)
    payload["flow"]["metadata_kind"] = m.P029_FLOW_KIND
    write(path, payload)
    return path, payload


def write(path, payload):
    protocol_path = path.parent / "training_protocol.json"
    protocol_path.write_text(json.dumps(payload["training_semantics"]))
    payload["training_protocol_sha256"] = legacy.MODULE.sha256(protocol_path)
    path.write_text(json.dumps(payload))


def test_p029_explicit_role_protocol(tmp_path, monkeypatch):
    path, payload = fixture(tmp_path, monkeypatch)
    identity = legacy.MODULE.validate_dual_fno_manifest(path)
    assert identity.payload["kind"] == "FC_P029_CONTROL_AWARE_FLOW_REPAIR"
    assert identity.flow.epoch == identity.aerodynamic.epoch == 1
    assert identity.payload["aerodynamic"]["metadata_kind"] == legacy.MODULE.P026_K1_AERO_KIND


@pytest.mark.parametrize("change", [
    lambda p: p.update(kind=legacy.MODULE.P028_SYSTEM_KIND),
    lambda p: p.update(status=legacy.MODULE.P028_MANIFEST_STATUS),
    lambda p: p["flow"].update(metadata_kind=legacy.MODULE.P028_FLOW_KIND),
    lambda p: p["training_semantics"].update(force_loss=False),
    lambda p: p["training_semantics"].update(field_weight=.4),
    lambda p: p["training_semantics"].update(scale_windows=44),
    lambda p: p["training_semantics"].update(future_truth_inputs=True),
    lambda p: p["training_semantics"].update(future_truth_inputs=0),
    lambda p: p["training_semantics"].update(horizon=10.0),
    lambda p: p.update(scales_receipt_sha256="not-a-hash"),
    lambda p: p["fixed_scales"].update(force=0),
    lambda p: p["fixed_scales"].update(field=True),
    lambda p: p["fixed_scales"].update(field=float("nan")),
    lambda p: p["fixed_scales"].update(field=float("inf")),
])
def test_rejects_cross_profile_or_changed_objective(tmp_path, monkeypatch, change):
    path, payload = fixture(tmp_path, monkeypatch)
    change(payload)
    write(path, payload)
    with pytest.raises(ValueError):
        legacy.MODULE.validate_dual_fno_manifest(path)


def test_original_numerical_commands_only_identity_differs():
    old = formal.commands(BASE, "a" * 64, "b" * 64)
    new = formal.commands(BASE, "a" * 64, "b" * 64, "FC-P029")
    edits = []
    for (on, og, oc), (nn, ng, nc) in zip(old, new, strict=True):
        assert (on, og) == (nn, ng)
        assert len(oc) == len(nc)
        edits += [(on, a, b) for a, b in zip(oc, nc) if a != b]
    assert edits == [("validation_diagnostic", legacy.MODULE.P028_SYSTEM_KIND,
                      legacy.MODULE.P029_SYSTEM_KIND)]


@pytest.mark.parametrize("bad", ["FC-P026", "p029", None, True])
def test_no_implicit_experiment_selection(bad):
    with pytest.raises(ValueError):
        repair_profile(bad)


def test_saved_scale_result_binding(tmp_path, monkeypatch):
    _, manifest = fixture(tmp_path, monkeypatch)
    protocol = manifest["training_semantics"]
    base = {k: v for k, v in protocol.items() if k not in ("fixed_scales", "scales_receipt_sha256")}
    result = {"protocol": base, "fixed_scales": copy.deepcopy(protocol["fixed_scales"]),
              "source_spec": {"protocol": base, "scales_receipt": {"sha256": "a" * 64}}}
    validate_p029_result_binding(result, protocol, manifest)
    result["fixed_scales"]["field"] *= 2
    with pytest.raises(ValueError):
        validate_p029_result_binding(result, protocol, manifest)


@pytest.mark.parametrize("mutation", [None, "flow_kind", "scale", "aero_kind"])
def test_loaded_checkpoint_role_metadata(tmp_path, monkeypatch, mutation):
    path, payload = fixture(tmp_path, monkeypatch)
    m = legacy.MODULE

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value[:, :1].expand(-1, 7, -1, -1)

    def load(directory, *, models, metadata_dict, device):
        common = dict(checkpoint_epoch=1, flow_parent_model_sha256=m.FLOW_MODEL_SHA256,
                      flow_parent_state_sha256=m.FLOW_STATE_SHA256,
                      optimizer_steps=171, accumulation_windows=8, training_windows=1368,
                      training_protocol_file="training_protocol.json")
        if directory.name == "flow":
            common.update(status=m.P029_FLOW_KIND, training_experiment="FC-P029",
                          training_protocol_sha256=payload["training_protocol_sha256"],
                          actual_learning_rate=m.P028_LEARNING_RATE,
                          parent_manifest_sha256=m.P028_PARENT_MANIFEST_SHA256,
                          aerodynamic_parent_model_sha256=m.P028_AERO_PARENT_MODEL_SHA256,
                          aerodynamic_parent_state_sha256=m.P028_AERO_PARENT_STATE_SHA256,
                          fixed_scales=copy.deepcopy(payload["fixed_scales"]),
                          scales_receipt_sha256=payload["scales_receipt_sha256"])
            if mutation == "flow_kind":
                common["status"] = m.P028_FLOW_KIND
            if mutation == "scale":
                common["fixed_scales"]["field"] *= 2
        else:
            common.update(status=m.P026_K1_AERO_KIND, training_experiment="FC-P026",
                          aerodynamic_initial_model_sha256=m.P026_AERO_INITIAL_MODEL_SHA256,
                          aerodynamic_initial_state_sha256=m.P026_AERO_INITIAL_STATE_SHA256,
                          actual_learning_rate=m.P026_LEARNING_RATE,
                          history_profile="p026_k1", history_k=1, model_in_channels=6,
                          training_protocol_sha256=m.P028_AERO_PROTOCOL_SHA256,
                          history_state_module_sha256=m.P026_HISTORY_STATE_SHA256,
                          history_inference_module_sha256=m.P026_HISTORY_INFERENCE_SHA256,
                          sampler_order_sha256=m.P026_ORDER_SHA256,
                          history_inventory=m._p026_inventory(), selection_performed=False,
                          validation_accessed=False, frozen_test_accessed=False, ppo_executed=False)
            if mutation == "aero_kind":
                common["status"] = m.P029_FLOW_KIND
        metadata_dict.update(common)
        return 1

    monkeypatch.setattr(m, "validate_runtime_precision", lambda: m.PRECISION_PROTOCOL)
    kwargs = dict(build_model=lambda cfg: Toy(), load_checkpoint=load)
    if mutation:
        with pytest.raises(ValueError):
            m.load_dual_fno(path, legacy._cfg(), torch.device("cpu"), **kwargs)
    else:
        adapter, identity = m.load_dual_fno(path, legacy._cfg(), torch.device("cpu"), **kwargs)
        assert identity.payload["training_experiment"] == "FC-P029"
        assert adapter(torch.ones(1, 6, 2, 2)).shape == (1, 7, 2, 2)
