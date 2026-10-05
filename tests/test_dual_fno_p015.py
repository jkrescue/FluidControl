"""Project identity tests; synthetic checkpoints are not scientific evidence."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_dual_fno import MODULE, manifest_fixture


def p015_fixture(tmp_path, monkeypatch):
    path, payload = manifest_fixture(tmp_path, monkeypatch)
    payload.update(status=MODULE.P015_MANIFEST_STATUS, kind=MODULE.P015_SYSTEM_KIND,
                   training_experiment="FC-P015", accumulation_windows=8,
                   training_windows=1368, optimizer_steps=171)
    payload["aerodynamic"]["metadata_kind"] = MODULE.P015_AERO_KIND
    path.write_text(json.dumps(payload))
    return path, payload


def test_p015_exact_manifest(tmp_path, monkeypatch):
    path, _ = p015_fixture(tmp_path, monkeypatch)
    identity = MODULE.validate_dual_fno_manifest(path, expected_sha256=MODULE.sha256(path))
    assert identity.aerodynamic.metadata_kind == MODULE.P015_AERO_KIND
    assert identity.payload["optimizer_steps"] == 171


@pytest.mark.parametrize("key,value", [
    ("status", MODULE.MANIFEST_STATUS), ("kind", "UNREVIEWED_EXPERIMENT"),
    ("training_experiment", "FC-P013"), ("accumulation_windows", 1),
    ("training_windows", 171), ("optimizer_steps", 1368),
])
def test_p015_rejects_mixed_or_unknown_identity(tmp_path, monkeypatch, key, value):
    path, payload = p015_fixture(tmp_path, monkeypatch)
    payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        MODULE.validate_dual_fno_manifest(path)


def test_p015_rejects_p013_checkpoint_kind(tmp_path, monkeypatch):
    path, payload = p015_fixture(tmp_path, monkeypatch)
    payload["aerodynamic"]["metadata_kind"] = MODULE.AERO_KIND
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="epoch/kind"):
        MODULE.validate_dual_fno_manifest(path)


@pytest.mark.parametrize("steps,accept", [(171, True), (1368, False)])
def test_p015_loaded_metadata_steps(tmp_path, monkeypatch, steps, accept):
    path, _ = p015_fixture(tmp_path, monkeypatch)
    import fluid_control.calibrated_checkpoint as calibrated
    monkeypatch.setattr(calibrated, "validate_calibrated_epoch_zero", lambda *a, **kw: None)
    monkeypatch.setattr(MODULE, "validate_runtime_precision", lambda: MODULE.PRECISION_PROTOCOL)
    cfg = SimpleNamespace(model=SimpleNamespace(**MODULE.ARCHITECTURE))

    def loader(directory, *, models, metadata_dict, device):
        if directory.name == "flow":
            return 0
        metadata_dict.update(status=MODULE.P015_AERO_KIND, checkpoint_epoch=1,
            flow_parent_model_sha256=MODULE.FLOW_MODEL_SHA256,
            flow_parent_state_sha256=MODULE.FLOW_STATE_SHA256,
            aerodynamic_initial_model_sha256=MODULE.FLOW_MODEL_SHA256,
            aerodynamic_initial_state_sha256=MODULE.FLOW_STATE_SHA256,
            training_experiment="FC-P015", accumulation_windows=8,
            training_windows=1368, optimizer_steps=steps,
            selection_performed=False, validation_accessed=False,
            frozen_test_accessed=False, ppo_executed=False)
        return 1

    def run():
        return MODULE.load_dual_fno(path, cfg, "cpu", build_model=lambda _: torch.nn.Linear(1, 1),
                                   load_checkpoint=loader, expected_manifest_sha256=MODULE.sha256(path))
    if accept:
        adapter, identity = run()
        assert identity.payload["kind"] == MODULE.P015_SYSTEM_KIND
        assert all(not parameter.requires_grad for parameter in adapter.parameters())
    else:
        with pytest.raises(ValueError, match="checkpoint metadata"):
            run()
