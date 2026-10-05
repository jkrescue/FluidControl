"""CPU software fixtures, not an accepted surrogate or scientific evaluation."""
import json

import pytest

import audit_dev30_validation_diagnostic as audit
import train_fcp013_independent_force_fno as trainer
from fluid_control import dual_fno
from test_dev30_validation_diagnostic import report, expected, segments


@pytest.fixture
def dual_case(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    files = {}
    for role, epoch in (("flow", 0), ("aerodynamic", 1)):
        directory = root / role
        directory.mkdir(parents=True)
        for key, name in (("model", f"FNO.0.{epoch}.mdlus"),
                          ("state", f"checkpoint.0.{epoch}.pt")):
            path = directory / name
            path.write_bytes(("software-fixture-" + role + key).encode())
            files[role + "_" + key] = path
    for module, model_key, state_key in (
        (trainer, "PARENT_MODEL_SHA", "PARENT_STATE_SHA"),
        (dual_fno, "FLOW_MODEL_SHA256", "FLOW_STATE_SHA256"),
    ):
        monkeypatch.setattr(module, model_key, audit.sha256(files["flow_model"]))
        monkeypatch.setattr(module, state_key, audit.sha256(files["flow_state"]))
    manifest = trainer.build_dual_manifest(root, files["aerodynamic_model"], files["aerodynamic_state"])
    identity = dual_fno.validate_dual_fno_manifest(manifest)
    value = report()
    value.update(checkpoint_dir="/workspace/dual/aerodynamic", checkpoint_epoch=1,
                 checkpoint_metadata={
                     "dual_fno": True, "manifest_sha256": identity.manifest_sha256,
                     "flow_model_sha256": identity.flow.model_sha256,
                     "flow_state_sha256": identity.flow.state_sha256,
                     "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
                     "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
                 })
    return value, files, manifest


def validate(value, files, kind=audit.P013_KIND):
    audit.validate_report_contract(value, expected(), candidate_kind=kind,
                                   checkpoint_dir=files["aerodynamic_model"].parent)


def test_only_explicit_dual_kind_accepts_verified_alias(dual_case):
    value, files, _ = dual_case
    validate(value, files)
    for kind in (None, "dev30_h20_development", "fc_p009_joint_force_row_calibrated_epoch0", "P013"):
        with pytest.raises(ValueError, match="runtime contract"):
            validate(value, files, kind)
    audit.validate_report_contract(report(), expected())
    assert audit.pooled_metrics(value, expected()) == audit.pooled_metrics(report(), expected())
    assert audit.strict_start0_ranking(segments(), expected())["strict_start0_cases"] == 10


@pytest.mark.parametrize("alias", ["/workspace/checkpoint", "/workspace/other/aerodynamic",
                                   "/workspace/dual/flow", "/workspace/dual/aerodynamic/../aerodynamic"])
def test_p013_rejects_every_other_literal_alias(dual_case, alias):
    value, files, _ = dual_case
    value["checkpoint_dir"] = alias
    with pytest.raises(ValueError, match="runtime contract"):
        validate(value, files)


@pytest.mark.parametrize("key", ["manifest_sha256", "flow_model_sha256", "flow_state_sha256",
                                 "aerodynamic_model_sha256", "aerodynamic_state_sha256"])
def test_report_must_bind_every_actual_dual_identity(dual_case, key):
    value, files, _ = dual_case
    value["checkpoint_metadata"][key] = "0" * 64
    with pytest.raises(ValueError, match="dual identity"):
        validate(value, files)


@pytest.mark.parametrize("key", ["flow_model", "flow_state", "aerodynamic_model", "aerodynamic_state"])
def test_changed_actual_checkpoint_bytes_are_rejected(dual_case, key):
    value, files, _ = dual_case
    files[key].write_bytes(b"changed-fixture")
    with pytest.raises(ValueError, match="SHA differs"):
        validate(value, files)


@pytest.mark.parametrize("fault", ["metadata", "epoch", "role", "missing_checkpoint", "manifest"])
def test_dual_flags_alone_cannot_admit_an_unbound_alias(dual_case, fault):
    value, files, manifest = dual_case
    checkpoint = files["aerodynamic_model"].parent
    if fault == "metadata":
        value["checkpoint_metadata"]["dual_fno"] = False
    elif fault == "epoch":
        value["checkpoint_epoch"] = 0
    elif fault == "role":
        checkpoint = files["flow_model"].parent
    elif fault == "missing_checkpoint":
        checkpoint = None
    else:
        content = json.loads(manifest.read_text())
        content["flow_parent_model_sha256"] = "0" * 64
        manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        audit.validate_report_contract(value, expected(), candidate_kind=audit.P013_KIND,
                                       checkpoint_dir=checkpoint)


@pytest.mark.parametrize("field,invalid", [("evaluation_data", "/workspace/other"),
                                          ("normalization_data", "/workspace/other"),
                                          ("split", "train"), ("action_scale", 1.0)])
def test_dual_alias_does_not_relax_other_runtime_checks(dual_case, field, invalid):
    value, files, _ = dual_case
    value[field] = invalid
    with pytest.raises(ValueError, match="runtime contract"):
        validate(value, files)
