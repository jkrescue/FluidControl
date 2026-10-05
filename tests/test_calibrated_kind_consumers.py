import importlib

import pytest


@pytest.mark.parametrize("module_name", ["audit_dev30_validation_diagnostic", "audit_full40_validation_gate"])
@pytest.mark.parametrize("kind", ["FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE", "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"])
def test_auditor_forwards_exact_kind(tmp_path, monkeypatch, module_name, kind):
    module = importlib.import_module(module_name)
    helper = importlib.import_module("fluid_control.calibrated_checkpoint")
    seen = {}

    def verify(path, epoch, **kwargs):
        seen.update(kwargs)
        return {"calibrated_epoch_zero": True, "candidate_kind": kwargs["expected_kind"]}

    monkeypatch.setattr(helper, "validate_calibrated_epoch_zero", verify)
    (tmp_path / "FNO.0.0.mdlus").write_bytes(b"identity fixture")
    result = module.validate_checkpoint(
        {"checkpoint_epoch": 0, "checkpoint_dir": str(tmp_path)},
        tmp_path,
        allow_calibrated_epoch_zero=True,
        expected_calibrated_model_sha256="a" * 64,
        expected_calibrated_state_sha256="b" * 64,
        expected_calibrated_kind=kind,
    )
    assert seen == {"allow": True, "expected_model_sha256": "a" * 64,
                    "expected_state_sha256": "b" * 64, "expected_kind": kind}
    assert result["candidate_kind"] == kind
