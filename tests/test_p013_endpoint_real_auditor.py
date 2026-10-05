"""CPU integration with the real numerical endpoint auditor; synthetic metrics.

No FNO inference, optimization, CFD, or scientific admission is performed.
Only the separate dual-provenance verifier is stubbed; endpoint.audit and its
predeclaration, checkpoint, runtime, force, and action checks execute unchanged.
"""
import copy
import json
from pathlib import Path

import pytest

from test_full40_validation_gate import MODULE as endpoint, fixtures
from test_full40_canonical_ppo_dry_run import MODULE as canonical
import fluid_control.dual_control_contract as dual_contract


@pytest.mark.parametrize("passes", [True, False])
def test_path_view_preserves_real_endpoint_numerics_and_source_bytes(tmp_path, monkeypatch, passes):
    repo = Path(__file__).resolve().parents[1]
    source_predecl = repo / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    if not source_predecl.is_file():
        pytest.skip("exact original predeclaration fixture is unavailable")
    assert endpoint.sha256(source_predecl) == endpoint.PREDECLARATION_SHA256
    predecl = tmp_path / "predeclaration.json"
    predecl.write_bytes(source_predecl.read_bytes())
    expected, report, segments = fixtures()
    assert endpoint.validate_predeclaration(predecl) == expected
    if not passes:
        # Deliberately violate actual fixed force and action criteria.
        for case in report["cases"]:
            case["horizons"]["100"]["total_drag_rmse"] = 0.8
            case["horizons"]["100"]["total_drag_mae"] = 0.4
        for row in segments["segments"]:
            if row["start"] == 0:
                row["predicted_total_drag"] = 4.0 - row["target_total_drag"]

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    data = tmp_path / "runtime/full40"
    write(data / "manifest.json", {"profile": endpoint.PROFILE, "max_abs_omega": 0.75,
                                   "trajectory_counts": {"train": 20, "validation": 10, "frozen_test": 10}})
    write(data / "normalization.json", {"fixture": "not scientific normalization"})
    checkpoint = tmp_path / "runtime/candidate/aerodynamic"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "FNO.0.1.mdlus"
    model.write_bytes(b"synthetic model identity; no inference is performed")
    config = tmp_path / "evaluation.yaml"
    config.write_text("synthetic configuration identity\n")
    image = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
    binding = {
        "dual_manifest_sha256": "a" * 64, "flow_model_sha256": "b" * 64,
        "flow_state_sha256": "c" * 64, "checkpoint_sha256": endpoint.sha256(model),
        "checkpoint_state_sha256": "d" * 64, "posteval_receipt_sha256": "e" * 64,
    }
    report.update(
        checkpoint_epoch=1, checkpoint_dir="/workspace/dual/aerodynamic",
        evaluation_data="/workspace/devdata", normalization_data="/workspace/devdata",
        force_channels=["front_cd", "front_cl", "rear_cd", "rear_cl"],
        action_scale=0.75, evaluation_action_limit=0.75,
        checkpoint_metadata={
            "dual_fno": True, "manifest_sha256": binding["dual_manifest_sha256"],
            "flow_model_sha256": binding["flow_model_sha256"],
            "flow_state_sha256": binding["flow_state_sha256"],
            "aerodynamic_model_sha256": binding["checkpoint_sha256"],
            "aerodynamic_state_sha256": binding["checkpoint_state_sha256"],
        },
    )
    posteval = tmp_path / "posteval"
    original_report = write(posteval / "validation10/evaluation.json", report)
    original_segments = write(posteval / "validation10/segments.json", segments)
    # Independently audit the same metrics with accessible runtime paths.
    baseline_report = copy.deepcopy(report)
    baseline_report.update(checkpoint_dir=str(checkpoint), evaluation_data=str(data),
                           normalization_data=str(data))
    baseline_path = write(tmp_path / "baseline_report.json", baseline_report)
    baseline = endpoint.audit(baseline_path, original_segments, predecl, checkpoint,
                              data, config, image)
    assert baseline["joint_terminal_readiness"] is passes
    assert baseline["status"].endswith("_PASS" if passes else "_FAIL")
    # These are the only two path-derived output identities in the real auditor.
    original_gate_value = {**baseline, "checkpoint_dir": report["checkpoint_dir"],
                           "report_sha256": endpoint.sha256(original_report)}
    original_gate = write(posteval / "validation10/endpoint_gate.json", original_gate_value)
    paths = [original_report, original_segments, original_gate]
    receipt = write(posteval / "receipt.json", {"sha256": {
        str(path.relative_to(posteval)): endpoint.sha256(path) for path in paths}})
    binding["posteval_receipt_sha256"] = endpoint.sha256(receipt)
    def verify(**kwargs):
        assert kwargs["expected_checkpoint_sha256"] == endpoint.sha256(model)
        assert kwargs["expected_posteval_receipt_sha256"] == endpoint.sha256(receipt)
        return binding
    monkeypatch.setattr(dual_contract, "verify_dual_control_binding", verify)
    before = {path: path.read_bytes() for path in [*paths, receipt, predecl, model, config]}
    result, mapping = canonical.recompute_p013_endpoint_view(
        dual_options={
            "dual_fno_manifest": tmp_path / "manifest.fixture",
            "expected_dual_fno_manifest_sha256": binding["dual_manifest_sha256"],
            "dual_training_config": tmp_path / "training.fixture",
            "dual_posteval_receipt": receipt,
            "expected_dual_posteval_receipt_sha256": binding["posteval_receipt_sha256"],
        },
        gate_module=endpoint, validation_gate=original_gate, validation_report=original_report,
        validation_segments=original_segments, predeclaration=predecl, checkpoint_dir=checkpoint,
        data=data, config=config, image_id=image,
    )
    assert result == original_gate_value
    assert result["status"] == baseline["status"]
    assert result["h100_force_gate"] == baseline["h100_force_gate"]
    assert result["h100_start0_action_difference"] == baseline["h100_start0_action_difference"]
    assert result["joint_terminal_readiness"] is passes
    assert mapping["numerical_evidence_changed"] is False
    assert mapping["source_sha256"]["validation10/evaluation.json"] == endpoint.sha256(original_report)
    assert all(path.read_bytes() == content for path, content in before.items())
