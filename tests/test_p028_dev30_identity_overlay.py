"""Identity-only P028 coverage; scientific diagnostic math is unchanged."""

import sys
from types import SimpleNamespace

import audit_dev30_validation_diagnostic as audit
from test_dev30_validation_diagnostic import expected, report


def test_p028_exact_kind_routes_p026_k1_history_without_metric_changes(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "candidate" / "aerodynamic"
    checkpoint.mkdir(parents=True)
    identity = SimpleNamespace(
        manifest_sha256="1" * 64,
        payload={
            "kind": audit.P028_KIND,
            "history_input": {"profile": "p026_k1"},
        },
        flow=SimpleNamespace(model_sha256="2" * 64, state_sha256="3" * 64),
        aerodynamic=SimpleNamespace(
            directory=checkpoint.resolve(),
            model_sha256="4" * 64,
            state_sha256="5" * 64,
        ),
    )
    fake = SimpleNamespace(validate_dual_fno_manifest=lambda path: identity)
    monkeypatch.setitem(sys.modules, "fluid_control.dual_fno", fake)
    value = report()
    value.update(
        checkpoint_dir="/workspace/dual/aerodynamic",
        checkpoint_epoch=1,
        fno_history_profile="p026_k1",
        checkpoint_metadata={
            "dual_fno": True,
            "manifest_sha256": identity.manifest_sha256,
            "flow_model_sha256": identity.flow.model_sha256,
            "flow_state_sha256": identity.flow.state_sha256,
            "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
            "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        },
    )
    audit.validate_report_contract(
        value,
        expected(),
        candidate_kind=audit.P028_KIND,
        checkpoint_dir=checkpoint,
    )
    assert audit.pooled_metrics(value, expected()) == audit.pooled_metrics(
        report(), expected()
    )


def test_p028_identity_overlay_rejects_wrong_kind_or_history(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate" / "aerodynamic"
    checkpoint.mkdir(parents=True)
    identity = SimpleNamespace(
        manifest_sha256="1" * 64,
        payload={"kind": audit.P028_KIND, "history_input": {"profile": "p026_k4"}},
        flow=SimpleNamespace(model_sha256="2" * 64, state_sha256="3" * 64),
        aerodynamic=SimpleNamespace(
            directory=checkpoint.resolve(), model_sha256="4" * 64, state_sha256="5" * 64
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fluid_control.dual_fno",
        SimpleNamespace(validate_dual_fno_manifest=lambda path: identity),
    )
    value = report()
    value.update(
        checkpoint_dir="/workspace/dual/aerodynamic",
        checkpoint_epoch=1,
        fno_history_profile="p026_k1",
        checkpoint_metadata={
            "dual_fno": True,
            "manifest_sha256": identity.manifest_sha256,
            "flow_model_sha256": identity.flow.model_sha256,
            "flow_state_sha256": identity.flow.state_sha256,
            "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
            "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        },
    )
    try:
        audit.validate_report_contract(
            value,
            expected(),
            candidate_kind=audit.P028_KIND,
            checkpoint_dir=checkpoint,
        )
    except ValueError as error:
        assert "history profile differs" in str(error)
    else:
        raise AssertionError("wrong P028 history profile was accepted")
