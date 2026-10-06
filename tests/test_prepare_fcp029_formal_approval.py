"""Synthetic pending drafts only; no candidate/model/HDF payload reads."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_prepare_fcp028_formal_approval import module as m


def fixture(tmp_path, monkeypatch):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    runner = write("orchestration/run_fcp029_posteval.py", "fixture entry")
    shared = write("orchestration/run_fcp028_posteval.py", "fixture shared")
    profiles = write("orchestration/flow_repair_profiles.py", "fixture profiles")
    numerical = write("original.py", "numerical fixture")
    monkeypatch.setattr(m, "NUMERICAL_RUNNER_SHA", m.sha(numerical))
    source_map = {"src/fluid_control/dual_fno.py": "a" * 64}
    source = write("source.json", dict(status="FC_P026_FORMAL_SOURCE_CHAIN_FROZEN",
                   numerical_base_commit=m.BASE, training_config_sha256=m.CONFIG_SHA,
                   final_files_sha256=source_map, overlay_sha256=source_map))
    runtime_map = {"src/fluid_control/dual_fno.py": "a" * 64,
                   **{f"synthetic{i}.py": "b" * 64 for i in range(11)}}
    runtime = write("runtime.json", runtime_map)
    freeze = write("freeze.json", dict(status="FC_P029_FORMAL_SOURCE_FREEZE_COMPLETE",
                   final_numerical_files_sha256=source_map,
                   external_orchestration={"scripts/" + p.name: m.sha(p) for p in (runner,shared,profiles)}))
    manifest = write("candidate/dual_model_manifest.json", dict(kind="FC_P029_CONTROL_AWARE_FLOW_REPAIR",
                     scientific_admission=False, parent_manifest_sha256="c" * 64))
    mapping = {"candidate/" + n: "d" * 64 for n in m.FILES}
    mapping["candidate/dual_model_manifest.json"] = m.sha(manifest)
    unit, inv = "fluid-control-fcp029-flow-train-20261006.service", "e" * 32
    common = dict(candidate_sha256=mapping, scientific_admission=False, ppo_authorized=False,
                  tensor_sha256={"flow": "1" * 64, "aerodynamic": "2" * 64},
                  training_protocol_sha256=mapping["candidate/training_protocol.json"],
                  dual_manifest_sha256=mapping["candidate/dual_model_manifest.json"],
                  candidate_result_sha256=mapping["candidate/result.json"])
    audit = write("audit.json", dict(common, status="FC_P029_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
                  training_unit=unit, training_invocation=inv, actual_optimizer_steps=171,
                  actual_training_windows=1368, role_loader_sha256="a" * 64,
                  terminal_evidence=dict(LoadState="loaded",ActiveState="active",SubState="exited",Result="success",
                                         ExecMainCode="1",ExecMainStatus="0",MainPID="0",InvocationID=inv)))
    reload = write("reload.json", dict(common, status="FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
                   candidate_audit_sha256=m.sha(audit), official_dual_reload_verified=True,
                   forward_performed=False, optimizer_created=False, model_saved=False, gpu_used=False,
                   runtime_source_sha256=runtime_map, runtime_source_manifest_sha256=m.sha(runtime)))
    return SimpleNamespace(repo=tmp_path, runner=runner, numerical_runner=numerical,
                           source_receipt=source, freeze_receipt=freeze, runtime_manifest=runtime,
                           candidate_audit=audit, official_reload=reload, candidate=manifest.parent,
                           training_unit=unit,training_invocation=inv,output_relative_directory="formal",
                           runner_sha256=m.sha(runner),source_sha256=m.sha(source),
                           freeze_sha256=m.sha(freeze),runtime_sha256=m.sha(runtime))


def test_only_pending_with_exact_external_three(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = m.build(args, "FC-P029")
    assert result["status"] == "FC_P029_FORMAL_EVALUATION_APPROVAL_PENDING_LEAD_REVIEW"
    assert result["formal_evaluation_authorized"] is False
    assert result["independent_terminal_audit"]["reviewed_by_lead"] is False
    assert result["official_dual_reload"]["reviewed_by_lead"] is False
    assert set(result["orchestration_sha256"]) == {"run_fcp029_posteval.py", "run_fcp028_posteval.py", "flow_repair_profiles.py"}


@pytest.mark.parametrize("change", ["status", "invocation", "audit_sha", "tensor", "runtime", "file_map", "source"])
def test_wrong_or_stale_proofs_rejected(tmp_path, monkeypatch, change):
    args = fixture(tmp_path, monkeypatch)
    payload = json.loads(args.official_reload.read_text())
    if change == "status": payload["status"] = "FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"
    if change == "audit_sha": payload["candidate_audit_sha256"] = "0" * 64
    if change == "tensor": payload["tensor_sha256"]["flow"] = "0" * 64
    if change == "runtime": payload["runtime_source_manifest_sha256"] = "0" * 64
    if change == "file_map": payload["candidate_sha256"].pop("candidate/result.json")
    if change == "invocation": args.training_invocation = "f" * 32
    if change == "source": args.runner_sha256 = "0" * 64
    args.official_reload.write_text(json.dumps(payload))
    with pytest.raises(ValueError): m.build(args, "FC-P029")
