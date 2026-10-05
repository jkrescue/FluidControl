"""CPU-only contract fixtures, not model or heldout evaluation."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT_DIR = (
    HERE.parent / "scripts"
    if (HERE.parent / "scripts/run_fcp026_posteval.py").is_file()
    else HERE
)
spec = importlib.util.spec_from_file_location(
    "p026_posteval_tested", SCRIPT_DIR / "run_fcp026_posteval.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def candidate(tmp_path, k=4):
    history = dict(
        profile=f"p026_k{k}",
        history_length=k,
        flow_input_channels=6,
        aerodynamic_input_channels=6 if k == 1 else 18,
        future_state_inputs=False,
        future_force_inputs=False,
    )
    protocol = dict(
        training_experiment="FC-P026",
        training_windows=1368,
        optimizer_steps=171,
        accumulation_windows=8,
        learning_rate=1.5625e-7,
        validation_accessed=False,
        frozen_test_accessed=False,
        selection_performed=False,
        history_input=history,
        sampler_order_sha256="a" * 64,
    )
    (tmp_path / "training_protocol.json").write_text(json.dumps(protocol))
    digest = m.sha(tmp_path / "training_protocol.json")
    manifest = dict(
        status=f"FC_P026_K{k}_DUAL_FNO_MANIFEST_VERIFIED",
        kind=f"FC_P026_K{k}_HISTORY_FORCE_FNO",
        training_experiment="FC-P026",
        training_protocol_sha256=digest,
        history_input=history,
    )
    for role, epoch in (("flow", 0), ("aerodynamic", 1)):
        (tmp_path / role).mkdir()
        model = tmp_path / role / f"FNO.0.{epoch}.mdlus"
        state = tmp_path / role / f"checkpoint.0.{epoch}.pt"
        model.write_bytes(b"fixture model")
        state.write_bytes(b"fixture state")
        manifest[role] = dict(model_sha256=m.sha(model), state_sha256=m.sha(state))
    result = dict(
        status=f"FC_P026_K{k}_TRAINING_COMPLETE_NOT_ADMISSION",
        history_k=k,
        optimizer_steps=171,
        training_windows=1368,
        protocol_sha256=digest,
        official_fresh_reload_verified=True,
        scientific_admission=False,
        records=[dict(update=i, consumed_windows=8 * i) for i in range(1, 172)],
        sampler_order_sha256="a" * 64,
    )
    for name, value in (
        ("result.json", result),
        ("dual_model_manifest.json", manifest),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    a = dict(
        history_k=k,
        candidate_sha256={
            str(p.relative_to(tmp_path)): m.sha(p)
            for p in tmp_path.rglob("*")
            if p.is_file()
        },
    )
    return a


@pytest.mark.parametrize("k", [1, 4])
def test_candidate_exact_profile(tmp_path, k):
    a = candidate(tmp_path, k)
    assert m.candidate_contract(tmp_path, a)["history_input"]["history_length"] == k


@pytest.mark.parametrize(
    "change",
    ["missing", "wrong_hash", "fixture", "counts", "reload", "cross_arm", "order"],
)
def test_candidate_rejects_missing_or_conflicting_proof(tmp_path, change):
    a = candidate(tmp_path)
    if change == "missing":
        del a["candidate_sha256"]["flow/FNO.0.0.mdlus"]
    elif change == "wrong_hash":
        a["candidate_sha256"]["result.json"] = "0" * 64
    elif change == "cross_arm":
        a["history_k"] = 1
    else:
        path = tmp_path / "result.json"
        value = json.loads(path.read_text())
        if change == "fixture":
            value["status"] = "FC_P026_ENGINEERING_FIXTURE_NOT_CANDIDATE"
        if change == "counts":
            value["records"].pop()
        if change == "reload":
            value["official_fresh_reload_verified"] = False
        if change == "order":
            value["sampler_order_sha256"] = "b" * 64
        path.write_text(json.dumps(value))
        a["candidate_sha256"]["result.json"] = m.sha(path)
    with pytest.raises(ValueError):
        m.candidate_contract(tmp_path, a)


def test_actual_terminal_not_running_or_collected():
    unit = dict(
        InvocationID="a" * 32,
        ActiveState="active",
        SubState="exited",
        Result="success",
        ExecMainCode="1",
        ExecMainStatus="0",
        MainPID="0",
    )
    m.terminal(unit, "a" * 32)
    for field, value in (
        ("InvocationID", "b" * 32),
        ("SubState", "running"),
        ("MainPID", "12"),
        ("ExecMainStatus", "1"),
        ("ActiveState", "inactive"),
    ):
        changed = dict(unit, **{field: value})
        with pytest.raises(ValueError):
            m.terminal(changed, "a" * 32)


def test_original_numerical_cli_and_explicit_history():
    for k in (1, 4):
        plan = m.commands(k, "model", "manifest")
        assert len(plan) == 7
        for name, gpu, cmd in plan:
            assert not any("ppo" in word or "train.py" in word for word in cmd)
            if name in ("validation10", "dynamic6"):
                assert cmd[
                    cmd.index("--horizons") + 1 : cmd.index("--horizons") + 5
                ] == ["1", "10", "50", "100"]
                assert cmd[cmd.index("--segment-stride") + 1] == (
                    "25" if name == "validation10" else "1"
                )
                assert cmd[cmd.index("--evaluation-batch-size") + 1] == (
                    "4" if name == "validation10" else "8"
                )
                assert cmd[cmd.index("--action-mode") + 1] == "observed"
            if gpu:
                assert cmd[cmd.index("--fno-history-profile") + 1] == f"p026_k{k}"
                assert cmd[cmd.index("--allocator-fraction") + 1] == ".15"
            if name == "development_gate":
                assert cmd == [
                    "python",
                    "scripts/audit_dynamic_fno_development_gates.py",
                    "--force-window",
                    "/workspace/output/force_window/result.json",
                    "--checkpoint-sha256",
                    "model",
                    "--output",
                    "/workspace/output/development_gate.json",
                ]


def test_bound_rejects_path_escape(tmp_path):
    path = tmp_path / "proof"
    path.write_text("proof")
    assert m.bound(tmp_path, "proof", m.sha(path)) == path
    with pytest.raises(ValueError):
        m.bound(tmp_path / "nested", "../proof", m.sha(path))


@pytest.mark.parametrize("gpu", [False, True])
def test_resource_failure_removes_only_exact_created_id(tmp_path, monkeypatch, gpu):
    out = tmp_path / "out"
    (out / "evidence").mkdir(parents=True)
    args = SimpleNamespace(
        source=tmp_path, candidate=tmp_path / "candidate", repo=tmp_path, output=out
    )
    cid = "a" * 64
    calls = []

    def output(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "create":
            return cid + "\n"
        return json.dumps(
            [
                dict(
                    Image=m.IMAGE,
                    Mounts=[dict(Source=str(out), Destination="/workspace/output")],
                )
            ]
        )

    class Process:
        returncode = None

        def poll(self):
            return None

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(m.subprocess, "check_output", output)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(
        m, "memory", lambda: dict(MemFree=19 * 1024**3, MemAvailable=80 * 1024**3)
    )
    with pytest.raises(ValueError, match="20GiB"):
        m.run_container(args, {}, "test", gpu, ["python", "fixture.py"], float("inf"))
    assert calls[-1] == ["docker", "rm", "-f", cid]
    assert calls[0][:2] == ["docker", "create"]
    assert not any(word.startswith("HOME=") for word in calls[0])
    if not gpu:
        assert "--gpus" not in calls[0]
        assert calls[0][calls[0].index("--runtime") + 1] == "runc"
        assert "NVIDIA_VISIBLE_DEVICES=void" in calls[0]
        assert "CUDA_VISIBLE_DEVICES=" in calls[0]


def test_unchanged_audit_pins_are_three_original_numerical_sources():
    assert set(m.UNCHANGED_AUDITS) == {
        "scripts/audit_full40_validation_gate.py",
        "scripts/audit_dynamic_fno_development_gates.py",
        "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
    }
    assert all(len(value) == 64 for value in m.UNCHANGED_AUDITS.values())


@pytest.mark.parametrize("k", [1, 4])
def test_dev30_p026_profile_and_identity_only_extension(tmp_path, monkeypatch, k):
    from test_dev30_validation_diagnostic import report, expected
    from fluid_control import dual_fno

    audit_spec = importlib.util.spec_from_file_location(
        "p026_dev30_tested", SCRIPT_DIR / "audit_dev30_validation_diagnostic.py"
    )
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    checkpoint = tmp_path / "aerodynamic"
    role = SimpleNamespace(directory=checkpoint, model_sha256="m", state_sha256="s")
    identity = SimpleNamespace(
        payload={
            "kind": f"FC_P026_K{k}_HISTORY_FORCE_FNO",
            "history_input": {"profile": f"p026_k{k}"},
        },
        aerodynamic=role,
        flow=role,
        manifest_sha256="manifest",
    )
    monkeypatch.setattr(dual_fno, "validate_dual_fno_manifest", lambda p: identity)
    value = report()
    value.update(
        checkpoint_epoch=1,
        checkpoint_dir="/workspace/dual/aerodynamic",
        fno_history_profile=f"p026_k{k}",
        checkpoint_metadata=dict(
            dual_fno=True,
            manifest_sha256="manifest",
            flow_model_sha256="m",
            flow_state_sha256="s",
            aerodynamic_model_sha256="m",
            aerodynamic_state_sha256="s",
        ),
    )
    kwargs = dict(
        candidate_kind=f"FC_P026_K{k}_HISTORY_FORCE_FNO", checkpoint_dir=checkpoint
    )
    audit.validate_report_contract(value, expected(), **kwargs)
    assert audit.pooled_metrics(value, expected()) == audit.pooled_metrics(
        report(), expected()
    )
    for changed in (None, "legacy_k1", "p026_k4" if k == 1 else "p026_k1"):
        wrong = dict(value, fno_history_profile=changed)
        with pytest.raises(ValueError, match="history profile"):
            audit.validate_report_contract(wrong, expected(), **kwargs)


def chain_fixture():
    base = {"src/base.py": "a" * 64, "scripts/evaluate_tandem_fno.py": "b" * 64}
    overlays = {name: "c" * 64 for name in m.OVERLAYS}
    final = dict(base, **overlays)
    approval = dict(
        training_config_sha256="d" * 64,
        reviewed_overlay_sha256=overlays,
        source_sha256=final,
    )
    receipt = dict(
        schema_version=1,
        status="FC_P026_FORMAL_SOURCE_CHAIN_FROZEN",
        numerical_base_commit=m.BASE,
        base_archive_sha256="e" * 64,
        base_files_sha256=base,
        training_config_sha256="d" * 64,
        overlay_sha256=overlays,
        final_files_sha256=final,
    )
    return receipt, approval, base


def test_chain_matches_actual_base_and_explicit_overlay_map():
    receipt, approval, base = chain_fixture()
    m.verify_chain_maps(receipt, approval, base, "e" * 64)


@pytest.mark.parametrize(
    "fault", ["base", "archive", "nonoverlay", "extra", "missing", "overlay"]
)
def test_chain_rejects_arbitrary_approved_source_bytes(fault):
    receipt, approval, base = copy.deepcopy(chain_fixture())
    if fault == "base":
        receipt["base_files_sha256"]["src/base.py"] = "0" * 64
    elif fault == "archive":
        receipt["base_archive_sha256"] = "0" * 64
    elif fault == "nonoverlay":
        approval["source_sha256"]["src/base.py"] = "0" * 64
        receipt["final_files_sha256"]["src/base.py"] = "0" * 64
    elif fault == "extra":
        receipt["overlay_sha256"]["scripts/extra.py"] = "0" * 64
    elif fault == "missing":
        del receipt["overlay_sha256"]["scripts/p026_state_history.py"]
    else:
        receipt["overlay_sha256"] = dict(receipt["overlay_sha256"])
        receipt["overlay_sha256"]["scripts/p026_state_history.py"] = "0" * 64
    with pytest.raises(ValueError):
        m.verify_chain_maps(receipt, approval, base, "e" * 64)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "flow_architecture",
        "aerodynamic_architecture",
        "history_state_module_sha256",
        "history_inference_module_sha256",
    ],
)
def test_exact_frozen_loader_rejects_malformed_candidates_before_gpu(tmp_path, fault):
    # Isolated review may point at the not-yet-integrated production loader;
    # after integration the canonical src file is used. No torch/model import.
    loader_path = Path(
        os.environ.get(
            "P026_REVIEW_LOADER_SOURCE",
            str(SCRIPT_DIR.parent / "src/fluid_control/dual_fno.py"),
        )
    )
    source = tmp_path / "source"
    target = source / "src/fluid_control/dual_fno.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(loader_path, target)
    spec = importlib.util.spec_from_file_location("p026_cpu_schema_fixture", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        payload = dict(
            schema_version=1,
            status=module.P026_K4_MANIFEST_STATUS,
            kind=module.P026_K4_SYSTEM_KIND,
            config_sha256=module.CONFIG_SHA256,
            normalization_sha256=module.NORMALIZATION_SHA256,
            precision_protocol=module.PRECISION_PROTOCOL,
            architecture=module.ARCHITECTURE,
            flow_architecture=module.ARCHITECTURE,
            aerodynamic_architecture=dict(module.ARCHITECTURE, in_channels=18),
            flow_parent_model_sha256=module.FLOW_MODEL_SHA256,
            flow_parent_state_sha256=module.FLOW_STATE_SHA256,
            aerodynamic_initial_model_sha256=module.P026_AERO_INITIAL_MODEL_SHA256,
            aerodynamic_initial_state_sha256=module.P026_AERO_INITIAL_STATE_SHA256,
            history_input=module._p026_history_input(4),
            history_inventory=module._p026_inventory(),
            history_state_module_sha256=module.P026_HISTORY_STATE_SHA256,
            history_inference_module_sha256=module.P026_HISTORY_INFERENCE_SHA256,
            flow={},
            aerodynamic={},
            **module._experiment_contract(module.P026_K4_SYSTEM_KIND)["extra"],
        )
        if fault == "incomplete":
            payload.pop("precision_protocol")
        elif "architecture" in fault:
            payload[fault] = dict(payload[fault], in_channels=10)
        else:
            payload[fault] = "0" * 64
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        manifest = candidate / "dual_model_manifest.json"
        manifest.write_text(json.dumps(payload))
        approval = dict(
            history_k=4,
            source_sha256={"src/fluid_control/dual_fno.py": m.sha(target)},
            candidate_sha256={"dual_model_manifest.json": m.sha(manifest)},
        )
        expected = (
            "incomplete" if fault == "incomplete" else "history or role architecture"
        )
        with pytest.raises(ValueError, match=expected):
            m.validate_candidate_loader(source, candidate, approval)
    finally:
        sys.modules.pop(spec.name, None)
