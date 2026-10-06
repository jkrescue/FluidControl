import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "formal_dashboard", Path(__file__).parents[1] / "scripts/serve_live_research_dashboard.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture
def formal(tmp_path):
    output = tmp_path / "output"
    evidence = output / "evidence"
    evidence.mkdir(parents=True)
    approval = dict(status="FC_P026_APPROVED_ORIGINAL_FORMAL_EVALUATION",
                    formal_evaluation_authorized=True, ppo_auto_launch=False,
                    history_k=1, output_relative_directory="output", official_image_id="pinned")
    raw = json.dumps(approval).encode()
    (tmp_path / "approval.json").write_bytes(raw)
    (evidence / "formal_approval.json").write_bytes(raw)
    log = output / "memory.jsonl"
    log.write_text(json.dumps(dict(time_unix=time.time(), step="validation10")) + "\n")
    reg = dict(invocation="actual", planned_updates={"K1": 7}, label="evaluation",
               description="not training", approval="approval.json",
               approval_sha256=hashlib.sha256(raw).hexdigest(), log="output/memory.jsonl")
    state = dict(InvocationID="actual", MainPID="42", ActiveState="activating",
                 SubState="start", Result="success", ExecMainCode="0", ExecMainStatus="0")
    def complete(name, changed_id=False):
        proof = dict(Id="created-" + name, Image="pinned",
                     Mounts=[dict(Source=str(output), Destination="/workspace/output", RW=True)])
        (evidence / (name + "_container.json")).write_text(json.dumps(proof))
        terminal = {**proof, "State": dict(Status="exited", Running=False, OOMKilled=False, ExitCode=0)}
        if changed_id:
            terminal["Id"] = "unrelated"
        (evidence / (name + "_container_terminal.json")).write_text(json.dumps(terminal))
    return tmp_path, reg, state, complete


def test_live_formal_stage_and_partial_final_line(formal):
    root, reg, state, complete = formal
    with (root / reg["log"]).open("a") as stream:
        stream.write('{"partial":')
    value = m._registered_formal_progress(root, reg, state, True)
    assert value["running"] and value["updates"] == {"K1": 0}
    assert "10个验证工况" in value["progress_detail"] and not value["admission"]


def test_counts_only_bound_contiguous_terminal_commands(formal):
    root, reg, state, complete = formal
    complete("validation10")
    assert m._registered_formal_progress(root, reg, state, True)["updates"] == {"K1": 1}
    complete("endpoint_gate")
    with pytest.raises(ValueError, match="contiguous"):
        m._registered_formal_progress(root, reg, state, True)


def test_unrelated_container_is_not_progress(formal):
    root, reg, state, complete = formal
    complete("validation10", changed_id=True)
    with pytest.raises(ValueError, match="identity"):
        m._registered_formal_progress(root, reg, state, True)


def test_terminal_completion_never_implies_accuracy_pass(formal):
    root, reg, state, complete = formal
    state.update(MainPID="0", ActiveState="active", SubState="exited", ExecMainCode="1")
    with pytest.raises(ValueError, match="all step evidence"):
        m._registered_formal_progress(root, reg, state, False)
    for name in m.FORMAL_STAGE_LABELS:
        complete(name)
    value = m._registered_formal_progress(root, reg, state, False)
    assert value["exited_success"] and value["updates"] == {"K1": 7}
    assert not value["running"] and not value["admission"]


def test_identity_approval_and_stale_samples(formal):
    root, reg, state, complete = formal
    assert not m._registered_formal_progress(root, reg, {**state, "InvocationID": "old"}, True)["verified"]
    assert not m._registered_formal_progress(root, reg, state, False)["verified"]
    (root / reg["log"]).write_text(json.dumps(dict(time_unix=0, step="validation10")))
    assert m._registered_formal_progress(root, reg, state, True)["progress_detail"] == "阶段采样待更新"
    (root / "output/evidence/formal_approval.json").write_text("changed")
    with pytest.raises(ValueError, match="approval copy"):
        m._registered_formal_progress(root, reg, state, True)


def p028_registration(root, reg, arm="P028", **overrides):
    approval = json.loads((root / "approval.json").read_text())
    approval.pop("history_k")
    approval.update(status=f"FC_{arm}_APPROVED_ORIGINAL_FORMAL_EVALUATION",
                    reviewed_by_lead=True,
                    independent_terminal_audit={"reviewed_by_lead": True},
                    official_dual_reload={"reviewed_by_lead": True})
    approval.update(overrides)
    raw = json.dumps(approval).encode()
    (root / "approval.json").write_bytes(raw)
    (root / "output/evidence/formal_approval.json").write_bytes(raw)
    return {**reg, "approval_sha256": hashlib.sha256(raw).hexdigest(),
            "planned_updates": {arm: 7}}


@pytest.mark.parametrize("arm", ["P028", "P029"])
def test_p028_uses_explicit_profile_without_false_history_arm(formal, arm):
    root, reg, state, complete = formal
    reg = p028_registration(root, reg, arm=arm)
    complete("validation10")
    value = m._registered_formal_progress(root, reg, state, True)
    assert value["running"] and value["updates"] == {arm: 1}
    assert value["admission"] is False
    with pytest.raises(ValueError, match="plan differs"):
        m._registered_formal_progress(root, {**reg, "planned_updates": {"K1": 7}}, state, True)


@pytest.mark.parametrize("override", [
    {"reviewed_by_lead": False},
    {"independent_terminal_audit": {}},
    {"official_dual_reload": {}},
    {"formal_evaluation_authorized": False},
    {"ppo_auto_launch": True},
    {"status": "UNKNOWN_FORMAL"},
])
@pytest.mark.parametrize("arm", ["P028", "P029"])
def test_p028_incomplete_approval_rejected(formal, override, arm):
    root, reg, state, complete = formal
    reg = p028_registration(root, reg, arm=arm, **override)
    with pytest.raises(ValueError, match="approved formal"):
        m._registered_formal_progress(root, reg, state, True)
