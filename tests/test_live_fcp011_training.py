import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dashboard_fcp011", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def sample(pid=123, step=32, stamp=990, loss=0.1):
    return json.dumps({"_PID": str(pid), "__REALTIME_TIMESTAMP": str(int(stamp * 1e6)),
                       "MESSAGE": json.dumps({"step": step, "total_loss": loss,
                                              "identity": {"split": "train", "rollout_steps": 100}})})


def live(*records, command="python scripts/train_fcp011_decoder_scope.py --scope head_only"):
    return "ActiveState=active\nMainPID=123\nExecStart=" + command + "\n" + "\n".join(records)


def test_live_training_uses_current_pid_and_finite_batch_values():
    status = dashboard._parse_fcp011_live(live(sample(pid=456, step=1368), sample()), 1000)
    assert status["running"] and status["step"] == 32
    assert status["progress_fresh"] and status["total_loss"] == 0.1
    assert status["admission"] is False


def test_stale_progress_does_not_invent_a_stopped_process_or_admission():
    status = dashboard._parse_fcp011_live(live(sample(stamp=100)), 1000)
    assert status["running"] and not status["progress_fresh"]
    assert status["step"] == 32 and not status["admission"]


def test_resource_probe_and_inactive_unit_are_not_training():
    status = dashboard._parse_fcp011_live(live(sample(), command="train_fcp011_decoder_scope.py --resource-probe"), 1000)
    assert not status["running"]
    status = dashboard._parse_fcp011_live("ActiveState=inactive\nMainPID=0\n" + sample(step=1368), 1000)
    assert not status["running"] and not status["admission"]


def test_missing_malformed_or_nonfinite_progress_is_unknown_not_zero():
    status = dashboard._parse_fcp011_live(live("not json", sample(loss=float("nan")), sample(step=1369)), 1000)
    assert status["running"] and status["step"] is None and status["total_loss"] is None
    assert not status["progress_fresh"]
    assert dashboard._parse_fcp011_live("", 1000)["service_state"] == "unknown"


def test_unbound_approval_does_not_show_paired_training(tmp_path):
    assert dashboard._fcp011_training(tmp_path) == {"ready": False}


def test_formal_process_requires_live_pid_and_exact_scope():
    value = "ActiveState=active\nMainPID=42\nExecStart=env FCP_POSTEVAL_PROFILE=p011_head_only /chain/scripts/run_fcp008_posteval_spark.sh --execute\n"
    assert dashboard._parse_fcp011_formal(value, "head-only")["running"]
    assert not dashboard._parse_fcp011_formal(value, "decoder-tail")["running"]
    assert not dashboard._parse_fcp011_formal(value.replace("MainPID=42", "MainPID=0"), "head-only")["running"]
    assert not dashboard._parse_fcp011_formal(value.replace("active", "inactive"), "head-only")["running"]
    assert not dashboard._parse_fcp011_formal(value, "head-only")["admission"]


def test_missing_formal_observation_is_unknown():
    value = dashboard._parse_fcp011_formal("", "head-only")
    assert value["service_state"] == "unknown" and not value["running"]


def test_terminal_must_match_reviewed_receipt_and_result(tmp_path):
    assert dashboard._fcp011_terminal(tmp_path, "head-only") == {"verified": False}
    base = tmp_path / "artifacts/fcp011_head_only_training_20261005"
    base.mkdir(parents=True)
    (base / "completion_receipt.json").write_text('{}')
    (base / "result.json").write_text('{"optimizer_steps":1368}')
    assert dashboard._fcp011_terminal(tmp_path, "head-only") == {"verified": False}


def test_formal_result_rejects_unbound_completion(tmp_path):
    assert dashboard._fcp011_formal_result(tmp_path, "head-only") == {"verified_fail": False}
    base = tmp_path / "posteval_fc_p011"
    base.mkdir()
    (base / "receipt.json").write_text('{"status":"FC_P011_HEAD_ONLY_POSTEVAL_COMPLETE"}')
    (base / "development_gate.json").write_text('{"status":"DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"}')
    assert dashboard._fcp011_formal_result(tmp_path, "head-only") == {"verified_fail": False}


def test_gradient_diagnostic_requires_exact_reviewed_result(tmp_path):
    assert dashboard._fcp012_diagnostic(tmp_path) == {"verified": False}
    base = tmp_path / "artifacts/fcp012_decoder_gradient_diagnostic_20261005/diagnostic"
    base.mkdir(parents=True)
    (base / "result.json").write_text('{"status":"COMPLETE","rows":[]}')
    assert dashboard._fcp012_diagnostic(tmp_path) == {"verified": False}
