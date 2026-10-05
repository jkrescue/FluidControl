import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dashboard_fcp013", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def row(step=8, pid=123, stamp=990, total=0.1):
    return json.dumps({"_PID": str(pid), "__REALTIME_TIMESTAMP": str(int(stamp * 1e6)),
                       "MESSAGE": json.dumps({"step": step, "total": total, "h1_balanced": 0.08,
                                              "ar_balanced": 0.12, "identity": {"split": "train", "rollout_steps": 100}})})


def live(*rows, command="/immutable/scripts/run_fcp013_training_spark.sh"):
    return "ActiveState=active\nMainPID=123\nExecStart=" + command + "\n" + "\n".join(rows)


def test_actual_training_pid_and_losses():
    value = dashboard._parse_fcp013_live(live(row(pid=777, step=1000), row()), 1000)
    assert value["running"] and value["progress_fresh"] and value["step"] == 8
    assert value["h1_balanced"] == 0.08 and value["ar_balanced"] == 0.12
    assert not value["admission"]


def test_resource_probe_and_other_command_are_not_training():
    for command in ("train_fcp013_independent_force_fno.py --resource-probe", "unrelated.py"):
        assert not dashboard._parse_fcp013_live(live(row(), command=command), 1000)["running"]


def test_stale_progress_is_not_fresh_or_completion():
    value = dashboard._parse_fcp013_live(live(row(stamp=100)), 1000)
    assert value["running"] and not value["progress_fresh"] and not value["admission"]
    value = dashboard._parse_fcp013_live("ActiveState=inactive\nMainPID=0\n" + row(step=1368), 1000)
    assert not value["running"] and not value["progress_fresh"] and not value["admission"]


def test_invalid_rows_are_unknown_not_zero():
    value = dashboard._parse_fcp013_live(live(row(total=float("nan")), row(step=1369), row(stamp=1006)), 1000)
    assert value["step"] is None and value["total"] is None
    assert dashboard._parse_fcp013_live("", 1000)["service_state"] == "unknown"


def test_training_needs_bound_approval(tmp_path):
    assert dashboard._fcp013_training(tmp_path) == {"ready": False}
