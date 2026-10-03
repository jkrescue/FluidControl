from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/monitor_research_window.py"
SPEC = importlib.util.spec_from_file_location("research_watchdog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _count(rows: str) -> int:
    result = subprocess.run(
        ["sh", "-c", f"{MODULE.TRACKED_PROCESS_AWK}"],
        input=rows,
        text=True,
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())


def test_current_quickscreen_leaf_is_counted_once_not_wrappers() -> None:
    rows = """\
bash /usr/bin/bash scripts/run_full40_dev30_quickscreen_pipeline_spark.sh --execute qs1
docker docker run image python -u /workspace/scripts/spark_gpu_guard.py -- python -u /workspace/scripts/train_tandem_fno.py
python python -u /workspace/scripts/spark_gpu_guard.py -- python -u /workspace/scripts/train_tandem_fno.py
python python -u /workspace/scripts/train_tandem_fno.py --config-name tandem_fno_full40_quickscreen_onestep
"""
    assert _count(rows) == 1


def test_tracked_science_entrypoints_and_solver_are_counted() -> None:
    rows = """\
python3 python3 /repo/scripts/train_tandem_fno_rollout.py --config-name h20
python python /repo/scripts/evaluate_tandem_fno.py --checkpoint model.pt
python python /repo/scripts/train_tandem_hydrogym_ppo_pilot.py --execute
pimpleFoam pimpleFoam -case /tmp/real-case
python python /repo/scripts/spark_gpu_guard.py -- python /repo/scripts/evaluate_tandem_fno.py
grep grep train_tandem_fno.py
"""
    assert _count(rows) == 4


def test_current_h20_and_dynamic6_services_are_fail_stop_monitored() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'QS1_H20_SERVICE = "fluid-control-dev30-quickscreen-h20-qs1.service"' in source
    assert 'DYNAMIC6_SERVICE = "fluid-control-dynamic6-serial-r2-20261003.service"' in source
    assert 'alerts.append("qs1_h20_stopped_before_epoch_5")' in source
    assert 'alerts.append("dynamic6_stopped_before_six_cases")' in source
    assert '"current_services"' in source
