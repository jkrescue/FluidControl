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
