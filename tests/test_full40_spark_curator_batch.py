from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load():
    path = REPO / "scripts/run_full40_curator_batch_spark.py"
    spec = importlib.util.spec_from_file_location("spark_full40_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduler_selects_only_eleven_planned_train_cases() -> None:
    module = load()
    curator = module.load(
        "scripts/curate_matched_start_full40_remainder.py", "test_batch_curator"
    )
    cases = module.selected_train_cases(curator.load_matrix(module.REPO))
    assert len(cases) == 11
    assert module.PILOT in cases
    assert all("_train_" in name for name in cases)


def test_scheduler_has_reviewed_resource_and_concurrency_guards() -> None:
    module = load()
    assert module.EXECUTION_REVIEWED is True
    assert module.MAX_PARALLEL == 4
    assert module.MIN_MEMORY_GIB == 40
    assert module.MIN_DISK_GIB == 250


def test_case_command_is_split_safe_and_atomic(tmp_path: Path) -> None:
    module = load()
    module.REPO = tmp_path
    name = "matched_start_acquisition_train_b06_m0375"
    command = " ".join(module.case_command(name))
    assert name in command
    assert "prepare_matched_start_full40_vtk.py" in command
    assert "curate_matched_start_full40_remainder.py" in command
    assert "FINALIZE" not in command

    marker = (
        tmp_path
        / "artifacts/matched_start_full40_extension/vtk_ready"
        / f"{name}.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    resumed_command = " ".join(module.case_command(name))
    assert "prepare_matched_start_full40_vtk.py" not in resumed_command
    assert "curate_matched_start_full40_remainder.py" in resumed_command
