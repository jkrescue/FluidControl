from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import types

import pytest

try:
    from physicsnemo.datapipes import DatasetBase  # noqa: F401
except ModuleNotFoundError:
    class DatasetBase:
        def __init__(self, num_workers=0):
            self.num_workers = num_workers
        def close(self):
            pass
    physicsnemo = types.ModuleType("physicsnemo")
    datapipes = types.ModuleType("physicsnemo.datapipes")
    datapipes.DatasetBase = DatasetBase
    physicsnemo.datapipes = datapipes
    sys.modules["physicsnemo"] = physicsnemo
    sys.modules["physicsnemo.datapipes"] = datapipes

from fluid_control import p064_controlled_aero_ab as old


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/workspace/fluid_control")
SOURCE_ROOT = REPO / "artifacts/p064_controlled_data_dose_c50_source_20261007_immutable"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new = load_module(ROOT / "src/fluid_control/p064_controlled_aero_abc.py", "new_schedule")


def parent_order() -> list[int]:
    return json.loads((REPO / "tests/p064_parent_order.json").read_text())


def test_old_a_b_schedules_are_exactly_unchanged():
    order = parent_order()
    for arm in ("A", "B"):
        expected = old.compile_schedule(order, arm)
        actual = new.compile_schedule(order, arm)
        assert [asdict(row) for row in actual] == [asdict(row) for row in expected]
        assert new.schedule_sha256(actual) == old.schedule_sha256(expected)


def test_c_is_exactly_half_controlled_with_fixed_positions_and_full_span():
    rows = new.compile_schedule(parent_order(), "C")
    assert len(rows) == new.TRAINING_WINDOWS == 256
    controlled = [row for row in rows if row.source == "controlled_b00"]
    assert len(controlled) == new.C00_WINDOWS == 128
    assert [row.b00_start for row in controlled] == list(
        new.evenly_spaced_indices(new.B00_VALID_STARTS, new.C00_WINDOWS)
    )
    assert controlled[0].b00_start == 0 and controlled[-1].b00_start == 700
    for update in range(new.OPTIMIZER_STEPS):
        group = rows[update * 8 : (update + 1) * 8]
        assert [row.within_update for row in group if row.b00_start is not None] == [
            0,
            2,
            4,
            6,
        ]


def test_runner_protocol_is_unchanged_for_a_b_and_fixed_for_c():
    old_runner = load_module(REPO / "scripts/train_fcp064_controlled_aero_ab.py", "old_runner")
    new_runner = load_module(ROOT / "scripts/train_fcp064_controlled_aero_abc.py", "new_runner")
    order = parent_order()
    for arm in ("A", "B"):
        assert new_runner.protocol(arm, new, order) == old_runner.protocol(arm, old, order)
    c = new_runner.protocol("C", new, order)
    assert c["training_windows"] == 256
    assert c["optimizer_steps"] == 32
    assert c["accumulation_windows"] == 8
    assert c["b00_windows"] == 128
    assert c["b00_weight"] == 0.5
    assert c["replacement_within_each_update"] == [0, 2, 4, 6]
    assert c["validation_accessed"] is False
    assert c["frozen_test_accessed"] is False
    assert c["selection_performed"] is False


def test_c_requires_controlled_dataset_and_a_rejects_it():
    order = parent_order()
    class Original:
        _datasets = [object(), object(), object()]
        def __len__(self):
            return 1368
    original = Original()
    with pytest.raises(ValueError, match="require b00_dataset"):
        new.ScheduledABCDataset(original, object(), order, "C")
    with pytest.raises(ValueError, match="must not expose"):
        new.ScheduledABCDataset(original, object(), order, "A", b00_dataset=object())


def test_invalid_arm_fails_closed():
    with pytest.raises(ValueError, match="arm must be exactly"):
        new.compile_schedule(parent_order(), "D")


def test_source_closure_and_fixed_development_handoff_are_bound():
    pending = json.loads((ROOT / "docs/P064_CONTROLLED_DATA_DOSE_C_TRAINING_R2_APPROVAL_20261007.json").read_text())
    manifest_path = SOURCE_ROOT / "source_manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == pending["source_manifest_sha256"]
    source_manifest = json.loads(manifest_path.read_text())
    assert len(source_manifest) == pending["source_file_count"] == 439
    for relative, expected in source_manifest.items():
        assert hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest() == expected

    candidate = json.loads((ROOT / "docs/P064_CONTROLLED_DATA_DOSE_C_DEVELOPMENT_PENDING_20261007.json").read_text())
    baseline = json.loads((REPO / "docs/P064_ARM_B_DEVELOPMENT_PENDING_20261006.json").read_text())
    assert candidate["execution_authorized"] is False
    assert candidate["future_bindings_complete"] is False
    assert candidate["candidate_label"] == "C"
    assert candidate["evaluation_protocol"] == baseline["evaluation_protocol"]
    assert candidate["resources"] == baseline["resources"]
    assert candidate["runtime_versions"] == baseline["runtime_versions"]
    assert candidate["runtime_sources"] == baseline["runtime_sources"]
    assert candidate["driver"] == {
        "path": str(SOURCE_ROOT / "scripts/evaluate_p064_c_development_h1_h5.py"),
        "sha256": source_manifest["scripts/evaluate_p064_c_development_h1_h5.py"],
    }
    assert candidate["base"] == baseline["base"]
    for key in ("config", "conversion_result", "normalization", "selection"):
        assert candidate["inputs"][key] == baseline["inputs"][key]
    for key in ("manifest", "training_approval", "training_result"):
        assert candidate["inputs"][key]["sha256"] is None
    assert candidate["training_terminal_observation"] is None
