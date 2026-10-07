import importlib.util
import json
import sys
import pytest
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/workspace/fluid_control")
HERE = Path(__file__).resolve().parent
NEW_PATH = (
    HERE / "p064_controlled_aero_abi.py"
    if (HERE / "p064_controlled_aero_abi.py").is_file()
    else HERE.parent / "src/fluid_control/p064_controlled_aero_abi.py"
)
RUNNER_PATH = (
    HERE / "train_fcp064_controlled_aero_abi.py"
    if (HERE / "train_fcp064_controlled_aero_abi.py").is_file()
    else HERE.parent / "scripts/train_fcp064_controlled_aero_abi.py"
)
OLD_PATH = ROOT / "artifacts/p064_b00_b02_coverage_d_source_20261007_immutable/src/fluid_control/p064_controlled_aero_abd.py"
ORDER = ROOT / "artifacts/p064_b00_b02_coverage_d_source_20261007_immutable/parent_order.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_old_abc_schedules_are_unchanged_and_i_is_fixed_32_32():
    new = load("p064_abi_new", NEW_PATH)
    old = load("p064_abd_old", OLD_PATH)
    order = json.loads(ORDER.read_text())
    for arm in ("A", "B", "C"):
        assert [asdict(x) for x in new.compile_schedule(order, arm)] == [asdict(x) for x in old.compile_schedule(order, arm)]
    rows = new.compile_schedule(order, "I")
    assert len(rows) == 256
    assert sum(x.source == "original44" for x in rows) == 192
    assert sum(x.source == "controlled_b00" for x in rows) == 32
    assert sum(x.source == "controlled_b04" for x in rows) == 32
    assert [x.within_update for x in rows if x.source == "controlled_b00"] == [0] * 32
    assert [x.within_update for x in rows if x.source == "controlled_b04"] == [4] * 32
    starts00 = [x.b00_start for x in rows if x.source == "controlled_b00"]
    starts04 = [x.b00_start for x in rows if x.source == "controlled_b04"]
    assert starts00 == starts04 == list(new.evenly_spaced_indices(701, 32))


def test_protocol_names_b04_as_fixed_excitation_not_policy():
    runner = load("p064_abi_runner", RUNNER_PATH)
    schedule = load("p064_abi_schedule", NEW_PATH)
    order = json.loads(ORDER.read_text())
    protocol = runner.protocol("I", schedule, order)
    assert protocol["b00_windows"] == 32 and protocol["b04_windows"] == 32
    assert protocol["training_windows"] == 256 and protocol["optimizer_steps"] == 32
    assert "fixed_b04_prbs_long_excitation" in protocol["controlled_b04_action_semantics"]
    assert "policy" not in protocol["controlled_b04_action_semantics"]


def test_real_curator_result_schema_is_accepted(tmp_path):
    runner = load("p064_abi_runner_schema", RUNNER_PATH)
    root = tmp_path / "view"
    (root / "train").mkdir(parents=True)
    hdf = root / "train" / "p064_b04_long_excitation_120_200.h5"
    hdf.write_bytes(b"fixture-hdf")
    norm = root / "normalization.json"
    norm.write_bytes(b"fixture-norm")
    manifest = {
        "profile": "p064_b04_long_excitation_train_only",
        "trajectory_counts": {"train": 1, "validation": 0, "test": 0},
        "frames_per_trajectory": 801,
        "pairs_per_trajectory": 800,
        "max_abs_omega": .75,
        "normalization_status": "byte-exact reuse; no refit",
        "normalization_sha256": runner.sha(norm),
        "validation_or_frozen_accessed": False,
        "hdf_sha256": runner.sha(hdf),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    receipt = {
        "status": "P064_B04_LONG_EXCITATION_CURATED_TRAIN_ONLY",
        "frames": 801,
        "hdf_sha256": runner.sha(hdf),
        "normalization_reused_byte_exact": True,
        "validation_or_frozen_accessed": False,
        "training_executed": False,
        "official_reader_h100_evidence": {
            "status": "B04_H100_READER_ENDPOINTS_VERIFIED_NOT_TRAINING",
            "windows": [{"start": 0}, {"start": 700}],
            "decoded_frames": 202,
            "model_forward": 0,
            "optimizer_steps": 0,
        },
    }
    receipt_path = root / "result.json"
    receipt_path.write_text(json.dumps(receipt))
    args = SimpleNamespace(
        b04_data_root=root,
        b04_manifest=manifest_path,
        b04_manifest_sha256=runner.sha(manifest_path),
        b04_conversion_receipt=receipt_path,
        b04_conversion_receipt_sha256=runner.sha(receipt_path),
        b04_source_hdf=hdf,
        base_normalization_sha256=runner.sha(norm),
    )
    assert runner.validate_b04_view(args) == manifest

    args.base_normalization_sha256 = "0" * 64
    with pytest.raises(ValueError, match="b04 normalization bytes differ"):
        runner.validate_b04_view(args)
