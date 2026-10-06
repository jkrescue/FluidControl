from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/train_fcp064_controlled_aero_ab.py"
SPEC = importlib.util.spec_from_file_location("p064_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_view(tmp_path):
    root = tmp_path / "view"
    (root / "train").mkdir(parents=True)
    hdf = root / "train/b00.h5"
    hdf.write_bytes(b"synthetic-only")
    normalization = root / "normalization.json"
    normalization.write_bytes(b'{"fixed":"parent-bytes"}')
    receipt = tmp_path / "conversion_receipt.json"
    source_hdf = tmp_path / "source_b00.h5"
    source_hdf.write_bytes(hdf.read_bytes())
    receipt_value = {
        "status": "B00_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING",
        "frames": 801,
        "trajectories": 1,
        "split": "train",
        "hdf": {"path": str(source_hdf), "sha256": digest(hdf)},
        "normalization_sha256": digest(normalization),
        "normalization_refit": False,
        "official_reader_verified": True,
        "source_unchanged": True,
        "model_loaded": False,
        "cfd_executed": False,
        "optimizer_steps": 0,
        "scientific_admission": False,
        "owned_containers_cleaned": True,
    }
    receipt.write_text(json.dumps(receipt_value, sort_keys=True))
    manifest = {
        "max_abs_omega": 0.75,
        "conversion_receipt_sha256": digest(receipt),
        "normalization_sha256": digest(normalization),
        "hdf_sha256": {"b00.h5": digest(hdf)},
        "validation_accessed": False,
        "frozen_test_accessed": False,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    args = SimpleNamespace(
        b00_data_root=root,
        b00_manifest=manifest_path,
        b00_manifest_sha256=digest(manifest_path),
        b00_conversion_receipt=receipt,
        b00_conversion_receipt_sha256=digest(receipt),
        b00_source_hdf=source_hdf,
    )
    return args, hdf


def test_view_contract_binds_one_hdf_normalization_and_receipt(tmp_path):
    args, hdf = make_view(tmp_path)
    value = MODULE.validate_b00_view(args)
    assert value["hdf_sha256"] == {"b00.h5": digest(hdf)}
    hdf.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="HDF bytes"):
        MODULE.validate_b00_view(args)


def test_runner_has_fixed_budget_fresh_optimizer_and_exact_single_change():
    text = RUNNER.read_text()
    tree = ast.parse(text)
    assert "range(1, 33)" in text
    assert "lr=1.5625e-7" in text
    assert "shuffle=False" in text
    assert "load_dual_fno(" in text
    assert "FC_P064_ARM_{args.arm}_CONTROLLED_AERO_FORCE_FNO" in text
    assert "scheduled.schedule[len(observed)]" in text
    assert "optimizer = torch.optim.AdamW" in text
    assert "drop_caches" not in text
    assert not any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(item, ast.Constant) and item.value == "MemFree"
            for item in ast.walk(node)
        )
        for node in ast.walk(tree)
    )


def test_protocol_is_terminal_32_updates_for_both_arms():
    class Schedule:
        PARENT_ORDER_SHA256 = MODULE.ORDER_SHA

        @staticmethod
        def compile_schedule(order, arm):
            return [{"arm": arm, "order": order[:2]}]

        @staticmethod
        def schedule_sha256(rows):
            return MODULE.canonical_sha(rows)

    order = list(range(1368))
    for arm, b00 in (("A", 0), ("B", 64)):
        value = MODULE.protocol(arm, Schedule, order)
        assert value["training_windows"] == 256
        assert value["optimizer_steps"] == 32
        assert value["accumulation_windows"] == 8
        assert value["learning_rate"] == 1.5625e-7
        assert value["b00_windows"] == b00
        assert value["selection_performed"] is False
