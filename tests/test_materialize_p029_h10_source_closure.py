"""Source-only closure tests; no model, HDF, container, or GPU access."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


STAGE = Path(__file__).resolve().parents[1]
REPO = Path("/workspace/fluid_control")
SCRIPT = STAGE / "scripts/materialize_p029_h10_source_closure.py"
if not SCRIPT.is_file():
    SCRIPT = STAGE / "scripts/materialize_p028_h10_source_closure.py"


def load_module():
    definition = importlib.util.spec_from_file_location("p029_h10_materializer_test", SCRIPT)
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


def actual_args(module, profile="p029", **changes):
    values = dict(
        repo=REPO,
        profile=profile,
        diagnostic=REPO / "scripts/diagnose_p028_short_horizon_comparison.py",
        launcher=REPO / "scripts/run_p028_h10_comparison.py",
        spec_generator=REPO / "scripts/prepare_p028_h10_comparison_spec.py",
        training_config=(REPO / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005"
                         / "resolved_config.yaml"),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_actual_p029_dry_inventory_is_exact13_and_pinned():
    module = load_module()
    _, inventory = module.source_inventory(actual_args(module))
    assert len(inventory) == 13
    for name, digest in module.PROFILES["p029"]["required_sha256"].items():
        assert inventory[name] == digest
    assert set(module.CANONICAL) <= set(inventory)


def test_p029_rejects_unreviewed_diagnostic_bytes(tmp_path):
    module = load_module()
    changed = tmp_path / "diagnose_p028_short_horizon_comparison.py"
    changed.write_text("# unreviewed\n")
    with pytest.raises(ValueError, match="reviewed P029 source differs"):
        module.source_inventory(actual_args(module, diagnostic=changed))


def test_linked_source_is_rejected(tmp_path):
    module = load_module()
    target = tmp_path / "target.py"
    target.write_text("pass\n")
    linked = tmp_path / "diagnose_p028_short_horizon_comparison.py"
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="linked source"):
        module.source_inventory(actual_args(module, diagnostic=linked))


def test_exclusive_output_rejects_broken_link_and_atomic_collision(tmp_path):
    module = load_module()
    linked = tmp_path / "linked-output"
    linked.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="already exists or is linked"):
        module.require_exclusive_output(linked)
    source = tmp_path / "temporary"
    target = tmp_path / "appeared"
    source.mkdir()
    target.mkdir()
    with pytest.raises(FileExistsError, match="appeared before publication"):
        module.publish_noreplace(source, target)
    assert source.is_dir() and target.is_dir()


def test_p029_main_without_execute_only_prints_inventory(monkeypatch, capsys, tmp_path):
    module = load_module()
    args = actual_args(module)
    monkeypatch.setattr(sys, "argv", [
        "materializer", "--profile", "p029", "--repo", str(args.repo),
        "--diagnostic", str(args.diagnostic), "--launcher", str(args.launcher),
        "--spec-generator", str(args.spec_generator),
        "--training-config", str(args.training_config),
        "--output", str(tmp_path / "must-not-exist"),
    ])
    module.main()
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "P029_H10_SOURCE_CLOSURE_PREPARATION_ONLY"
    assert value["profile"] == "p029" and len(value["files_sha256"]) == 13
    assert not (tmp_path / "must-not-exist").exists()


def test_p028_default_profile_semantics_remain_available():
    module = load_module()
    assert module.PROFILES["p028"] == {
        "preparation_status": "P028_H10_SOURCE_CLOSURE_PREPARATION_ONLY",
        "frozen_status": "P028_H10_SOURCE_CLOSURE_FROZEN",
    }
