import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_p030_train_horizon.py"
BASE = Path(os.environ.get(
    "P030_REVIEWED_LAUNCHER",
    Path(__file__).resolve().parents[1] / "scripts" / "run_p028_h10_comparison.py",
))
SPEC = importlib.util.spec_from_file_location("p030_launcher_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(tmp_path, name):
    path = tmp_path / name
    path.write_text("# source\n")
    return {"path": str(path), "sha256": digest(path)}


def approved(tmp_path):
    config = tmp_path / "config.yaml"; config.write_text("x")
    audit = tmp_path / "audit.json"; audit.write_text("{}")
    mapping = tmp_path / "mapping.json"; mapping.write_text("{}")
    pre = tmp_path / "pre.json"; pre.write_text("{}")
    reload_receipt = tmp_path / "reload.json"; reload_receipt.write_text("{}")
    terminal = tmp_path / "terminal.json"; terminal.write_text("{}")
    manifests = []
    for name in ("k1", "p029"):
        folder = tmp_path / name; folder.mkdir()
        manifest = folder / "dual_model_manifest.json"; manifest.write_text("{}")
        manifests.append((name, manifest))
    data = {}
    for family in ("base", "train8", "train16"):
        folder = tmp_path / family; (folder / "train").mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")
        (folder / "normalization.json").write_text("{}")
        data[family] = {"root": str(folder)}
    return {
        "status": MOD.APPROVED, "execution_authorized": True,
        "candidates": {name: {"manifest": str(path), "manifest_sha256": digest(path)}
                       for name, path in manifests},
        "data": data,
        "source_files": [source(tmp_path, "diagnose_p030_train_horizon.py")],
        "config": {"path": str(config), "sha256": digest(config)},
        "train_audit": {"path": str(audit)},
        "source_phase_mapping": {"path": str(mapping)},
        "train16_predeclaration": {"path": str(pre)},
        "p029_official_cpu_reload": {"path": str(reload_receipt),
                                      "sha256": digest(reload_receipt)},
        "p029_terminal_proof": {"path": str(terminal), "sha256": digest(terminal),
                                "reviewed_by_lead": True},
        "comparison_contract": {"split": "train", "starts": [0], "windows": 44,
                                "rollout_steps": 100, "report_leads": [1, 10, 25, 50, 100],
                                "selection_performed": False},
        "resource_contract": {"official_image": MOD.IMAGE, "gpu": 0,
                              "allocator_fraction": .06, "memory_gib": 12,
                              "deadline_seconds": 900, "startup_memfree_gib": 30,
                              "startup_memavailable_gib": 50,
                              "runtime_memory_floor_gib": 20,
                              "cuda_free_floor_gib": 20},
    }


def test_pending_status_rejected_and_approved_loaded(tmp_path):
    value = approved(tmp_path)
    path = tmp_path / "spec.json"; path.write_text(json.dumps(value))
    assert MOD.load_spec(path, digest(path)) == value
    value["status"] = "P030_TRAIN_HORIZON_DIAGNOSTIC_PREPARATION_ONLY_NOT_APPROVED"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="not approved"):
        MOD.load_spec(path, digest(path))


def test_mounts_are_narrow_and_exclude_k4_origin51_baselines(tmp_path, monkeypatch):
    value = approved(tmp_path)
    spec_path = tmp_path / "spec.json"; spec_path.write_text(json.dumps(value))
    monkeypatch.setattr(MOD, "GUARD_PATH", tmp_path / "guard.py")
    MOD.GUARD_PATH.write_text("# guard")
    mounts = MOD.readonly_mounts(value, spec_path)
    text = "\n".join(str(row[0]) for row in mounts)
    assert "k1" in text and "p029" in text
    assert "k4" not in text and "baseline" not in text
    assert all(row[2] for row in mounts)


def test_command_keeps_reviewed_resource_envelope(tmp_path, monkeypatch):
    value = approved(tmp_path)
    spec_path = tmp_path / "spec.json"; spec_path.write_text(json.dumps(value))
    output = tmp_path / "output"; (output / "evidence").mkdir(parents=True)
    monkeypatch.setattr(MOD, "GUARD_PATH", tmp_path / "guard.py")
    MOD.GUARD_PATH.write_text("# guard")
    command = MOD.create_command(value, spec_path, digest(spec_path), output)
    joined = " ".join(command)
    assert "--gpus device=0" in joined
    assert "--memory 12g --memory-swap 12g" in joined
    assert "--min-free-gib 20 --allocator-fraction .06 --margin-gib 4" in joined
    assert "timeout -k 20 900" in joined
    assert command[-1] == "--execute"


def test_result_requires_exact_selection_manifest_binding(tmp_path):
    value = approved(tmp_path)
    output = tmp_path / "output"; output.mkdir()
    selection = output / "selection.json"; selection.write_text("{}")
    raw = output / "raw_records.json"; raw.write_text("{}")
    result = {"status": MOD.COMPLETE, "source_spec": value,
              "scientific_admission": False, "optimizer_created": False,
              "model_saved": False, "validation_accessed": False,
              "frozen_test_accessed": False,
              "selection_manifest": {"path": str(selection.resolve()),
                                     "sha256": digest(selection)},
              "raw_records": {"path": str(raw.resolve()), "sha256": digest(raw)}}
    MOD.validate_result(result, value, output)
    result["selection_manifest"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="selection"):
        MOD.validate_result(result, value, output)


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("name", ["cleanup_owned", "running_gpu_containers", "recover_created_cid"])
def test_owned_cleanup_lifecycle_is_exact_reviewed_source(name):
    assert function_ast(SCRIPT, name) == function_ast(BASE, name)
