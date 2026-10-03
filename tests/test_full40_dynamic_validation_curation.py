import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "cfd/tandem_cylinders/curate_full40_dynamic_validation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dynamic6_curator", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_is_fail_closed():
    module = load_module()
    assert module.EXECUTION_REVIEWED is False
    assert module.EXPECTED == {
        f"full40_dynamic_validation_b{phase:02d}_{profile}"
        for phase in (1, 5)
        for profile in ("minus", "zero", "plus")
    }


def test_exclusive_writer_refuses_overwrite(tmp_path):
    module = load_module()
    output = tmp_path / "receipt.json"
    module.exclusive(output, {"status": "FIRST"})
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.exclusive(output, {"status": "SECOND"})
    assert json.loads(output.read_text()) == {"status": "FIRST"}


def test_authorization_rejects_predeclaration_sha(tmp_path, monkeypatch):
    module = load_module()
    path = tmp_path / module.PREDECL
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    monkeypatch.setattr(module, "sha", lambda candidate: "wrong")
    with pytest.raises(ValueError, match="predeclaration SHA differs"):
        module.make_authorization(tmp_path)


def test_checked_authorization_rejects_stale_receipt(tmp_path, monkeypatch):
    module = load_module()
    path = tmp_path / module.AUTH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "STALE"}))
    monkeypatch.setattr(
        module, "make_authorization", lambda repo: {"status": "CURRENT"}
    )
    with pytest.raises(ValueError, match="authorization is stale"):
        module.checked_auth(tmp_path)


def test_vtk_receipt_rejects_missing_frames(tmp_path, monkeypatch):
    module = load_module()
    name = "full40_dynamic_validation_b01_zero"
    monkeypatch.setattr(
        module,
        "checked_auth",
        lambda repo: {
            "cases": {name: {"run_window": [130.0, 150.0]}},
        },
    )
    (tmp_path / "cfd/tandem_cylinders/cases" / name / "VTK_curator").mkdir(parents=True)
    with pytest.raises(ValueError, match="expected exactly 201 VTK frames"):
        module.make_vtk_receipt(tmp_path, name)
