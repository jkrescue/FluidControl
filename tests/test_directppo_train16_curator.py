import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cfd/tandem_cylinders/curate_directppo_train16.py"


def load():
    spec = importlib.util.spec_from_file_location("directppo_train16_curator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorization_audits_all_historical_segments_and_forces():
    pytest.importorskip("h5py")
    module = load()
    payload = module.make_authorization(ROOT)
    assert payload["status"] == "DIRECTPPO_TRAIN16_CURATION_AUTHORIZED_TRAIN_ONLY"
    assert len(payload["cases"]) == 16
    assert payload["validation_or_frozen_accessed"] is False
    for row in payload["cases"].values():
        assert len(row["solver_log_sha256"]) == 128
        assert all(len(files) == 128 for files in row["force_file_sha256"].values())
        assert len(row["action_points"]) == 129


def test_execution_is_reviewed_but_explicitly_token_gated():
    module = load()
    assert module.EXECUTION_REVIEWED is True
    assert module.PREDECL_SHA == "7d9fc2a71ebe4bb0e817b1ce41f5e9a42310348bc5faf530bc1b3ad6a5229736"
    assert "observed quantization <=6.11e-6" in SCRIPT.read_text()
    source = (ROOT / "cfd/tandem_cylinders/run_directppo_train16_vtk_case.sh").read_text()
    assert "--cpus 1 --memory 8g" in source
    assert "refusing-existing-vtk" in source
