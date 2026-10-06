import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_binding_requires_success_and_explicit_conversion_disposition():
    bind = load(ROOT / "scripts/finalize_b02_conversion_spec.py", "b02_bind")
    result = {
        "status": bind.EXPECTED_STATUS, "cycles": 800,
        "owned_containers_cleaned": True, "source_restart_unchanged": True,
    }
    progress = {"completed_cycles": 800, "rows": [{}] * 800}
    bind.validate_terminal(result, progress)
    bind.validate_review("conversion READY abc invocation", "abc", "invocation")
    with pytest.raises(ValueError, match="terminal result cycles"):
        bind.validate_terminal(dict(result, cycles=799), progress)
    with pytest.raises(ValueError, match="conversion disposition"):
        bind.validate_review("abc invocation reviewed", "abc", "invocation")


def test_binding_rejects_symlink_and_pending_names_conversion_unit(tmp_path):
    bind = load(ROOT / "scripts/finalize_b02_conversion_spec.py", "b02_bind_path")
    regular = tmp_path / "regular"
    regular.write_text("x")
    symlink = tmp_path / "link"
    symlink.symlink_to(regular)
    assert bind.checked(regular) == regular.resolve()
    with pytest.raises(ValueError, match="regular file"):
        bind.checked(symlink)
    pending = json.loads((ROOT / "docs/P064_B02_CONTROLLED_TRAIN_CONVERSION_PENDING_20261007.json").read_text())
    assert pending["unit"] == "fluid-control-p064-b02-controlled-train-conversion-20261007.service"


def make_hdf(path: Path):
    with h5py.File(path, "w") as handle:
        state = np.arange(801 * 3 * 2 * 3, dtype=np.float32).reshape(801, 3, 2, 3) / 100
        mask = np.ones((801, 1, 2, 3), dtype=np.uint8)
        omega = np.linspace(-.75, .75, 801, dtype=np.float32)[:, None]
        force = np.arange(801 * 4, dtype=np.float32).reshape(801, 4) / 50
        handle["state"] = state
        handle["mask"] = mask
        handle["omega"] = omega
        handle["force"] = force
        handle["time"] = np.arange(801, dtype=np.float64)[:, None] / 10 + 106
        handle["x"] = np.arange(3, dtype=np.float32)
        handle["y"] = np.arange(2, dtype=np.float32)


def test_actual_official_datapipe_first_and_last_h100(tmp_path, monkeypatch):
    pytest.importorskip("physicsnemo")
    view = load(ROOT / "scripts/build_b02_train_view.py", "b02_view")
    project = Path("/workspace/fluid_control")
    monkeypatch.syspath_prepend(str(project / "src"))
    root = tmp_path / "view"
    train = root / "train"
    train.mkdir(parents=True)
    hdf = train / "b02_canonical_ppo_train.h5"
    make_hdf(hdf)
    norm = json.loads((project / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json").read_text())
    (root / "normalization.json").write_text(json.dumps(norm))
    (root / "manifest.json").write_text(json.dumps({"max_abs_omega": .75}))
    checks = view.verify_windows(root, hdf)
    assert [(item["start"], item["endpoint"]) for item in checks] == [(0, 100), (700, 800)]


def test_view_adapter_is_strictly_train_only_and_does_not_refit():
    source = (ROOT / "scripts/build_b02_train_view.py").read_text()
    assert '"validation", "test", "frozen_test"' in source
    assert "normalization_refit" in source
    assert "TandemRolloutDataset" in source
    assert "os.link" in source
