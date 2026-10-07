import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

P = Path(__file__).with_name("curate_p064_b04_long_excitation.py")
S = importlib.util.spec_from_file_location("b04_curator_under_test", P)
M = importlib.util.module_from_spec(S)
sys.modules[S.name] = M
S.loader.exec_module(M)


def test_numeric_times_and_view(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("constant", "system", "postProcessing"):
        (raw / name).mkdir()
    for value in 120 + .1 * np.arange(801):
        (raw / f"{value:g}").mkdir()
    view = tmp_path / "view"
    points = [[float(t), 0.0] for t in 120 + .1 * np.arange(801)]
    M.make_view(raw, view, "b04", points)
    assert len(M.numeric_times(raw)) == 801
    assert (view / "constant").is_symlink()
    config = json.loads((view / "case_config.json").read_text())
    assert config["start_time"] == 120 and config["end_time"] == 200
    assert config["action_points"] == points
    assert Path(config["source_restart_case"]).is_absolute()


def test_validate_hdf_preserves_normalization(tmp_path, monkeypatch):
    hdf = tmp_path / "x.h5"
    with h5py.File(hdf, "w") as f:
        f["state"] = np.zeros((801, 3, 128, 256), np.float32)
        f["mask"] = np.ones((801, 1, 128, 256), np.uint8)
        f["omega"] = np.zeros((801, 1), np.float32)
        f["force"] = np.zeros((801, 4), np.float32)
        f["time"] = (120 + .1 * np.arange(801, dtype=np.float64)).astype(np.float32)[:, None]
        f["x"] = np.linspace(8, 25, 256, dtype=np.float32)
        f["y"] = np.linspace(4, 11, 128, dtype=np.float32)
    norm = tmp_path / "normalization.source.json"
    norm.write_bytes(b'{"immutable":true}\n')
    output = tmp_path / "out"
    output.mkdir()
    view = tmp_path / "view"
    view.mkdir()
    points = [[float(t), 0.0] for t in 120 + .1 * np.arange(801)]
    (view / "case_config.json").write_text(json.dumps({"action_points": points}))
    (view / "vtk_result.json").write_text(json.dumps({"actual_vtk_times_float64": [x[0] for x in points]}))
    fake_verifier = type("Verifier", (), {"verify_b04_h100": staticmethod(lambda root, path: {"status": "fixture-pass"})})
    monkeypatch.setattr(M, "expected_force", lambda spec, raw, times: np.zeros((801, 4), np.float32))
    monkeypatch.setattr(M, "load_module", lambda name, path: fake_verifier)
    spec = {
        "view": str(view),
        "inputs": {
            "normalization": {"path": str(norm), "sha256": M.sha256(norm)},
            "h100_verifier": {"path": str(tmp_path / "verifier.py")},
        },
    }
    result = M.validate_hdf(spec, tmp_path / "raw", hdf, output)
    assert result["hdf_sha256"] == M.sha256(hdf)
    assert (output / "normalization.json").read_bytes() == norm.read_bytes()
    assert result["official_reader_h100_evidence"]["status"] == "fixture-pass"
