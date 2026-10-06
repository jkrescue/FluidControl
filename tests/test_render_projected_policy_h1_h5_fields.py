import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_projected_policy_h1_h5_fields.py"
SPEC = importlib.util.spec_from_file_location("render_replay_fields", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def packet(path: Path):
    y, x = np.mgrid[:128, :256]
    initial = np.stack((x / 255, y / 127, (x - 127.5) / 255)).astype(np.float32)
    truth = np.stack([initial + np.float32(.01 * step) for step in range(1, 6)])
    pred = truth + np.float32(.002)
    np.savez(path, initial_state=initial, truth_states=truth, predicted_states=pred,
             mask=np.ones((1, 128, 256), dtype=np.uint8),
             x=np.linspace(8, 25, 256, dtype=np.float32),
             y=np.linspace(4, 11, 128, dtype=np.float32),
             time=np.arange(6, dtype=np.float64).reshape(6, 1) / 10 + 148,
             omega=np.zeros((6, 1), dtype=np.float32),
             truth_forces=np.ones((6, 4), dtype=np.float32),
             predicted_forces=np.ones((5, 4), dtype=np.float32))


def test_render_saved_h1_h5_without_inference(tmp_path):
    source = tmp_path / "fields_mpc_0000.npz"
    destination = tmp_path / "preview.png"
    packet(source)
    item = MODULE.render_one(source, destination, "0000")
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert item["horizons"] == [1, 5]
    assert item["source_npz_sha256"] == MODULE.sha256(source)
    assert item["png_sha256"] == MODULE.sha256(destination)


def test_rejects_bad_shape_and_nonfinite(tmp_path):
    source = tmp_path / "fields_mpc_0000.npz"
    packet(source)
    with np.load(source) as archive:
        values = {name: archive[name] for name in archive.files}
    values["truth_states"] = values["truth_states"][:, :, :, :-1]
    np.savez(source, **values)
    with pytest.raises(ValueError, match="truth_states"):
        MODULE._checked_arrays(source)
