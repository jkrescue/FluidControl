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


def test_vtk_float32_time_is_snapped_only_inside_fixed_tolerance():
    module = load()
    assert module.canonical_time(148.10000610351562, 148.1) == 148.1
    with pytest.raises(ValueError, match="cannot be matched"):
        module.canonical_time(148.10002, 148.1)


def test_endpoint_view_maps_only_first_and_last_reviewed_frames():
    module = load()

    class FakeSource:
        def __len__(self):
            return 129

        def __getitem__(self, index):
            return f"frame-{index}"

        def relative_path(self, index):
            return f"path-{index}"

    view = module.EndpointView(FakeSource())
    assert len(view) == 2
    assert view[0] == "frame-0"
    assert view[1] == "frame-128"
    assert view.relative_path(0) == "path-0"
    assert view.relative_path(1) == "path-128"


def test_endpoint_probe_is_separate_and_explicitly_token_gated():
    module = load()
    assert "endpoint-probe" in SCRIPT.read_text()
    assert module.ENDPOINT_PROBE != module.OUTPUT
    assert "DIRECTPPO_TRAIN16_ENDPOINT_PROBE_PASS" in SCRIPT.read_text()


def test_exact_force_series_requires_unique_real_endpoint_rows(tmp_path):
    import numpy as np

    module = load()
    name = "case"
    case = tmp_path / "cfd/tandem_cylinders/cases" / name
    (case / "postProcessing/forceFront/1").mkdir(parents=True)
    (case / "postProcessing/forceRear/1").mkdir(parents=True)
    (case / "postProcessing/forceFront/1/coefficient.dat").write_text("fixture")
    (case / "postProcessing/forceRear/1/coefficient.dat").write_text("fixture")

    class FakeBase:
        @staticmethod
        def load_merged_coefficients(paths):
            object_name = paths[0].parts[-3]
            offset = 0.0 if object_name == "forceFront" else 10.0
            return np.asarray(
                [[1.0, offset + 1.0, offset + 2.0], [1.1, offset + 3.0, offset + 4.0]]
            )

    actual = module.exact_force_series(
        FakeBase(), tmp_path, name, {}, np.asarray([1.0, 1.1])
    )
    assert actual.tolist() == [
        [1.0, 2.0, 11.0, 12.0],
        [3.0, 4.0, 13.0, 14.0],
    ]
    with pytest.raises(ValueError, match="no unique exact"):
        module.exact_force_series(FakeBase(), tmp_path, name, {}, [1.05])
