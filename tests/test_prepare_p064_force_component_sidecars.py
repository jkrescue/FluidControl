import json
from pathlib import Path

import h5py
import numpy as np

import prepare_p064_force_component_sidecars as m


def property_text(total, pressure, viscous):
    def block(name, offset):
        return f"""{name}
{{
scalar
{{
Cd {total[offset]};
Cl {total[offset+1]};
CdPressure {pressure[offset]};
ClPressure {pressure[offset+1]};
CdViscous {viscous[offset]};
ClViscous {viscous[offset+1]};
}}
}}"""
    return block("forceFront", 0) + "\n" + block("forceRear", 2) + "\n"


def test_scan_write_and_official_reader(tmp_path):
    raw = tmp_path / "raw"
    hdf = tmp_path / "case.h5"
    times = np.array([[148.0], [148.1], [148.2]], dtype=np.float32)
    pressure = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]], dtype=np.float64)
    viscous = np.array([[.1,.2,.3,.4], [.2,.3,.4,.5], [.3,.4,.5,.6]], dtype=np.float64)
    total = pressure + viscous
    with h5py.File(hdf, "w") as handle:
        handle["time"] = times
        handle["force"] = total.astype(np.float32)
    for i, time in enumerate((148, 148.1, 148.2)):
        path = raw / f"{time:g}" / "uniform/functionObjects"
        path.mkdir(parents=True)
        (path / "functionObjectProperties").write_text(property_text(total[i], pressure[i], viscous[i]))
    entry = {"family":"synthetic","case":"case","hdf_path":str(hdf),"hdf_sha256":m.sha(hdf),"raw_root":str(raw)}
    summary, arrays = m.scan_entry(entry, include_rows=True)
    assert summary["frames"] == 3 and len(summary["property_inventory"]) == 3
    assert np.array_equal(arrays["pressure"], pressure)
    assert np.array_equal(arrays["viscous"], viscous)
    sidecar = tmp_path / "sidecar.h5"
    m.write_sidecar(sidecar, entry, arrays)
    m.verify_official(sidecar, arrays)


def test_component_mismatch_and_time_mismatch_fail(tmp_path):
    path = tmp_path / "148" / "uniform/functionObjects"
    path.mkdir(parents=True)
    bad = property_text([1,2,3,4], [.5,1,1.5,2], [.4,1,1.5,2])
    (path / "functionObjectProperties").write_text(bad)
    try:
        m.aligned_components(148.0, m.property_map(tmp_path))
    except ValueError as error:
        assert "component algebra" in str(error)
    else:
        raise AssertionError("component mismatch accepted")
    try:
        m.aligned_components(149.0, m.property_map(tmp_path))
    except ValueError as error:
        assert "time mismatch" in str(error)
    else:
        raise AssertionError("time mismatch accepted")


def test_real_family_boundary_frames(actual_spec):
    entries = m.source_entries(actual_spec)
    assert len(entries) == 45
    for family in m.FAMILIES:
        entry = next(x for x in entries if x["family"] == family)
        props = m.property_map(Path(entry["raw_root"]))
        with h5py.File(entry["hdf_path"], "r") as handle:
            for index in (0, len(handle["time"]) - 1):
                time = float(handle["time"][index, 0])
                hdf_total = np.asarray(handle["force"][index], dtype=np.float64)
                _, raw_total, pressure, viscous, delta = m.aligned_components(time, props)
                assert delta <= m.TIME_ATOL
                assert np.max(np.abs(pressure + viscous - raw_total)) <= m.SUM_ATOL
                assert np.max(np.abs(hdf_total - raw_total)) < 2e-5


def pytest_generate_tests(metafunc):
    if "actual_spec" in metafunc.fixturenames:
        path = Path("/tmp/p064-force-component-sidecar-review/PENDING.json")
        metafunc.parametrize("actual_spec", [json.loads(path.read_text())])
