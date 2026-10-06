import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("/workspace/fluid_control")
DRIVER = ROOT / "scripts/evaluate_p064_c_development_h1_h5.py"
PENDING = ROOT / "docs/P064_CONTROLLED_DATA_DOSE_C_DEVELOPMENT_PENDING_20261007.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("c_development_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c_is_the_only_new_candidate_identity():
    old = (ROOT / "scripts/evaluate_p064_candidate_development_h1_h5.py").read_text()
    new = DRIVER.read_text()
    expected = old.replace(
        '"""Fixed retrospective 16 x H5 inference; explicit approval, no optimization."""',
        '"""Fixed retrospective 16 x H5 inference with the reviewed P064 C identity."""',
    ).replace("('K1','A','B')", "('K1','A','B','C')")
    assert new == expected


def test_c_pending_reaches_bound_metadata_validation(monkeypatch):
    module = load_driver()
    payload = json.loads(PENDING.read_text())
    payload["status"] = module.STATUS
    payload["execution_authorized"] = True

    def fake_bound(item):
        if item is payload["driver"]:
            return DRIVER
        return ROOT / "README.md"

    monkeypatch.setattr(module, "bound", fake_bound)
    inputs, output = module.validate_spec(payload)
    assert payload["candidate_label"] == "C"
    assert set(inputs) == set(payload["inputs"])
    assert output == ROOT / payload["output"]


def test_actual_project_imports_resolve_to_bound_stage_sources():
    payload = json.loads(PENDING.read_text())
    code = """
import hashlib, importlib, json
from pathlib import Path
s=json.loads(Path(%r).read_text())
for key,name in s['import_bindings'].items():
    m=importlib.import_module(name)
    actual=Path(m.__file__).resolve()
    expected=Path(s['sources'][key]['path']).resolve()
    assert actual == expected, (key,actual,expected)
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == s['sources'][key]['sha256']
print('C_DEVELOPMENT_IMPORT_ORIGINS_PASS')
""" % str(PENDING)
    completed = subprocess.run(
        [str(ROOT / ".venv-curator-py312/bin/python"), "-c", code],
        env={
            "PYTHONPATH": ":".join(payload["pythonpath"]),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "C_DEVELOPMENT_IMPORT_ORIGINS_PASS" in completed.stdout
