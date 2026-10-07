import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_p064_b04_long_excitation_cfd.py"
if not SOURCE.is_file():
    SOURCE = HERE.parent / "scripts/run_p064_b04_long_excitation_cfd.py"
spec = importlib.util.spec_from_file_location("driver", SOURCE)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def test_patch_roundtrip_801_points():
    points = [[120 + i / 10, 0.0 if i in (0, 800) else ((i % 3) - 1) / 10] for i in range(801)]
    text = "boundaryField\n{\n rearCylinder\n {\n type rotatingWallVelocity;\n omega constant 0;\n }\n frontBack\n {\n }\n}\n"
    out = driver.replace_rear_patch(text, points)
    path = HERE / ".tmp_U"
    try:
        path.write_text(out)
        assert driver.parse_table(path) == points
    finally:
        path.unlink(missing_ok=True)


def test_patch_rejects_missing_or_duplicate():
    with pytest.raises(RuntimeError):
        driver.replace_rear_patch("frontBack {}", [[120, 0]])
    duplicated = "rearCylinder\n{\n}\nfrontBack\n{}\nrearCylinder\n{\n}\nfrontBack\n{}"
    # Replacement is deliberately count=1 but malformed empty first patch cannot silently pass parse.
    with pytest.raises(RuntimeError):
        driver.replace_rear_patch(duplicated, [[120, 0]])


def test_control_replacement_is_fail_closed():
    assert driver.replace_once("startTime 0;", "startTime 0;", "startTime 120;", "x") == "startTime 120;"
    with pytest.raises(RuntimeError):
        driver.replace_once("", "startTime 0;", "startTime 120;", "x")
