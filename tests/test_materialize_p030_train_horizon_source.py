import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_p030_train_horizon_source.py"
SPEC = importlib.util.spec_from_file_location("p030_materializer_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_broken_or_existing_output_is_rejected(tmp_path):
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="already exists"):
        MOD.require_exclusive_output(output)


def test_noreplace_collision_does_not_replace(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    target = tmp_path / "target"; target.mkdir()
    with pytest.raises(FileExistsError):
        MOD.publish_noreplace(source, target)
    assert source.is_dir() and target.is_dir()
