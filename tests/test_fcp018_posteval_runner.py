"""Software-only frozen import closure; no models, Docker or evaluation runs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = "7216214b545fbbd50b2fb5ed866f231039b06b18"
HELPERS = (
    "audit_fcp018_candidate.py", "audit_fcp015_candidate.py",
    "audit_fcp013_dual_candidate.py", "audit_fcp011_candidate.py",
    "evaluate_fcp013_fixed_train_windows.py", "train_fcp013_independent_force_fno.py",
    "diagnose_fcp014_train_objective.py", "verify_fcp015_dual_reload.py",
    "verify_fcp018_dual_reload.py", "finalize_fcp018_training.py",
    "validate_fcp008_posteval.py", "run_fcp008_posteval_spark.sh",
)
OVERLAYS = ("src/fluid_control/dual_fno.py", "src/fluid_control/calibrated_checkpoint.py",
            "scripts/evaluate_tandem_fno.py", "scripts/diagnose_fno_force_window.py",
            "scripts/audit_dev30_validation_diagnostic.py")


def test_runner_declares_full_p018_contract_and_original_commands():
    script = ROOT / "scripts/run_fcp008_posteval_spark.sh"
    text = script.read_text()
    for name in HELPERS:
        assert "scripts/" + name in text
    assert "docs/FC_P018_TRAINING_PROTOCOL_20261005.json" in text
    assert '--execution-observation-sha256 "$FCP018_EXECUTION_OBSERVATION_SHA256"' in text
    assert "from finalize_fcp018_training import read_unit,terminal_state" in text
    assert "assert terminal_state(read_unit())" in text
    assert '"$candidate/candidate:/workspace/dual:ro"' in text
    assert "actual_learning_rate=1.5625e-7" in text
    assert 'training_protocol_sha256=sha(candidate/"candidate/training_protocol.json")' in text
    assert 'dual_reload_receipt_sha256=sha(candidate/"dual_reload_receipt.json")' in text
    assert '--segment-stride 25 --evaluation-batch-size 4' in text
    assert '--segment-stride 1 --evaluation-batch-size 8' in text
    assert text.count('"${dual_args[@]}"') == 3
    assert f'numerical_commit="{NUMERICAL}"' in text
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.fixture
def staged(tmp_path):
    numerical = tmp_path / "numerical_source"
    numerical.mkdir()
    archive = subprocess.check_output(["git", "archive", NUMERICAL, "--", "src", "scripts", "conf", "cfd"], cwd=ROOT)
    subprocess.run(["tar", "-x", "-C", str(numerical)], input=archive, check=True)
    for relative in OVERLAYS:
        shutil.copyfile(ROOT / relative, numerical / relative)
    (tmp_path / "scripts").mkdir()
    for name in HELPERS:
        shutil.copyfile(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    return tmp_path


@pytest.mark.parametrize("missing", [None, "audit_fcp015_candidate.py", "audit_fcp013_dual_candidate.py",
                                      "diagnose_fcp014_train_objective.py"])
def test_isolated_actual_numerical_snapshot_import_closure(staged, missing):
    if missing:
        (staged / "scripts" / missing).unlink()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(
        str(staged / name) for name in ("scripts", "numerical_source/src", "numerical_source/scripts")))
    code = """
import pathlib
import audit_fcp018_candidate as audit
import verify_fcp018_dual_reload as reload
import finalize_fcp018_training as finalizer
import validate_fcp008_posteval as validator
root = pathlib.Path.cwd()
for module in (audit, reload, finalizer, validator):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
validator.configure_profile('p018')
assert validator.CHAIN_STATUS == 'FC_P018_IMMUTABLE_POSTEVAL_CHAIN_STAGED'
assert audit.EXPERIMENT['actual_learning_rate'] == 1.5625e-7
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=staged, env=env,
                            text=True, capture_output=True, timeout=30)
    if missing:
        assert result.returncode != 0
        assert "ModuleNotFoundError" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
