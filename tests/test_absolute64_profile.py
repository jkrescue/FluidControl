import ast
import importlib.util
import json
from pathlib import Path

import pytest


STAGE = Path("/tmp/p064-absolute64-canonical-ppo-review")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_identical_to_reviewed_g_profile():
    old = ast.parse(Path(
        "/workspace/fluid_control/artifacts/"
        "p064_g_canonical_ppo_source_20261007_immutable/scripts/"
        "train_p064_g_symmetry_canonical_32768_ppo.py"
    ).read_text())
    new = ast.parse((STAGE / "full/scripts/train_p064_absolute64_symmetry_canonical_32768_ppo.py").read_text())
    def protocol(tree):
        node = next(x for x in tree.body if isinstance(x, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PROTOCOL" for t in x.targets))
        return ast.literal_eval(node.value)
    assert protocol(new) == protocol(old)


def test_supervisor_accepts_only_absolute64_identity():
    module = load(STAGE / "full/scripts/supervise_p064_absolute64_symmetry_canonical_ppo.py", "abs64_supervisor")
    good = {
        "candidate_arm": "ABSOLUTE64",
        "candidate_manifest_sha256": "a" * 64,
        "candidate_manifest_kind": "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO",
    }
    assert module.candidate_binding(good)[0] == "ABSOLUTE64"
    for key, value in (("candidate_arm", "G"), ("candidate_manifest_kind", "unknown")):
        wrong = dict(good, **{key: value})
        with pytest.raises(ValueError):
            module.candidate_binding(wrong)


def test_actual_terminal_lineage_validates_without_execution():
    module = load(STAGE / "full/scripts/train_p064_absolute64_symmetry_canonical_32768_ppo.py", "abs64_runner")
    pending = json.loads((STAGE / "PPO_PENDING.json").read_text())
    pending["status"] = module.STATUS
    pending["execution_authorized"] = True
    checked = module.validate_spec(pending)
    assert checked["protocol"]["seed"] == 20261007
    assert checked["protocol"]["timesteps"] == 32768
    for table in (checked["source_files"], checked["runtime_sources"]):
        for name, expected in table.items():
            assert module.sha(name) == expected


def test_runtime_allowlists_keep_old_profiles_and_add_absolute64():
    contract = (STAGE / "full/runtime_src/fluid_control/dual_control_contract.py").read_text()
    hydro = (STAGE / "full/runtime_src/fluid_control/tandem_hydrogym.py").read_text()
    kind = "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO"
    assert kind in contract and kind in hydro
    assert "FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO" in contract
    for old in ("FC_P064_ARM_A_CONTROLLED_AERO_FORCE_FNO",
                "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO",
                "FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO"):
        assert old in hydro
