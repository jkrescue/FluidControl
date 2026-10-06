import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


REPO = Path(os.environ.get("P064_REPO", Path(__file__).resolve().parents[1])).resolve()
OLD = REPO / "artifacts/exploratory_diverse_h5_32768_source_20261006_immutable/train_exploratory_diverse_h5_32768_ppo.py"
NEW = REPO / "scripts/train_p064_candidate_diverse_h5_32768_ppo.py"
OLD_APPROVAL = REPO / "docs/EXPLORATORY_DIVERSE_H5_32768_PPO_R2_APPROVAL_20261006.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("old_32768", OLD)
new = load("new_p064_32768", NEW)
TMP = tempfile.TemporaryDirectory()
TMPROOT = Path(TMP.name)


def fixture(arm="B"):
    value = json.loads(OLD_APPROVAL.read_text())
    value["status"] = new.STATUS
    value["execution_authorized"] = True
    value["candidate_arm"] = arm
    value["candidate_manifest_kind"] = new.P064_KINDS[arm]
    value["import_bindings"]["fluid_control.dual_control_contract"] = str(
        REPO / "src/fluid_control/dual_control_contract.py"
    )
    manifest_path = TMPROOT / f"{arm}_candidate" / "dual_model_manifest.json"
    result_path = manifest_path.parent / "result.json"
    review_path = TMPROOT / f"{arm}_review.json"
    manifest_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps({
        "status": f"FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION",
        "arm": arm, "optimizer_steps": 32, "training_windows": 256,
        "official_fresh_reload_verified": True, "scientific_admission": False,
        "aerodynamic_terminal_tensor_sha256": "1" * 64,
    }))
    manifest_path.write_text(json.dumps({
        "status": f"FC_P064_ARM_{arm}_DUAL_FNO_MANIFEST_VERIFIED",
        "kind": new.P064_KINDS[arm], "arm": arm,
        "training_windows": 256, "optimizer_steps": 32,
        "aerodynamic": {"model_sha256": "1" * 64},
    }))
    value["inputs"]["manifest"] = {"path": str(manifest_path), "sha256": new.sha(manifest_path)}
    value["candidate_manifest_sha256"] = value["inputs"]["manifest"]["sha256"]
    value["candidate_terminal_result"] = {"path": str(result_path), "sha256": new.sha(result_path)}
    review_path.write_text(json.dumps({
        "status": "P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION",
        "result_sha256": value["candidate_terminal_result"]["sha256"],
        "records": 32, "consumed": 256, "producer_official_reload": True,
        "unit": {"Result": "success", "ExecMainStatus": "0"},
    }))
    value["candidate_terminal_review"] = {"path": str(review_path), "sha256": new.sha(review_path)}
    return value


class CandidateTemplateTest(unittest.TestCase):
    def test_numerical_ppo_protocol_is_exactly_retained(self):
        self.assertEqual(new.PROTOCOL, old.PROTOCOL)
        self.assertEqual(new.fixed_panel(), old.fixed_panel())
        self.assertEqual(
            new.REQUIRED_IMPORTS,
            old.REQUIRED_IMPORTS | {"fluid_control.dual_control_contract"},
        )

    def test_explicit_p064_candidate_identity_is_accepted(self):
        for arm in ("A", "B"):
            self.assertIs(new.validate_spec(fixture(arm))["execution_authorized"], True)

    def test_old_or_cross_arm_identity_is_rejected(self):
        value = fixture("B")
        value["candidate_manifest_kind"] = new.P064_KINDS["A"]
        with self.assertRaisesRegex(ValueError, "kind differs"):
            new.validate_spec(value)
        value = fixture("B")
        value["status"] = old.STATUS
        with self.assertRaisesRegex(ValueError, "approval required"):
            new.validate_spec(value)

    def test_candidate_manifest_hash_is_bound_to_input(self):
        value = fixture("B")
        value["candidate_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest binding"):
            new.validate_spec(value)


if __name__ == "__main__":
    unittest.main()
