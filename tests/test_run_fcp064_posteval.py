import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_fcp064_posteval.py"
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("p064_formal", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeBase:
    def require(self, condition, message):
        if not condition:
            raise ValueError(message)

    def fields(self, value, expected):
        self.require(all(value.get(k) == v for k, v in expected.items()), str(expected))

    def bound(self, root, name, digest):
        self.require((Path(root) / name).is_file() and len(digest) == 64, name)

    def sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def commands(self, history, aero_sha, manifest_sha):
        self.require(history == 1 and len(aero_sha) == len(manifest_sha) == 64, "identity")
        return [["validation_diagnostic", True,
                 ["python", "x", "--candidate-kind", "FC_P026_K1_HISTORY_FORCE_FNO"]],
                ["other", False, ["python", "y"]]]


class FormalIdentityTest(unittest.TestCase):
    def test_actual_candidate_metadata_contract(self):
        candidate = REPO / "artifacts/fcp064_controlled_aero_arm_b_20261006"
        files = {}
        for name in module.FILES:
            import hashlib
            files[name] = hashlib.sha256((candidate / name).read_bytes()).hexdigest()
        manifest = module.candidate_contract(FakeBase(), candidate, {"candidate_sha256": files})
        self.assertEqual(manifest["kind"], module.KIND)
        self.assertTrue(manifest["flow"]["frozen"])
        self.assertFalse(manifest["aerodynamic"]["frozen"])

    def test_only_numerical_command_change_is_candidate_kind(self):
        plan = module.commands(FakeBase(), "a" * 64, "b" * 64)
        self.assertEqual(plan[0][2][-1], module.KIND)
        self.assertEqual(plan[1], ["other", False, ["python", "y"]])

    def test_profile_labels_are_nonadmitting(self):
        profile = module.p028.repair_profile("FC-P064")
        self.assertEqual(profile.status("ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION"),
                         "FC_P064_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION")
        self.assertEqual(profile.kind, module.KIND)

    def test_preflight_binds_all_four_orchestration_sources(self):
        base = FakeBase()
        args = type("Args", (), {
            "numerical_runner": REPO / "scripts/run_fcp026_posteval.py",
            "entry_file": SCRIPT,
        })()
        paths = {
            "scripts/run_fcp026_posteval.py": args.numerical_runner,
            "scripts/run_fcp028_posteval.py": module.P028_PATH,
            "scripts/flow_repair_profiles.py": Path(module.p028.flow_repair_profiles.__file__),
            "scripts/run_fcp064_posteval.py": SCRIPT,
        }
        approval = {"orchestration_sha256": {name: base.sha(path) for name, path in paths.items()}}
        inherited = module._inherited_preflight
        try:
            module._inherited_preflight = lambda _base, _args: approval
            self.assertIs(module.preflight(base, args), approval)
            approval["orchestration_sha256"].pop("scripts/flow_repair_profiles.py")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                module.preflight(base, args)
        finally:
            module._inherited_preflight = inherited


if __name__ == "__main__":
    unittest.main()
