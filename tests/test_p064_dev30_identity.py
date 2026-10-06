import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_dev30_validation_diagnostic_p064.py"
spec = importlib.util.spec_from_file_location("p064_dev30", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Dev30IdentityTest(unittest.TestCase):
    def test_p064_uses_existing_k1_history_contract(self):
        root = Path("/tmp/p064-formal-test-candidate")
        checkpoint = root / "aerodynamic"
        identity = SimpleNamespace(
            payload={"kind": module.P064_KIND, "history_input": {"profile": "p026_k1"}},
            manifest_sha256="a" * 64,
            flow=SimpleNamespace(model_sha256="b" * 64, state_sha256="c" * 64),
            aerodynamic=SimpleNamespace(directory=checkpoint.resolve(), model_sha256="d" * 64,
                                        state_sha256="e" * 64),
        )
        package = types.ModuleType("fluid_control")
        dual = types.ModuleType("fluid_control.dual_fno")
        dual.validate_dual_fno_manifest = lambda path: identity
        old_package = sys.modules.get("fluid_control")
        old_dual = sys.modules.get("fluid_control.dual_fno")
        sys.modules["fluid_control"] = package
        sys.modules["fluid_control.dual_fno"] = dual
        try:
            names = [f"case{i}" for i in range(10)]
            report = {
                "split": "validation", "action_mode": "observed",
                "evaluation_data": "/workspace/devdata", "normalization_data": "/workspace/devdata",
                "checkpoint_dir": "/workspace/dual/aerodynamic", "force_channels": module.FORCE_CHANNELS,
                "action_scale": .75, "evaluation_action_limit": .75, "fno_history_profile": "p026_k1",
                "checkpoint_epoch": 1,
                "checkpoint_metadata": {"dual_fno": True, "manifest_sha256": "a" * 64,
                    "flow_model_sha256": "b" * 64, "flow_state_sha256": "c" * 64,
                    "aerodynamic_model_sha256": "d" * 64, "aerodynamic_state_sha256": "e" * 64},
                "cases": [{"case": name, "horizons": {str(h): {} for h in module.HORIZONS}}
                          for name in names],
            }
            module.validate_report_contract(report, {name: ("01", 0.) for name in names},
                                            candidate_kind=module.P064_KIND,
                                            checkpoint_dir=checkpoint)
        finally:
            if old_package is None: sys.modules.pop("fluid_control", None)
            else: sys.modules["fluid_control"] = old_package
            if old_dual is None: sys.modules.pop("fluid_control.dual_fno", None)
            else: sys.modules["fluid_control.dual_fno"] = old_dual


if __name__ == "__main__":
    unittest.main()
