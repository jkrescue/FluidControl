import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_p064_symmetry_canonical_b_ppo_long_cfd.py"
PENDING = ROOT / "docs/P064_B_SYMMETRY_CANONICAL_CFD_PENDING_20261007.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("canonical_cfd", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_policy_call_sign_restore_and_one_physical_filter():
    source = DRIVER.read_text()
    tree = ast.parse(source)
    assert "projected_request" not in source
    assert "raw_policy_request" not in source
    assert "reflected_policy_request" not in source
    assert source.count("canonical_action = predict(model, vec, canonical.value)") == 1
    assert source.count("symmetry.restore_physical_action(") == 1
    assert source.count("transport.apply_action_rate_limit(physical_request, previous)") == 1
    assert "canonical_orientation_applied" in source
    assert "next_canonical_orientation" in source
    assert "physical_requested_omega_before_filter" in source
    ast.fix_missing_locations(tree)


def test_pending_is_preparation_only_and_validates_when_future_hashes_are_bound():
    driver = load_driver()
    spec = json.loads(PENDING.read_text())
    assert spec["execution_authorized"] is False
    assert spec["inputs"]["training_result"]["sha256"] is None
    assert spec["inputs"]["policy"]["sha256"] is None
    assert spec["inputs"]["vecnormalize"]["sha256"] is None
    assert spec["inputs"]["training_review"]["sha256"] is None
    spec["status"] = driver.STATUS
    spec["execution_authorized"] = True
    for key in ("training_result", "policy", "vecnormalize", "training_review"):
        spec["inputs"][key]["sha256"] = "0" * 64
    assert driver.validate_spec(spec) is spec


def test_same_adapter_bytes_are_bound_as_source_and_input():
    spec = json.loads(PENDING.read_text())
    key = "artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/src/fluid_control/symmetry_canonical_wrapper.py"
    assert spec["source_files"][key] == spec["inputs"]["symmetry_adapter"]["sha256"]
    assert spec["inputs"]["symmetry_adapter"]["path"] == key
    assert spec["source_files"][key] == "a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae"


def test_training_approval_adapter_cross_binding_rejects_old_or_mismatched_identity():
    driver = load_driver()
    spec = json.loads(PENDING.read_text())
    adapter = Path("/repo/adapter.py")
    approval = {
        "status": "P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_EXECUTION_APPROVED",
        "execution_authorized": True,
        "import_bindings": {"symmetry_canonical_wrapper": str(adapter)},
        "source_files": {str(adapter): spec["inputs"]["symmetry_adapter"]["sha256"]},
        "protocol": {"symmetry_training_and_deployment_same_bytes": True},
    }
    inputs = {"symmetry_adapter": adapter}
    driver.validate_symmetry_binding(approval, spec, inputs)
    bad = json.loads(json.dumps(approval))
    bad["status"] = "P064_CANDIDATE_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED"
    try:
        driver.validate_symmetry_binding(bad, spec, inputs)
    except ValueError:
        pass
    else:
        raise AssertionError("old noncanonical training status accepted")
    bad = json.loads(json.dumps(approval))
    bad["source_files"][str(adapter)] = "0" * 64
    try:
        driver.validate_symmetry_binding(bad, spec, inputs)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched training adapter SHA accepted")
