import ast
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train_p064_symmetry_canonical_b_32768_ppo.py"
SUPERVISOR = ROOT / "scripts/supervise_p064_symmetry_canonical_ppo.py"
PENDING = ROOT / "docs/P064_B_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_change_is_one_wrapper_inside_monitor_and_fixed_seed():
    source = TRAIN.read_text()
    tree = ast.parse(source)
    assert '"seed": 20261007' in source
    assert source.count("CanonicalSymmetryWrapper(wrapped)") == 1
    assert "Monitor(CanonicalSymmetryWrapper(wrapped))" in source
    assert ".5 *" not in source and "projected_request" not in source
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "symmetry_canonical_wrapper" for node in imports)


def test_pending_approved_clone_and_exact_source_bindings():
    runner = load("canonical_runner", TRAIN)
    spec = json.loads(PENDING.read_text())
    spec["status"] = runner.STATUS
    spec["execution_authorized"] = True
    spec["reviewed_by_lead"] = True
    runner.validate_spec(spec)
    assert spec["protocol"] == runner.PROTOCOL
    bound_runner = Path(spec["runner"])
    bound_supervisor = Path(spec["supervisor"])
    bound_adapter = Path(spec["import_bindings"]["symmetry_canonical_wrapper"])
    assert spec["source_files"][str(bound_runner)] == runner.sha(TRAIN)
    assert spec["source_files"][str(bound_supervisor)] == runner.sha(SUPERVISOR)
    assert spec["source_files"][str(bound_adapter)] == runner.sha(
        ROOT / "src/fluid_control/symmetry_canonical_wrapper.py"
    )


def test_actual_reward_and_filter_are_globally_reflection_invariant():
    joint = load("canonical_joint", ROOT / "src/fluid_control/canonical_joint_v1.py")
    baseline = {
        "source": "synthetic-contract-only",
        "total_drag": 2.0,
        "rear_cl_fluctuation_rms": 0.5,
    }
    times = np.arange(62, dtype=np.float64) * 0.1
    force = np.column_stack(
        (
            np.linspace(0.8, 1.0, 62),
            np.sin(times),
            np.linspace(0.9, 1.1, 62),
            np.cos(times),
        )
    )
    reflected = force.copy()
    reflected[:, 1] *= -1
    reflected[:, 3] *= -1
    for ledger_fn in (
        lambda value: joint.canonical_force_ledger(value, baseline, window_ready=True),
        lambda value: joint.causal_window_ledger(times, value, baseline),
    ):
        left, right = ledger_fn(force), ledger_fn(reflected)
        assert left["mean_rear_cl"] == -right["mean_rear_cl"]
        for key in (
            "total_drag_reduction",
            "rear_cl_fluctuation_ratio",
            "abs_mean_rear_cl_over_baseline_clprime_rms",
            "canonical_joint_gate_pass",
        ):
            assert np.isclose(left[key], right[key])
        for ready in (True, False):
            left_cost = joint.canonical_joint_cost_components(
                {**left, "window_ready": ready}, omega=0.31, delta_omega=-0.08
            )
            right_cost = joint.canonical_joint_cost_components(
                {**right, "window_ready": ready}, omega=-0.31, delta_omega=0.08
            )
            assert left_cost == right_cost
    left = joint.apply_action_rate_limit(0.6, 0.2)
    right = joint.apply_action_rate_limit(-0.6, -0.2)
    assert left["applied_omega"] == -right["applied_omega"]
    assert left["applied_delta_omega"] == -right["applied_delta_omega"]
    assert left["rate_limited"] == right["rate_limited"]
