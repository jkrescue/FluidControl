import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(start=106.0):
    rows = []
    previous = 0.0
    for index in range(800):
        applied = min(0.75, previous + 0.001)
        input_observation = np.zeros(69, dtype=float)
        output_observation = np.zeros(69, dtype=float)
        zero_observation = np.zeros(69, dtype=float)
        input_observation[64:68] = index + np.arange(4) / 10
        output_observation[64:68] = index + 1 + np.arange(4) / 10
        input_observation[68] = previous
        output_observation[68] = applied
        rows.append({
            "step": index + 1,
            "start_time": start + 0.1 * index,
            "end_time": start + 0.1 * (index + 1),
            "requested_omega": applied,
            "applied_omega": applied,
            "input_observation": input_observation.tolist(),
            "output_observation": output_observation.tolist(),
            "zero_observation": zero_observation.tolist(),
        })
        previous = applied
    return {"completed_cycles": 800, "rows": rows}


def test_b02_selection_has_exact_time_action_force_contract():
    core = load(ROOT / "scripts/diagnose_b02_controlled_train.py", "b02_core")
    driver = load(ROOT / "scripts/convert_b02_controlled_train.py", "b02_converter")
    selected, omega, force = driver.selection(progress(), core)
    frames = selected["records"][0]["frames"]
    assert len(frames) == 801
    assert frames[0]["time"] == 106.0
    assert frames[-1]["time"] == 186.0
    assert frames[0]["files"] == {"U": "case_mpc/106/U", "p": "case_mpc/106/p"}
    assert frames[-1]["files"] == {"U": "case_mpc/186/U", "p": "case_mpc/186/p"}
    assert omega.shape == (801, 1) and force.shape == (801, 4)
    np.testing.assert_array_equal(omega[0], np.array([0], dtype=np.float32))
    np.testing.assert_array_equal(force[0], np.array([0, .1, .2, .3], dtype=np.float32))
    np.testing.assert_array_equal(force[-1], np.array([800, 800.1, 800.2, 800.3], dtype=np.float32))


def test_old_b00_time_grid_is_rejected():
    core = load(ROOT / "scripts/diagnose_b02_controlled_train.py", "b02_core_old")
    with pytest.raises(ValueError, match="start grid"):
        core.validate_progress(progress(148.0))


def test_all_801_frames_are_new_and_batched_without_cache():
    driver = load(ROOT / "scripts/convert_b02_controlled_train.py", "b02_batches")
    frames = list(range(801))
    batches = driver.batches(frames)
    assert [len(batch) for batch in batches] == [48] * 16 + [33]
    assert [item for batch in batches for item in batch] == frames
    source = (ROOT / "scripts/convert_b02_controlled_train.py").read_text()
    assert "reused_frames=0" in source
    assert "newly_sampled_frames=801" in source
    assert "projected_policy_h1_h5_conversion_20261006_r2/packets" not in source


def test_only_terminal_reviewable_cfd_result_is_accepted():
    driver = load(ROOT / "scripts/convert_b02_controlled_train.py", "b02_result")
    terminal = {
        "status": "P064_B_SYMMETRY_CANONICAL_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION",
        "cycles": 800,
        "owned_containers_cleaned": True,
        "source_restart_unchanged": True,
    }
    driver.validate_source_result(terminal)
    for key, value in (("cycles", 799), ("owned_containers_cleaned", False),
                       ("source_restart_unchanged", False)):
        invalid = dict(terminal)
        invalid[key] = value
        with pytest.raises(ValueError, match="source result terminal"):
            driver.validate_source_result(invalid)


def test_source_approval_is_exact_b02_acquisition():
    driver = load(ROOT / "scripts/convert_b02_controlled_train.py", "b02_approval")
    approval = {
        "execution_authorized": True,
        "steps": 800,
        "start_time": 106,
        "end_time": 186,
        "output": "artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007",
    }
    driver.validate_source_approval(approval)
    invalid = dict(approval, start_time=148)
    with pytest.raises(ValueError, match="source approval time"):
        driver.validate_source_approval(invalid)
