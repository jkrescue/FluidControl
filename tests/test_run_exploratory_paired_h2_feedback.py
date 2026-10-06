import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_exploratory_paired_h2_feedback.py"
SPEC = importlib.util.spec_from_file_location("run_paired_h2", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def approval(tmp_path, **updates):
    payload = {
        "status": M.STATUS,
        "execution_authorized": True,
        "steps": 10,
        "start_time": 148.0,
        "inference_device": "cpu",
        "state_abs_limit": M.STATE_ABS_LIMIT,
        "deadline_seconds": 900,
        "original_long_ar_gate_passed": False,
        "scientific_admission": False,
        "source_files": {"x.py": "a" * 64},
        "inputs": {"k1_manifest": {"path": "candidate/dual_model_manifest.json",
                                    "sha256": "b" * 64}},
    }
    payload.update(updates)
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(payload))
    return payload, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_pending_approval_contract(tmp_path):
    payload, path, digest = approval(tmp_path)
    M.validate_spec(payload, path, digest)
    for update in ({"execution_authorized": False}, {"inference_device": "cuda:0"},
                   {"original_long_ar_gate_passed": True}, {"steps": 9}):
        value, candidate, candidate_digest = approval(tmp_path, **update)
        with pytest.raises(ValueError):
            M.validate_spec(value, candidate, candidate_digest)


def test_execute_loads_models_once_and_replans_inside_ten_cycle_callback():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute")
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    names = [getattr(node.func, "id", getattr(node.func, "attr", "")) for node in calls]
    assert names.count("load_bound_k1") == 1
    assert names.count("run_paired_ten_cycle") == 1
    assert any(isinstance(node, ast.FunctionDef) and node.name == "plan"
               for node in ast.walk(execute))


def test_actual_transport_uses_openfoam_run_and_current_time_export():
    source = SCRIPT.read_text()
    assert '"/openfoam/run"' in source
    assert '"foamToVTK"' in source
    assert 'f"{current_time:g}"' in source
    assert '"(U p)"' in source
    assert "total_drag_observation_at" in source
    assert "read_force_window" in source


def test_cpu_only_inference_and_measured_latency():
    source = SCRIPT.read_text()
    assert 'torch.device("cpu")' in source
    assert 'CUDA_VISIBLE_DEVICES") == ""' in source
    assert "cpu_inference_wall_seconds" in source
    assert "wall_latency_is_observational_not_realtime_claim" in source
    assert "cuda:0" not in source


def test_narrow_pair_mounts_and_owned_cleanup_are_present():
    source = SCRIPT.read_text()
    assert 'f"type=bind,src={case},dst=/case"' in source
    assert '["docker", "rm", "-f", cid]' in source
    assert '["docker", "ps", "-aq", "--no-trunc"]' in source
