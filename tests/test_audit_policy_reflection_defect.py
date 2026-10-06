import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_policy_reflection_defect.py"
SPEC = importlib.util.spec_from_file_location("reflection_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def observation(omega=0.0):
    value = np.arange(69, dtype=np.float32) / 10
    value[64:68] = [.3, -.4, .5, -.6]
    value[68] = omega
    return value


def row(step, previous, requested, applied):
    return {
        "step": step, "start_time": 148 + .1 * (step - 1), "end_time": 148 + .1 * step,
        "input_observation": observation(previous).tolist(),
        "requested_omega": requested, "applied_omega": applied,
        "applied_delta_omega": applied - previous,
    }


def test_reflection_exact_layout_and_involution():
    source = observation(.125)
    mirrored = MODULE.reflect_physical69(source)
    probes, reflected = source[:64].reshape(32, 2), mirrored[:64].reshape(32, 2)
    assert np.array_equal(reflected[:, 0], probes[::-1, 0])
    assert np.array_equal(reflected[:, 1], -probes[::-1, 1])
    assert np.array_equal(mirrored[64:69], np.array([.3, .4, .5, .6, -.125], np.float32))
    assert np.array_equal(MODULE.reflect_physical69(mirrored), source)


def test_snapshot_reads_source_once_and_binds_prefix(tmp_path, monkeypatch):
    source, output = tmp_path / "progress.json", tmp_path / "snapshot.json"
    payload = {"completed_cycles": 2, "rows": [row(1, 0., .04, .04), row(2, .04, .3, .14)]}
    source.write_text(json.dumps(payload))
    calls = 0
    original = Path.read_bytes

    def counted(path):
        nonlocal calls
        if path == source:
            calls += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    result = MODULE.snapshot_progress(source, output, cycles=1)
    assert calls == 1
    assert result["source_completed_cycles"] == 2 and result["prefix_cycles"] == 1
    assert result["last_end_time"] == pytest.approx(148.1)
    assert MODULE.sha(output) == MODULE.sha_bytes(output.read_bytes())


def test_snapshot_rejects_incomplete_or_inconsistent_rows(tmp_path):
    source = tmp_path / "progress.json"
    source.write_text(json.dumps({"completed_cycles": 2, "rows": [row(1, 0., .04, .04)]}))
    with pytest.raises(ValueError, match="completed prefix"):
        MODULE.snapshot_progress(source, tmp_path / "snapshot.json")
    source.write_text(json.dumps({"completed_cycles": 1, "rows": [row(1, .2, .04, .04)]}))
    with pytest.raises(ValueError, match="prior"):
        MODULE.snapshot_progress(source, tmp_path / "other.json")


def test_antisymmetric_projection_and_single_filter_reproduce_recorded():
    rows = [row(1, 0., .04, .04), row(2, .04, .3, .14)]

    def policy(obs):
        # This deliberately has an even defect .02 and odd dependence on omega.
        return float(.02 + .5 * obs[68])

    # Make the stored requests agree with the toy policy.
    rows[0]["requested_omega"] = .02
    rows[0]["applied_omega"] = .02
    rows[0]["applied_delta_omega"] = .02
    rows[1]["input_observation"][-1] = .02
    rows[1]["requested_omega"] = .03
    rows[1]["applied_omega"] = .03
    rows[1]["applied_delta_omega"] = .01
    result = MODULE.audit_policy(rows, policy, require_recorded_reproduction=True)
    records = result["records"]
    assert [entry["raw_symmetry_defect"] for entry in records] == pytest.approx([.04, .04])
    assert [entry["antisymmetrized_request"] for entry in records] == pytest.approx([0., .01])
    assert result["summary"]["raw_filter"] == {
        "action_clipped": 0, "rate_limited": 0, "saturated": 0}


def test_projection_uses_recorded_prior_for_filter_not_counterfactual_prior():
    rows = [row(1, 0., .5, .1), row(2, .1, .5, .2)]
    result = MODULE.audit_policy(rows, lambda obs: .5, require_recorded_reproduction=False)
    # A constant policy projects to zero. Step 2 starts from recorded +.1, so it
    # reaches zero; it must not carry the step-1 counterfactual projected state.
    assert result["records"][0]["antisymmetrized_filtered"]["applied_omega"] == 0.
    assert result["records"][1]["prior_recorded_omega"] == .1
    assert result["records"][1]["antisymmetrized_filtered"]["applied_omega"] == 0.


def test_audit_uses_injected_reviewed_action_filter():
    rows = [row(1, 0., .02, .02)]
    calls = []

    def reviewed(requested, previous):
        calls.append((requested, previous))
        return MODULE.constrain(requested, previous)

    MODULE.audit_policy(rows, lambda obs: .02, require_recorded_reproduction=True,
                        filter_action=reviewed)
    assert calls == [(pytest.approx(.02), 0.), (0., 0.)]


def test_wrong_recorded_policy_is_rejected():
    rows = [row(1, 0., .2, .1)]
    with pytest.raises(ValueError, match="recorded request"):
        MODULE.audit_policy(rows, lambda obs: .1, require_recorded_reproduction=True)


def test_filter_matches_symmetric_clip_and_rate_contract():
    positive = MODULE.constrain(2., .7)
    negative = MODULE.constrain(-2., -.7)
    assert positive["applied_omega"] == .75 and negative["applied_omega"] == -.75
    assert positive["action_clipped"] and negative["action_clipped"]
    assert positive["applied_delta_omega"] == pytest.approx(.05)
    assert negative["applied_delta_omega"] == pytest.approx(-.05)


def test_physical_memory_guards_use_distinct_startup_and_runtime_floors(monkeypatch):
    monkeypatch.setattr(MODULE, "mem_available_bytes", lambda: 50 * MODULE.GIB)
    assert MODULE.host_guard(startup=True) == 50 * MODULE.GIB
    monkeypatch.setattr(MODULE, "mem_available_bytes", lambda: 22 * MODULE.GIB)
    assert MODULE.host_guard(startup=False) == 22 * MODULE.GIB
    with pytest.raises(ValueError, match="MemAvailable"):
        MODULE.host_guard(startup=True)


def test_result_language_does_not_claim_no_optimizer_construction():
    source = SCRIPT.read_text()
    assert '"optimizer_created": False' not in source
    assert '"optimizer_steps": 0' in source
    assert '"policy_optimizer_reconstructed_by_load": True' in source
    assert "not a projected counterfactual rollout" in source
