import importlib.util
import json
from pathlib import Path


STAGE = Path("/tmp/p064-symmetry-seed20261006-replication-20261007")
RUNNER = STAGE / "scripts/train_p064_symmetry_canonical_b_seed20261006_32768_ppo.py"
PENDING = STAGE / "docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_PENDING_20261007.json"
BASE = Path("/workspace/fluid_control/scripts/train_p064_symmetry_canonical_b_32768_ppo.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("seed_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_diff_is_exactly_one_seed_line():
    candidate = RUNNER.read_text()
    base = BASE.read_text()
    assert candidate.count('"seed": 20261006,') == 1
    assert base.count('"seed": 20261007,') == 1
    assert candidate.replace('"seed": 20261006,', '"seed": 20261007,') == base


def test_pending_approved_clone_validates_with_same_profile_and_resources():
    runner = load_runner()
    pending = json.loads(PENDING.read_text())
    assert pending["execution_authorized"] is False
    assert pending["protocol"]["seed"] == 20261006
    approved = json.loads(json.dumps(pending))
    approved["status"] = runner.STATUS
    approved["execution_authorized"] = True
    approved["reviewed_by_lead"] = True
    runner.validate_spec(approved)
    assert approved["protocol"] == runner.PROTOCOL
    assert approved["protocol"]["symmetry_adapter"] == "physical69_max_abs_first_tie_fixed_plus_v1"
    assert approved["required_systemd"] == {
        "MemoryMax": 12884901888,
        "MemorySwapMax": 0,
        "CPUQuota": "100%",
        "TasksMax": 256,
        "Type": "exec",
        "RuntimeMaxSec": 1950,
        "TimeoutStopSec": 20,
        "KillMode": "control-group",
        "OOMPolicy": "stop",
    }
    assert approved["protocol"]["startup_available_gib"] == 50
    assert approved["protocol"]["runtime_available_gib"] == 22
    assert approved["protocol"]["physical_reserve_gib"] == 20


def test_same_adapter_supervisor_candidate_and_final_only_contract():
    pending = json.loads(PENDING.read_text())
    assert pending["candidate_manifest_sha256"] == "92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891"
    assert pending["source_files"][pending["import_bindings"]["symmetry_canonical_wrapper"]] == "a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae"
    assert pending["source_files"][pending["supervisor"]] == "516ca485b99175d3a7c3d97a6eb24691bb5638de511d78a8e84453cea9750107"
    assert pending["protocol"]["final_policy_only"] is True
    assert pending["protocol"]["timesteps"] == 32768
    assert pending["protocol"]["episode_steps"] == 5
    assert pending["protocol"]["reset_count"] == 24
