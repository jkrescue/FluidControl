import ast
import copy
import importlib.util
import json
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
NEW_RUNNER = STAGE / "scripts/train_p064_b_seed20261007_diverse_h5_32768_ppo.py"
OLD_RUNNER = Path(
    "/workspace/fluid_control/artifacts/"
    "p064_candidate_policy_source_20261006_immutable/scripts/"
    "train_p064_candidate_diverse_h5_32768_ppo.py"
)
OLD_SUPERVISOR = Path(
    "/workspace/fluid_control/artifacts/"
    "p064_candidate_policy_source_20261006_immutable/scripts/"
    "supervise_p064_candidate_ppo.py"
)
CFD_DRIVER = Path(
    "/workspace/fluid_control/artifacts/"
    "p064_candidate_policy_source_20261006_immutable/scripts/"
    "run_p064_candidate_projected_32768_ppo_long_cfd.py"
)
PPO_PENDING = STAGE / "docs/P064_B_SEED20261007_PPO_PENDING_20261006.json"
CFD_PENDING = STAGE / "docs/P064_B_SEED20261007_CFD_PENDING_20261006.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized_tree(path: Path):
    tree = ast.parse(path.read_text())
    tree.body[0] = ast.Expr(value=ast.Constant(value="DOCSTRING"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "seed":
                    value.value = "SEED"
    return ast.dump(tree, include_attributes=False)


def test_runner_only_changes_docstring_and_seed():
    assert normalized_tree(NEW_RUNNER) == normalized_tree(OLD_RUNNER)
    old = load(OLD_RUNNER, "p064_old_seed_runner")
    new = load(NEW_RUNNER, "p064_new_seed_runner")
    assert old.PROTOCOL["seed"] == 20261006
    assert new.PROTOCOL["seed"] == 20261007
    assert {k: v for k, v in old.PROTOCOL.items() if k != "seed"} == {
        k: v for k, v in new.PROTOCOL.items() if k != "seed"
    }


def test_seed_is_used_by_all_existing_rng_entrypoints():
    text = NEW_RUNNER.read_text()
    for call in (
        'random.seed(PROTOCOL["seed"])',
        'np.random.seed(PROTOCOL["seed"])',
        'torch.manual_seed(PROTOCOL["seed"])',
        'torch.cuda.manual_seed_all(PROTOCOL["seed"])',
        'seed=PROTOCOL["seed"]',
    ):
        assert call in text


def test_pending_profiles_are_single_seed_and_not_authorized():
    ppo = json.loads(PPO_PENDING.read_text())
    cfd = json.loads(CFD_PENDING.read_text())
    assert ppo["protocol"]["seed"] == 20261007
    assert ppo["execution_authorized"] is False
    assert "PREPARATION_ONLY" in ppo["status"]
    assert ppo["candidate_manifest_sha256"] == (
        "92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891"
    )
    assert cfd["execution_authorized"] is False
    assert "PREPARATION_ONLY" in cfd["status"]
    assert cfd["steps"] == 800 and cfd["start_time"] == 148
    for key in ("training_result", "policy", "vecnormalize", "training_approval", "training_review"):
        assert cfd["inputs"][key]["sha256"] is None


def test_actual_runner_contract_accepts_only_new_seed_profile():
    runner = load(NEW_RUNNER, "p064_seed_contract_runner")
    pending = json.loads(PPO_PENDING.read_text())
    approved = copy.deepcopy(pending)
    approved["status"] = runner.STATUS
    approved["execution_authorized"] = True
    assert runner.validate_spec(approved) is approved
    approved["protocol"]["seed"] = 20261006
    try:
        runner.validate_spec(approved)
    except ValueError as exc:
        assert "protocol" in str(exc)
    else:
        raise AssertionError("old seed unexpectedly accepted")


def test_supervisor_and_cfd_driver_are_seed_agnostic():
    for path in (OLD_SUPERVISOR, CFD_DRIVER):
        source = path.read_text()
        assert "20261007" not in source
        assert "20261006" not in source
