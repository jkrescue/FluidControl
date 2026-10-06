"""Training launcher CPU-only contract tests; no Docker/model/data execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
path = HERE / "run_p028_flow_training.py"
if not path.exists():
    path = HERE.parent / "scripts/run_p028_flow_training.py"
spec = importlib.util.spec_from_file_location("p028_training_launcher", path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
BASE = Path("/workspace/fluid_control/scripts/run_p028_resource_probe.py")


def approval():
    return dict(status="FC_P028_EXECUTION_APPROVED", mode="train", resources={"allocator_fraction": .06},
                source_root="/immutable/source", source_sha256={"scripts/train_p028_flow_rollout.py": "a" * 64},
                config={"path": "/config.yaml"}, train_audit={"path": "/audit.json"},
                resource_probe_receipt={"path": "/r3/result.json", "sha256": "b" * 64},
                parent_manifest={"path": "/candidate/dual_model_manifest.json"},
                data={k: {"root": "/data/" + k} for k in ("base", "train8", "train16")})


def result():
    return dict(status="FC_P028_TRAINING_COMPLETE_NOT_ADMISSION", mode="train",
                optimizer_steps=171, training_windows=1368, scientific_admission=False,
                official_fresh_reload_verified=True, source_spec=approval(),
                sampler_order_sha256="177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f",
                flow_initial_tensor_sha256="a" * 64, flow_terminal_tensor_sha256="b" * 64,
                frozen_aerodynamic_tensor_sha256="c" * 64, training_protocol_sha256="d" * 64,
                records=[dict(update=i, consumed_windows=i * 8) for i in range(1, 172)])


def test_train_command_resources_readonly_receipt_aliases():
    args = launcher.create_command(approval(), Path("/approval.json"), "e" * 64, Path("/output"))
    assert args[args.index("--mode") + 1] == "train"
    assert args[args.index("--memory") + 1] == args[args.index("--memory-swap") + 1] == "12g"
    assert args[args.index("timeout") + 3] == "14400"
    assert args[args.index("--allocator-fraction") + 1] == ".06"
    mounts = launcher.readonly_mounts(approval(), Path("/approval.json"))
    assert (Path("/r3/result.json"), Path("/r3/result.json"), True) in mounts
    for family in ("base", "train8", "train16"):
        assert (Path("/data") / family / "train", Path("/workspace") / family / "train", True) in mounts
    assert all(read_only for _, _, read_only in mounts)


def test_load_spec_resource_mode_and_missing_receipt_rejected(tmp_path):
    p = tmp_path / "approval.json"
    for mutation in (None, "mode", "receipt"):
        value = approval()
        if mutation == "mode":
            value["mode"] = "resource-probe"
        if mutation == "receipt":
            value.pop("resource_probe_receipt")
        p.write_text(json.dumps(value))
        if mutation is None:
            assert launcher.load_spec(p, launcher.sha256(p)) == value
        else:
            with pytest.raises(RuntimeError):
                launcher.load_spec(p, launcher.sha256(p))


@pytest.mark.parametrize("key,value", [("optimizer_steps", 170), ("training_windows", 1360),
    ("official_fresh_reload_verified", False), ("scientific_admission", True),
    ("flow_terminal_tensor_sha256", None), ("mode", "resource-probe"), ("source_spec", {}),
    ("records", []), ("sampler_order_sha256", "e" * 64)])
def test_terminal_malformed_rejected(key, value):
    valid = result()
    launcher.validate_training_result(valid, approval())
    bad = copy.deepcopy(valid)
    bad[key] = value
    with pytest.raises(RuntimeError):
        launcher.validate_training_result(bad, approval())


def test_lifecycle_functions_ast_unchanged_from_reviewed_resource_launcher():
    before = {n.name: n for n in ast.parse(BASE.read_text()).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in ("host_memory", "validate_created", "inspect_container", "cleanup_owned",
                 "running_gpu_containers", "recover_created_cid", "validate_mount_sources"):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)


def test_execute_lifecycle_only_deadline_and_result_validation_changed():
    before = next(n for n in ast.parse(BASE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    after = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value in (900, 14400):
                node.value = 900
            if isinstance(node.value, str):
                node.value = node.value.replace("14400", "900").replace("training result absent", "resource probe result absent")
            return node
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "validate_training_result":
                    return None
                if call.args and isinstance(call.args[-1], ast.Constant) and call.args[-1].value == "no-update resource result contract differs":
                    return None
            return self.generic_visit(node)
    assert ast.dump(Normalize().visit(before), include_attributes=False) == ast.dump(Normalize().visit(after), include_attributes=False)
