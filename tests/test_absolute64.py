import ast
import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
RUNNER = HERE / "scripts/train_p064_absolute64.py"
ORIGINAL = Path(
    "/workspace/fluid_control/"
    "artifacts/fcp064_training_source_20261006_immutable/"
    "scripts/train_fcp064_controlled_aero_ab.py"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("absolute64", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_source_and_fixed_protocol():
    assert sha(ORIGINAL) == "8066f4a1e092c566e1b84f706ba56737afa06e13998446fee2f79dc2590920dd"
    module = load_runner()

    class Schedule:
        @staticmethod
        def compile_schedule(parent_order, arm):
            assert arm == "B"
            return list(range(256))

        @staticmethod
        def schedule_sha256(value):
            assert value == list(range(256))
            return "schedule"

    protocol = module.protocol("B", Schedule, [1, 2, 3])
    assert protocol["training_windows"] == 512
    assert protocol["optimizer_steps"] == 64
    assert protocol["accumulation_windows"] == 8
    assert protocol["schedule_epochs"] == 2
    assert protocol["b00_windows"] == 128
    assert protocol["b00_windows_per_schedule_epoch"] == 64
    assert protocol["learning_rate"] == 1.5625e-7
    assert protocol["objective"] == "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE"
    assert protocol["update32_reference_tensor_sha256"] == module.ORIGINAL_B_UPDATE32_TENSOR_SHA
    assert module.ALLOCATOR_BYTES == 16 * 2**30


def test_loop_keeps_one_optimizer_and_two_exact_schedule_epochs():
    tree = ast.parse(RUNNER.read_text())
    adam_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "AdamW"
    ]
    assert len(adam_calls) == 1
    text = RUNNER.read_text()
    assert "for schedule_epoch in range(2):" in text
    assert "for local_step in range(1, 33):" in text
    assert "step = schedule_epoch * 32 + local_step" in text
    assert "epoch_observed != list(scheduled.schedule)" in text
    assert "update32_actual == ORIGINAL_B_UPDATE32_TENSOR_SHA" in text
    assert "updates_total=64" in text
    assert "diagnostic_counts=[0, 512]" in text
    assert "panels.append(dict(consumed_windows=256" not in text
    assert "panels.append(dict(consumed_windows=512, **panel()))" in text


def test_only_terminal_update64_is_saved():
    text = RUNNER.read_text()
    assert text.count("save_checkpoint(") == 1
    assert "checkpoint_epoch=2" in text
    assert "epoch=2" in text
    assert 'model_file="FNO.0.2.mdlus"' in text
    assert 'state_file="checkpoint.0.2.pt"' in text
    assert 'checkpoint_epoch=2' in text
    assert 'args.output.name + "_update32_comparison.json"' in text
    assert "official_fresh_reload_verified=True" in text
    assert "scientific_admission=False" in text
    assert 'if args.arm != "B"' in text
