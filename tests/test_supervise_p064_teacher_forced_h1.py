import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "supervise_p064_teacher_forced_h1.py"


def load():
    spec = importlib.util.spec_from_file_location("p064_tf_supervisor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_resource_and_protocol_contract(tmp_path):
    module = load()
    assert module.RESOURCES["allocator_bytes"] == 6 * 2**30
    assert module.RESOURCES["startup_available_gib"] == 50
    assert module.RESOURCES["runtime_available_gib"] == 22
    assert module.UNIT["MemorySwapMax"] == "0"
    assert module.UNIT["RuntimeMaxUSec"] == "10min 30s"
    assert module.UNIT["Type"] == "exec"


def test_pending_spec_validates_without_execution_or_payload_reads(tmp_path):
    module = load()
    module.ROOT = tmp_path
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("# worker\n")
    copied_supervisor = tmp_path / "supervisor.py"
    copied_supervisor.write_bytes(SCRIPT.read_bytes())
    module.__file__ = str(copied_supervisor)
    inputs = {}
    for name in ("data_manifest", "normalization", "normalization_manifest", "config",
                 "aerodynamic_model", "candidate_manifest", "training_config",
                 "autoregressive_reference"):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        inputs[name] = {"path": path.relative_to(tmp_path).as_posix(), "sha256": digest(path)}
    source_map = {}
    for index in range(411):
        path = source / f"bound_{index:03d}.py"
        path.write_text(str(index))
        source_map[path.relative_to(source).as_posix()] = digest(path)
    receipt = tmp_path / "source_receipt.json"
    receipt.write_text(json.dumps({
        "status": "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN",
        "final_files_sha256": source_map,
    }))
    inputs["source_receipt"] = {
        "path": receipt.relative_to(tmp_path).as_posix(), "sha256": digest(receipt)}
    spec = {
        "status": module.PENDING, "execution_authorized": False,
        "official_image_id": module.IMAGE,
        "container_name": "p064-teacher-forced-h1-20261006",
        "resources": module.RESOURCES,
        "protocol": {"mode": "teacher_forced_h1", "cases": 6, "starts_per_case": 100,
            "dual_steps": 600, "flow_forwards": 600, "aerodynamic_forwards": 600,
            "total_submodel_forwards": 1200, "batch_size": 1, "history_k": 1,
            "optimizer_steps": 0, "precision": "high_tf32"},
        "worker": {"path": "worker.py", "sha256": digest(worker)},
        "supervisor": {"path": "supervisor.py", "sha256": digest(copied_supervisor)},
        "inputs": inputs, "python_source_root": "source", "output": "output",
        "unit": "future.service",
    }
    actual = module.validate_spec(spec, allow_pending=True)
    assert actual[0] == worker
    command = module.command(spec, *actual)
    assert command.count("--mode") == 1
    assert command[command.index("--mode") + 1] == "teacher_forced_h1"
    assert "--execute" not in command
    assert command[:2] == ["docker", "run"]
    assert module.IMAGE in command


def test_execute_path_has_continuous_reserve_deadline_and_cleanup():
    text = SCRIPT.read_text()
    assert 'available_gib() >= 50' in text
    assert 'row["mem_available_gib"] >= 22' in text
    assert 'elapsed < 600' in text
    assert "os.killpg(process.pid, signal.SIGTERM)" in text
    assert "os.killpg(process.pid, signal.SIGKILL)" in text
    assert 'start_new_session=True' in text
    assert "signal.signal(signal.SIGTERM, interrupted)" in text
    assert '"OMP_NUM_THREADS=1"' in text
    assert '"MKL_NUM_THREADS=1"' in text
    assert '"OPENBLAS_NUM_THREADS=1"' in text
    assert 'torch' not in text


def test_approval_and_execution_are_fail_closed(tmp_path):
    module = load()
    assert module.PENDING != module.APPROVED
    source = SCRIPT.read_text()
    assert 'allow_pending=not args.execute' in source
    assert 'args.spec_sha256' in source
    assert 'execution_authorized' in source
