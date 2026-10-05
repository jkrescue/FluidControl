from pathlib import Path
import subprocess
import train_fcp026_history as trainer

SCRIPTS = Path(trainer.__file__).resolve().parent


def test_launcher_syntax():
    subprocess.run(
        ["bash", "-n", str(SCRIPTS / "run_fcp026_history_training_spark.sh")],
        check=True,
    )


def test_separate_arms_protocol_and_memory_guard():
    source = (SCRIPTS / "run_fcp026_history_training_spark.sh").read_text()
    assert "fcp026_history_training_execution.lock" in source
    assert '"$arm" == 1 || "$arm" == 4' in source
    assert "a['effective_protocol']==trainer.protocol(arm)" in source
    assert (
        "--flow-parent /workspace/candidate/flow --aerodynamic-parent /workspace/candidate/aerodynamic"
        in source
    )
    assert '--history-k "$arm"' in source
    assert (
        "14540 docker run" in source
        and "14400 python" in source
        and "SECONDS-begin>14500" in source
    )
    assert "--memory 12g" in source and "--allocator-fraction .06" in source
    assert "available<20*1024*1024 || free<20*1024*1024" in source
    assert "a>=50*1024*1024 && f>=30*1024*1024" in source


def test_launcher_preflight_and_guard_fail_closed():
    source = (SCRIPTS / "run_fcp026_history_training_spark.sh").read_text()
    assert "inventory_receipt_sha256" in source and "pre['trainer_sha256']" in source
    assert "gpu_pids=$(timeout" in source and 'if ! kill -0 "$watch_pid"' in source
    loop = source.split('while kill -0 "$run_pid"', 1)[1].split("done", 1)[0]
    assert "owned_stop || true" in loop and 'kill -TERM "$run_pid"' not in loop
    assert (
        "raw_source_view" not in source
        and "causal_audit" not in source
        and "prior_result" not in source
    )
    assert "O_NOFOLLOW" in source and "POSIX_FADV_DONTNEED" in source
    assert "runtime_container.json" in source


def test_preflight_no_batch_iteration_and_original_order():
    source = (SCRIPTS / "preflight_fcp026_training_cpu.py").read_text()
    assert "list(iter(loader.sampler))" in source
    assert "iter(loader)" not in source
    assert "cuda.is_available()" in source
    assert "full_hdf_hash_performed=False" in source
