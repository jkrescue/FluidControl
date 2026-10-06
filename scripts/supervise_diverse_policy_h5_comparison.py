"""Bounded UMA process-group supervisor; no Docker/CFD ownership or admission."""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

BASE_SHA = "56c17443245c60f51b4e4c8f15b1851136a091c73441e5ecd76eebe0695f5e74"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def run(args):
    require(sha(args.approval) == args.approval_sha256, "approval SHA differs")
    spec = json.loads(args.approval.read_text())
    require(
        spec["status"] == "DIVERSE_POLICY_H5_COMPARISON_EXECUTION_APPROVED"
        and spec["execution_authorized"] is True,
        "execution not authorized",
    )
    require(
        spec["source_files"].get(str(Path(__file__).resolve())) == sha(__file__),
        "supervisor source not bound",
    )
    base_path = Path(spec["supervisor_base"])
    require(sha(base_path) == BASE_SHA, "reviewed lifecycle helper differs")
    module_spec = importlib.util.spec_from_file_location(
        "uma_supervisor_base", base_path
    )
    base = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(base)
    runner = Path(spec["runner"])
    require(
        spec["source_files"].get(str(runner)) == sha(runner), "runner source differs"
    )
    limits = base.cgroup_limits()  # <=12GiB, swap.max exactly0.
    require(
        base.memory_ok(base.memory(), startup=True),
        "MemAvailable startup50GiB required",
    )
    parent = Path(spec["supervision_output"])
    parent.mkdir(parents=False, exist_ok=False)
    output = Path(spec["output"])
    require(
        output.parent == parent and not output.exists(),
        "exclusive payload child required",
    )
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(spec["pythonpath"]),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        DIVERSE_POLICY_H5_EVAL_SUPERVISED=args.approval_sha256,
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        spec["python"],
        str(runner),
        "--approval",
        str(args.approval),
        "--approval-sha256",
        args.approval_sha256,
        "--output",
        str(output),
        "--execute",
    ]
    begin = time.monotonic()
    process = None
    observations = []
    error = None

    def interrupted(signum, frame):
        raise RuntimeError(f"supervisor received signal {signum}")

    previous = {
        s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with (
            (parent / "worker.log").open("x") as log,
            (parent / "memory.jsonl").open("x") as trace,
        ):
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            while True:
                row = {"elapsed": time.monotonic() - begin, **base.memory()}
                observations.append(row)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                require(base.memory_ok(row), "MemAvailable runtime22GiB reserve guard")
                require(row["elapsed"] < 240, "240-second deadline")
                status = process.poll()
                if status is not None:
                    require(status == 0, f"worker exited {status}")
                    break
                time.sleep(0.5)
        result = json.loads((output / "result.json").read_text())
        require(
            result["status"] == "DIVERSE_POLICY_H5_COMPARISON_COMPLETE_NOT_ADMISSION"
            and result["optimizer_steps"] == 0
            and result["fno_tensors_unchanged"] is True,
            "worker completion differs",
        )
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if process is not None:
            base.stop(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        with (parent / "supervisor_result.json").open("x") as f:
            json.dump(
                {
                    "approval_sha256": args.approval_sha256,
                    "command": command,
                    "limits": limits,
                    "error": error,
                    "returncode": None if process is None else process.poll(),
                    "minimum_available_bytes": min(
                        (x["MemAvailable"] for x in observations), default=None
                    ),
                    "result_sha256": sha(output / "result.json")
                    if (output / "result.json").exists()
                    else None,
                    "scientific_admission": False,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    require(arguments.execute, "explicit --execute required")
    run(arguments)
