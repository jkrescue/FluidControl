"""Thin host lifecycle guard for one approved P064 pressure-aux training."""
from __future__ import annotations

import argparse, hashlib, json, os, signal, subprocess, time
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise ValueError("MemAvailable missing")


def properties(unit):
    keys = ("InvocationID", "ActiveState", "MemoryMax", "MemorySwapMax",
            "CPUQuotaPerSecUSec", "TasksMax", "RuntimeMaxUSec")
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=" + ",".join(keys)], check=True, text=True, capture_output=True)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.spec) == args.spec_sha256, "spec SHA")
    spec = json.loads(args.spec.read_text())
    require(sha(__file__) == spec["supervisor_sha256"], "supervisor SHA")
    require(sha(spec["argv"][2]) == spec["runner_sha256"], "runner SHA")
    require(sha(spec["pressure_helper"]["path"]) == spec["pressure_helper"]["sha256"], "helper SHA")
    require(Path(spec["planned_output"]).resolve() ==
            Path(spec["argv"][spec["argv"].index("--output") + 1]).resolve(), "output argv")
    require(not Path(spec["planned_output"]).exists(), "output exists")
    require(available() >= 50, "startup memory")
    command = list(spec["argv"])
    if not args.execute:
        command.remove("--execute")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": spec["runtime_pythonpath"]})
    if not args.execute:
        subprocess.run(command, check=True, env=env)
        print("P064_TOTAL_PRESSURE_AUX_SUPERVISOR_PREPARATION_PASS")
        return
    require(spec["execution_authorized"] is True, "execution not authorized")
    props = properties(spec["planned_unit"])
    require(props["InvocationID"] == os.environ.get("INVOCATION_ID"), "invocation")
    require(props["ActiveState"] == "active", "unit inactive")
    require(props["MemoryMax"] == str(24 * 2**30), "MemoryMax")
    require(props["MemorySwapMax"] == "0", "swap")
    require(props["CPUQuotaPerSecUSec"] == "8s", "CPU quota")
    require(props["TasksMax"] == "2048", "TasksMax")
    require(props["RuntimeMaxUSec"] == "1h 1min", "runtime max")
    child = subprocess.Popen(command, env=env, start_new_session=True)
    deadline = time.monotonic() + 3600
    while child.poll() is None:
        if time.monotonic() > deadline or available() < 22:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise RuntimeError("runtime deadline or memory guard")
        time.sleep(0.5)
    require(child.returncode == 0, f"training child exit {child.returncode}")
    print(json.dumps({"status": "P064_TOTAL_PRESSURE_AUX_SUPERVISOR_COMPLETE",
        "unit": spec["planned_unit"], "invocation": props["InvocationID"],
        "limits": props}, sort_keys=True))


if __name__ == "__main__": main()
