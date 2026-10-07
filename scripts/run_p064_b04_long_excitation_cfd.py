#!/usr/bin/env python3
"""Run one predeclared b04 120->200 OpenFOAM acquisition (never Curator/training)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IMAGE = "opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
FIELDS = ("U", "U_0", "p", "phi", "phi_0", "uniform/cumulativeContErr", "uniform/time")
START, END = 120.0, 200.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load_module(path: Path, expected_sha: str):
    require(path.is_file() and sha(path) == expected_sha, "contract helper differs")
    spec = importlib.util.spec_from_file_location("p064_b04_bound_contract", path)
    require(spec is not None and spec.loader is not None, "helper import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bound_inventory(root: Path, expected: dict[str, str]) -> dict[str, str]:
    return {name: sha(root / name) for name in expected if (root / name).is_file()}


def cgroup_int(name: str) -> int | None:
    row = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    relative = row.split("::", 1)[1].lstrip("/")
    text = (Path("/sys/fs/cgroup") / relative / name).read_text().strip()
    return None if text == "max" else int(text)


def mem_available_gib() -> float:
    row = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return int(row.split()[1]) / 2**20


def disk_available_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def host_guard(path: Path, minimum_memory: float) -> None:
    require(mem_available_gib() >= minimum_memory, f"MemAvailable below {minimum_memory} GiB")
    require(disk_available_gib(path) >= 20.0, "disk available below 20 GiB")


def verify_resources(spec: dict) -> dict:
    expected = spec["resources"]
    maximum, swap = cgroup_int("memory.max"), cgroup_int("memory.swap.max")
    require(maximum is not None and maximum <= 8 * 2**30, "controller cgroup exceeds 8 GiB")
    require(swap == 0, "controller cgroup swap is not zero")
    return {"memory_max": maximum, "memory_swap_max": swap, "declared_cpu_quota_percent": expected["cpu_quota_percent"]}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    require(text.count(old) == 1, f"{label}: expected one {old!r}")
    return text.replace(old, new)


def foam_table(points: list[list[float]]) -> str:
    return "table\n        (\n" + "".join(f"            ({t:g} {w:.10g})\n" for t, w in points) + "        )"


def replace_rear_patch(text: str, points: list[list[float]]) -> str:
    pattern = r"(\n\s*rearCylinder\s*\n\s*\{).*?(\n\s*\}\s*\n\s*frontBack)"
    require(len(re.findall(r"\brearCylinder\b", text)) == 1, "expected exactly one rearCylinder patch")
    require(len(re.findall(pattern, text, re.S)) == 1, "rearCylinder/frontBack structure differs")
    replacement = (
        "\n    rearCylinder\n    {\n        type rotatingWallVelocity;\n"
        "        origin (15 7.5 0);\n        axis (0 0 1);\n"
        f"        omega {foam_table(points)};\n"
        "        value uniform (0 0 0);\n    }\n    frontBack"
    )
    value, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    require(count == 1, "rearCylinder patch replacement failed")
    return value


def parse_table(path: Path) -> list[list[float]]:
    text = path.read_text()
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
    table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
    require(table is not None, "generated omega table absent")
    return [[float(a), float(b)] for a, b in re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]


def preflight(spec_path: Path, spec_sha: str) -> tuple[dict, dict, object]:
    require(sha(spec_path) == spec_sha, "spec SHA differs")
    cfg = json.loads(spec_path.read_text())
    require(cfg["status"] in {"PREPARATION_ONLY_NOT_AUTHORIZED", "EXECUTION_APPROVED"}, "invalid status")
    require(bool(cfg["execution_authorized"]) == (cfg["status"] == "EXECUTION_APPROVED"), "authorization mismatch")
    require(sha(Path(__file__)) == cfg["driver_sha256"], "driver SHA differs")
    helper = load_module(Path(cfg["helper"]["path"]), cfg["helper"]["sha256"])
    contract = helper.build_contract(Path(cfg["repo"]))
    require(contract["action"]["points_sha256"] == cfg["action_points_sha256"], "action contract differs")
    source = Path(cfg["source_root"])
    require(bound_inventory(source, cfg["source_inventory"]) == cfg["source_inventory"], "source restart/mesh/system inventory differs")
    require(not Path(cfg["planned_output"]).exists(), "planned output exists")
    require(sha(Path(cfg["solver_log_checker"]["path"])) == cfg["solver_log_checker"]["sha256"], "log checker differs")
    require(sha(Path(cfg["curation_adapter"])) == cfg["curation_source_sha256"], "Curator source differs")
    require(sha(Path(cfg["normalization"]["path"])) == cfg["normalization"]["sha256"], "normalization bytes differ")
    for row in cfg["initial_force_sources"].values():
        require(sha(Path(row["path"])) == row["sha256"], "initial force source differs")
    require(cfg["normalization"]["policy"] == "reuse bytes; do not refit", "normalization policy differs")
    require(cfg["solver_protocol"] == {
        "start": 120.0, "end": 200.0, "delta_t": 0.005, "steps": 16000,
        "field_write_interval": 0.1, "frames_including_restart": 801,
        "image": IMAGE, "mount": "only generated case directory writable at /case",
    }, "solver protocol differs")
    return cfg, contract, helper


def stage_case(cfg: dict, contract: dict, output: Path) -> Path:
    source = Path(cfg["source_root"])
    case = output / "case"
    case.mkdir(parents=True)
    shutil.copytree(source / "constant", case / "constant")
    shutil.copytree(source / "system", case / "system")
    shutil.copytree(source / "120", case / "120")
    u = case / "120/U"
    u.write_text(replace_rear_patch(u.read_text(), contract["action"]["points"]))
    control = case / "system/controlDict"
    text = control.read_text()
    text = replace_once(text, "startTime 0;", "startTime 120;", "controlDict")
    text = replace_once(text, "endTime 160;", "endTime 200;", "controlDict")
    text = replace_once(text, "writeInterval 2;", "writeInterval 0.1;", "controlDict")
    control.write_text(text)
    require(parse_table(u) == contract["action"]["points"], "staged table differs")
    return case


def execute(cfg: dict, contract: dict, spec_path: Path, spec_sha: str) -> None:
    output = Path(cfg["planned_output"])
    output.mkdir(parents=True)
    case = stage_case(cfg, contract, output)
    resources = verify_resources(cfg)
    host_guard(output.parent, 50.0)
    log = output / "log.pimpleFoam"
    name = cfg["container_name"]
    existing = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"], check=True, text=True, capture_output=True
    ).stdout.strip()
    require(not existing, "refusing pre-existing container name")
    cidfile = output / "solver.cid"
    command = [
        "docker", "run", "--rm", "--cidfile", str(cidfile), "--name", name, "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", "--cpus", "1", "--memory", "8g", "--memory-swap", "8g",
        "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", "--mount", f"type=bind,src={case},dst=/case", "--workdir", "/case",
        IMAGE, "pimpleFoam", "-case", "/case",
    ]
    deadline = time.monotonic() + 1200
    stopping = False

    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        if cidfile.is_file():
            cid = cidfile.read_text().strip()
            actual = subprocess.run(
                ["docker", "inspect", "--format", "{{.Name}}", cid], text=True, capture_output=True
            )
            if actual.returncode == 0 and actual.stdout.strip() == f"/{name}":
                subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        resource_log = output / "resources.jsonl"
        minimum_memory = float("inf")
        minimum_disk = float("inf")
        with log.open("wb") as stream, resource_log.open("w") as resource_stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if stopping or time.monotonic() >= deadline:
                    stop()
                    raise TimeoutError("solver interrupted or exceeded 1200 s")
                host_guard(output.parent, 22.0)
                available, free = mem_available_gib(), disk_available_gib(output.parent)
                minimum_memory, minimum_disk = min(minimum_memory, available), min(minimum_disk, free)
                resource_stream.write(json.dumps({"monotonic": time.monotonic(), "MemAvailable_GiB": available, "disk_available_GiB": free}) + "\n")
                resource_stream.flush()
                time.sleep(0.5)
            require(process.returncode == 0, f"OpenFOAM exited {process.returncode}")
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        stop()
    require(not subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"], check=True, text=True, capture_output=True
    ).stdout.strip(), "owned container remains after solver")
    subprocess.run([
        sys.executable, cfg["solver_log_checker"]["path"], "--log", str(log),
        "--expected-steps", "16000", "--expected-end", "200",
    ], check=True, stdout=(output / "solver_qc.json").open("w"))
    require(bound_inventory(Path(cfg["source_root"]), cfg["source_inventory"]) == cfg["source_inventory"], "source inventory changed during run")
    frames = []
    for path in case.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if START <= value <= END and (path / "U").is_file() and (path / "p").is_file():
            frames.append(value)
    frames = sorted(round(value, 10) for value in frames)
    require(frames == [round(START + index * 0.1, 10) for index in range(801)], "saved 0.1 grid differs")
    result = {
        "status": "P064_B04_LONG_EXCITATION_OPENFOAM_COMPLETE_PENDING_CURATOR",
        "spec": str(spec_path), "spec_sha256": spec_sha,
        "driver_sha256": sha(Path(__file__)), "action_points_sha256": cfg["action_points_sha256"],
        "source_inventory": cfg["source_inventory"], "resources": resources,
        "initial_force_sources": cfg["initial_force_sources"],
        "solver_log_sha256": sha(log), "solver_qc_sha256": sha(output / "solver_qc.json"),
        "resource_log_sha256": sha(resource_log), "minimum_MemAvailable_GiB": minimum_memory,
        "minimum_disk_available_GiB": minimum_disk,
        "saved_frames": len(frames), "first_time": min(frames), "last_time": max(frames),
        "curator_executed": False, "training_executed": False,
        "next_path": cfg["curation_adapter"], "normalization_policy": cfg["normalization"]["policy"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg, contract, _ = preflight(args.spec.resolve(), args.spec_sha256)
    if not args.execute:
        print("P064_B04_LONG_EXCITATION_PREFLIGHT_PASS_NOT_RUNNING")
        return
    require(cfg["execution_authorized"], "execution not authorized")
    execute(cfg, contract, args.spec.resolve(), args.spec_sha256)


if __name__ == "__main__":
    main()
