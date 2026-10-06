#!/usr/bin/env python3
"""Convert the fixed 96 saved OpenFOAM frames and verify 16 official-reader HDFs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path("/workspace/fluid_control")
EXPECTED_STATUS = "PROJECTED_POLICY_H1_H5_CONVERSION_EXECUTION_APPROVED"
EXPECTED_IMAGE = "sha256:24205c9677d39c95221eb903988094dd7a228fc41a2054df3eaa13f80e465fcb"
STARTUP_AVAILABLE = 50 * 2**30
RUNTIME_AVAILABLE = 22 * 2**30
DEADLINE = 1200


class ContinuousGuard:
    """Independent host-memory/deadline watcher around synchronous native calls."""
    def __init__(self, output, deadline_at):
        self.output = output
        self.deadline_at = deadline_at
        self.stop_event = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                resource_row(self.output, "continuous", self.deadline_at)
            except Exception as error:
                self.failure = str(error)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.stop_event.wait(.5)

    def start(self):
        self.thread.start()

    def check(self):
        require(self.failure is None, self.failure or "resource guard failure")

    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=2)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            if guard is not None:
                guard()
    return digest.hexdigest()


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.rename(temporary, path)


def checked(root, item):
    require(set(item) == {"path", "sha256"}, "bound file schema")
    relative = Path(item["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "bound path escape")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and sha(path) == item["sha256"],
            f"bound file differs: {relative}")
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree_files(root):
    require(root.is_dir() and not root.is_symlink(), "tree root")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "source symlink")
        if path.is_file():
            result[str(path.relative_to(root))] = {"size": path.stat().st_size,
                                                   "sha256": sha(path)}
    return result


def selected_inventory(source, selection, guard=None):
    paths = set()
    for record in selection["records"]:
        paths.update(value for frame in record["frames"] for value in frame["files"].values())
        case = source / f"case_{record['branch']}"
        for name in ("constant", "system"):
            for path in (case / name).rglob("*"):
                if path.is_file():
                    paths.add(str(path.relative_to(source)))
    result = {}
    for relative in sorted(paths):
        path = source / relative
        require(path.is_file() and not path.is_symlink(), "selected source file")
        result[relative] = {"size": path.stat().st_size, "sha256": sha(path, guard)}
    return result


def copy_selected(source, destination, selection, inventory, guard=None):
    destination.mkdir(exist_ok=False)
    for relative in inventory:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        if guard is not None:
            guard()
    require(selected_inventory(destination, selection, guard) == inventory,
            "copied inventory differs")


def resource_row(output, phase, deadline_at=None):
    value = memory_available()
    row = {"time": time.time(), "phase": phase, "MemAvailable": value}
    with (output / "resources.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    require(value >= RUNTIME_AVAILABLE, "MemAvailable below runtime floor")
    if deadline_at is not None:
        require(time.monotonic() < deadline_at, "absolute conversion deadline")
    return row


def bounded(command, log, output, deadline, deadline_at):
    with log.open("x") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        begin = time.monotonic()
        try:
            while child.poll() is None:
                resource_row(output, "child", deadline_at)
                require(time.monotonic() - begin < deadline, "child deadline")
                time.sleep(.5)
            require(child.returncode == 0, f"child failed {child.returncode}")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def validated_owned_container(candidate_id, name, image, command, case):
    inspect = json.loads(subprocess.check_output(
        ["docker", "inspect", candidate_id], text=True, timeout=10))[0]
    mounts = {(row["Source"], row["Destination"], row["RW"])
              for row in inspect.get("Mounts", [])}
    require(inspect["Name"] == "/" + name and inspect["Image"] == image
            and inspect["Config"]["Cmd"] == command
            and mounts == {(str(case), "/case", True)},
            "owned container identity")
    return inspect


def export_branch(case, role, image, output, deadline_at):
    name = f"projected-h1h5-convert-{role}-20261006"
    require(not subprocess.check_output(
        ["docker", "container", "ls", "-aq", "--filter", f"name=^{name}$"],
        text=True, timeout=10).strip(), "owned container name exists")
    cid = None
    try:
        command = ["foamToVTK", "-case", "/case", "-fields", "(U p)",
                   "-no-boundary", "-name", "VTK_replay"]
        create = ["docker", "create", "--name", name, "--runtime", "runc",
                  "--network", "none", "--read-only", "--memory", "4g",
                  "--memory-swap", "4g", "--cpus", "1", "--pids-limit", "64",
                  "--security-opt", "no-new-privileges", "-e", "CUDA_VISIBLE_DEVICES=",
                  "-e", "NVIDIA_VISIBLE_DEVICES=void", "-v", f"{case}:/case:rw",
                  image, *command]
        try:
            cid = subprocess.check_output(create, text=True, timeout=30).strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, RuntimeError):
            ids = []
            for _ in range(10):
                ids = subprocess.check_output(
                    ["docker", "container", "ls", "-aq", "--no-trunc",
                     "--filter", f"name=^{name}$"], text=True, timeout=10).split()
                if ids:
                    break
                time.sleep(.5)
            if len(ids) == 1:
                candidate_id = ids[0]
                validated_owned_container(candidate_id, name, image, command, case)
                cid = candidate_id
            raise
        require(cid, "empty container id")
        bounded(["docker", "start", "-a", cid], output / f"{role}_foamToVTK.log",
                output, 600, deadline_at)
        inspect = json.loads(subprocess.check_output(
            ["docker", "inspect", cid], text=True, timeout=10))[0]
        atomic_json(output / f"{role}_container.json", inspect)
        require(inspect["State"]["ExitCode"] == 0 and not inspect["State"]["OOMKilled"],
                "foamToVTK terminal")
    finally:
        if cid:
            validated_owned_container(cid, name, image, command, case)
            subprocess.run(["docker", "rm", "-f", cid], check=True, timeout=30,
                           capture_output=True)
            remaining = subprocess.check_output(
                ["docker", "container", "ls", "-aq", "--no-trunc"],
                text=True, timeout=10).split()
            require(cid not in remaining, "owned container remains")


def series_mapping(case):
    document = json.loads((case / "VTK_replay/case.vtm.series").read_text())
    result = {}
    for row in document.get("files", []):
        value = float(row["time"])
        stem = Path(row["name"]).stem
        path = case / "VTK_replay" / stem / "internal.vtu"
        require(value not in result and path.is_file(), "VTK series member")
        result[value] = path
    require(len(result) == 48, "expected 48 exported VTK frames")
    return result


def sample_branch(case, role, selection, sampler, output, guard):
    mapping = series_mapping(case)
    packets = {}
    for record in [row for row in selection["records"] if row["branch"] == role]:
        values = []
        for frame in record["frames"]:
            expected = frame["time"]
            matches = [path for value, path in mapping.items() if abs(value - expected) <= 1e-8]
            require(len(matches) == 1, "VTK time match")
            view = output / "sample_views" / role / str(frame["global_index"])
            (view / "frame").mkdir(parents=True, exist_ok=False)
            copied_vtu = view / "frame/internal.vtu"
            source_sha = sha(matches[0], guard)
            shutil.copy2(matches[0], copied_vtu)
            require(sha(copied_vtu, guard) == source_sha, "sample view copy differs")
            packet_path = output / "packets" / role / f"{frame['global_index']:04d}.npz"
            packet_path.parent.mkdir(parents=True, exist_ok=True)
            sampler.sample_frame(view, packet_path)
            guard()
            with np.load(packet_path, allow_pickle=False) as loaded:
                values.append({key: loaded[key] for key in loaded.files})
        packets[record["start_index"]] = values
    return packets


def output_path(spec):
    relative = Path(spec["output"])
    require(not relative.is_absolute() and ".." not in relative.parts, "output path escape")
    path = ROOT / relative
    require(ROOT in path.parents, "output outside repo")
    require(not path.exists() and not path.is_symlink(), "exclusive output required")
    cursor = path.parent
    while cursor != ROOT:
        require(cursor.exists() and cursor.is_dir() and not cursor.is_symlink(),
                "output ancestor")
        cursor = cursor.parent
    require(cursor == ROOT, "output outside repo")
    return path


def verify_outer_unit(spec):
    expected = {"MemoryMax": str(12 * 2**30), "MemorySwapMax": "0",
                "CPUQuotaPerSecUSec": "1s", "TasksMax": "64",
                "RuntimeMaxUSec": "20min"}
    require(spec["required_systemd"] == expected, "systemd contract")
    unit = spec["unit"]
    require(isinstance(unit, str) and unit.endswith(".service"), "unit name")
    keys = ("MainPID", *expected)
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", unit,
         *[item for key in keys for item in ("-p", key)]], text=True, timeout=10)
    actual = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(actual.get("MainPID") == str(os.getpid()), "outer unit PID")
    require(all(actual.get(key) == value for key, value in expected.items()),
            "outer unit resources")


def validate_spec(spec, expected_output):
    require(spec["status"] == EXPECTED_STATUS and spec["execution_authorized"] is True,
            "execution not approved")
    require(spec["starts"] == [0,100,200,300,400,500,600,700]
            and spec["horizon"] == 5 and spec["frames"] == 96, "fixed selection")
    require(spec["deadlines"] == {"conversion_seconds": 1200,
                                  "future_inference_seconds": 600}, "deadlines")
    require(spec["resources"] == {"startup_available_gib":50,
                                  "runtime_available_gib":22,
                                  "supervisor_memory_gib":12,
                                  "supervisor_swap_gib":0}, "resources")
    require(spec["image"] == EXPECTED_IMAGE, "OpenFOAM image identity")
    require(spec["python"] == ".venv-curator-py312/bin/python", "Python identity")
    require(spec["runtime_versions"] == {
        "torch":"2.14.1","numpy":"2.5.3","pyvista":"0.49.0",
        "physicsnemo-curator":"0.1.0","nvidia-physicsnemo":"2.2.2"},
        "runtime versions")
    require(isinstance(spec.get("unit"), str) and isinstance(spec.get("required_systemd"), dict),
            "outer unit schema")
    inventory = spec["selected_source_inventory"]
    require(isinstance(inventory, dict) and inventory, "precomputed source inventory")
    for relative, identity in inventory.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "inventory escape")
        require(set(identity) == {"size","sha256"} and type(identity["size"]) is int
                and identity["size"] >= 0 and isinstance(identity["sha256"], str)
                and len(identity["sha256"]) == 64, "inventory identity")
    require(output_path(spec) == expected_output, "output CLI/spec mismatch")


def execute(spec, spec_sha, output):
    require(memory_available() >= STARTUP_AVAILABLE, "startup MemAvailable below 50GiB")
    validate_spec(spec, output)
    verify_outer_unit(spec)
    deadline_at = time.monotonic() + DEADLINE
    guard = lambda: resource_row(output, "conversion", deadline_at)
    require(Path(sys.executable).resolve() == (ROOT / spec["python"]).resolve(),
            "executing Python differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void", "CPU-only environment")
    versions = {name:importlib.metadata.version(name) for name in spec["runtime_versions"]}
    require(versions == spec["runtime_versions"], "installed runtime differs")
    require(spec["image"] == EXPECTED_IMAGE and subprocess.check_output(
        ["docker", "image", "inspect", EXPECTED_IMAGE, "--format", "{{.Id}}"],
        text=True, timeout=10).strip() == EXPECTED_IMAGE, "OpenFOAM image")
    driver_path = checked(ROOT, spec["driver"])
    require(driver_path.resolve() == Path(__file__).resolve(), "executed driver differs")
    core_path = checked(ROOT, spec["core"]); sampler_path = checked(ROOT, spec["sampler"])
    official_path = checked(ROOT, spec["official_reader_adapter"])
    checked(ROOT, spec["official_reader_source"])
    progress_path = checked(ROOT, spec["progress"]); checked(ROOT, spec["result"])
    checked(ROOT, spec["review"])
    core = load_module(core_path, "projected_h1h5_core")
    sampler = load_module(sampler_path, "projected_h1h5_sampler")
    official = load_module(official_path, "projected_h1h5_official_reader")
    source = (ROOT / spec["source_root"]).resolve()
    expected_source = (ROOT / "artifacts/exploratory_projected_32768_ppo_long_cfd_20261006").resolve()
    require(source == expected_source and source.is_dir() and not source.is_symlink(), "source root")
    document = json.loads(progress_path.read_text())
    selection = core.make_selection(document, source)
    selection["source_spec_sha256"] = spec_sha
    output.mkdir(exist_ok=False)
    monitor = ContinuousGuard(output, deadline_at); monitor.start()
    try:
        atomic_json(output / "selection.json", selection)
        guard(); monitor.check()
        inventory = selected_inventory(source, selection, guard)
        require(inventory == spec["selected_source_inventory"], "approved source inventory differs")
        atomic_json(output / "source_inventory.json", inventory)
        copied = output / "selected_cases"
        copy_selected(source, copied, selection, inventory, guard)
        receipts = []
        for role in ("mpc", "zero"):
            case = copied / f"case_{role}"
            export_branch(case, role, EXPECTED_IMAGE, output, deadline_at)
            packets = sample_branch(case, role, selection, sampler, output, monitor.check)
            for record in [row for row in selection["records"] if row["branch"] == role]:
                hdf = output / "hdf" / f"{role}_start_{record['start_index']:04d}.h5"
                receipt = core.pack_and_verify_mini_hdf(
                    record, packets[record["start_index"]], hdf, official.read_trajectory)
                monitor.check()
                receipts.append({**receipt, "branch": role,
                                 "start_index": record["start_index"]})
        require(len(receipts) == 16, "HDF count")
        require(len({(row["mask_sha256"],row["x_sha256"],row["y_sha256"])
                     for row in receipts}) == 1, "global grid/mask changed")
        require(selected_inventory(source, selection, guard) == inventory,
                "original source changed")
        atomic_json(output / "result.json", {
            "status":"PROJECTED_POLICY_H1_H5_CONVERSION_COMPLETE_NOT_ADMISSION",
            "source_spec_sha256":spec_sha,"selection_sha256":sha(output/"selection.json"),
            "source_inventory_sha256":sha(output/"source_inventory.json"),
            "hdf":receipts,"frames":96,"trajectories":16,"optimizer_steps":0,
            "model_loaded":False,"cfd_executed":False,"scientific_admission":False,
            "source_unchanged":True,"owned_containers_cleaned":True})
    finally:
        monitor.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--spec-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    require(sha(args.spec)==args.spec_sha256,"spec SHA")
    spec=json.loads(args.spec.read_text())
    if not args.execute:
        print(json.dumps({"status":"PREPARATION_ONLY_NOT_EXECUTED",
                          "spec_sha256":args.spec_sha256,"output":str(args.output)}))
        return
    execute(spec,args.spec_sha256,args.output)


if __name__=="__main__":
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(RuntimeError("SIGTERM")))
    main()
