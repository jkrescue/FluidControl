#!/usr/bin/env python3
"""Paired ten-cycle exploratory real-CFD feedback with canonical-history H2 cost.

No execution is authorized by this file.  ``--execute`` additionally requires
an exact separately reviewed spec.  OpenFOAM remains the real solver transport;
HydroGym is not used as a solver.  The original K1 formal failure is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np


IMAGE = "opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
STATUS = "EXPLORATORY_PAIRED_CANONICAL_HISTORY_H2_REAL_CFD_EXECUTION_APPROVED"
SOURCE_TIME = 148.0
STEPS = 10
STATE_ABS_LIMIT = 12.155583713529318
HOST_START_AVAILABLE_GIB = 50
HOST_RUNTIME_GIB = 22
VERSIONS = {"torch": "2.14.1", "numpy": "2.5.3", "h5py": "3.16.0",
            "pyvista": "0.49.0", "physicsnemo-curator": "0.1.0",
            "nvidia-physicsnemo": "2.2.2"}
RUNTIME_PINS = {
    "physicsnemo/datapipes/readers/hdf5.py":
        "cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0",
    "physicsnemo_curator/domains/mesh/sources/vtk.py":
        "a717a49d51c5695d306776d5df202f97e6ec918a0dcbd7c0e02437a182280b50",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def confined(repo, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "relative project path required")
    path = (repo / relative).resolve()
    require(path.is_relative_to(repo.resolve()), "project path escapes repository")
    return path


def tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "regular source tree required")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "source tree symlink")
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path)
        else:
            require(path.is_dir(), "source tree nonregular entry")
    return result


def atomic_json(path, payload, *, exclusive=False):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def memory():
    rows = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemFree:", "MemAvailable:"))}
    require(set(rows) == {"MemFree", "MemAvailable"}, "host memory fields")
    return rows


def host_guard(*, startup=False):
    row = memory()
    available = HOST_START_AVAILABLE_GIB if startup else HOST_RUNTIME_GIB
    require(row["MemAvailable"] >= available * 2**30, "host available-memory reserve")
    return row


def validate_runtime(repo):
    require(Path(sys.prefix).resolve() == (repo / ".venv-curator-py312").resolve(),
            "exact official Curator/PhysicsNeMo environment required")
    require({key: importlib.metadata.version(key) for key in VERSIONS} == VERSIONS,
            "runtime package versions differ")
    package_root = Path(sys.prefix) / "lib/python3.12/site-packages"
    require(all(sha(package_root / relative) == expected
                for relative, expected in RUNTIME_PINS.items()),
            "official runtime source bytes differ")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def validate_spec(spec, spec_path, spec_sha):
    require(sha(spec_path) == spec_sha, "approval SHA differs")
    require(spec.get("status") == STATUS and spec.get("execution_authorized") is True,
            "separate execution approval required")
    require(spec.get("steps") == STEPS and spec.get("start_time") == SOURCE_TIME,
            "fixed ten-cycle protocol differs")
    require(spec.get("inference_device") == "cpu"
            and spec.get("state_abs_limit") == STATE_ABS_LIMIT,
            "fixed CPU inference/state bound required")
    require(spec.get("deadline_seconds") == 900, "fixed 900 second bounded attempt required")
    require(spec.get("original_long_ar_gate_passed") is False
            and spec.get("scientific_admission") is False,
            "original failure and exploratory scope must remain explicit")
    for section in ("source_files", "inputs"):
        require(isinstance(spec.get(section), dict) and spec[section], f"{section} required")
    sources = spec.get("causal_history_sources")
    require(isinstance(sources, dict) and set(sources) == {"forceFront", "forceRear"},
            "exact causal history sources required")
    for rows in sources.values():
        require(isinstance(rows, list) and rows
                and all(isinstance(row, dict) and set(row) == {"path", "sha256"}
                        and isinstance(row["path"], str)
                        and isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                        for row in rows), "causal history source records differ")


def build_pair(source, output):
    cases = {}
    for role in ("mpc", "zero"):
        case = output / f"case_{role}"
        case.mkdir()
        for part in ("constant", "system", "148"):
            shutil.copytree(source / part, case / part)
        cases[role] = case
    return cases


class PairSolvers:
    def __init__(self, cases, output, image_id):
        self.cases, self.output, self.image_id = cases, output, image_id
        self.names = {role: f"fluid-control-exploratory-h2-{os.getpid()}-{role}" for role in cases}
        self.cids = {}

    def __enter__(self):
        try:
            for role, case in self.cases.items():
                command = ["docker", "create", "--name", self.names[role],
                       "--label", "fluid-control.engineering=exploratory-short-h2-real-cfd",
                       "--runtime", "runc", "--network", "none", "--read-only",
                       "--memory", "8g", "--memory-swap", "8g", "--cpus", "2",
                       "--pids-limit", "128", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
                       "--user", f"{os.getuid()}:{os.getgid()}",
                       "--mount", f"type=bind,src={case},dst=/case",
                       "--workdir", "/case", IMAGE, "sh", "-c", "while :; do sleep 3600; done"]
                cid = subprocess.check_output(command, text=True, timeout=30).strip()
                self.cids[role] = cid
                row = json.loads(subprocess.check_output(
                    ["docker", "inspect", cid], text=True, timeout=10))[0]
                require(row["Image"] == self.image_id and row["HostConfig"]["Runtime"] == "runc",
                        "OpenFOAM container identity")
                mounts = [(item["Source"], item["Destination"], item["RW"])
                          for item in row["Mounts"]]
                require(mounts == [(str(case), "/case", True)], "narrow case mount required")
                subprocess.run(["docker", "start", cid], check=True, timeout=15,
                               stdout=subprocess.DEVNULL)
        except Exception:
            # A timed-out client can leave a daemon-created container. Recover
            # only our exact names; close() revalidates ownership before removal.
            try:
                for _ in range(6):
                    known = set(subprocess.check_output(
                        ["docker", "ps", "-a", "--format", "{{.Names}}"],
                        text=True, timeout=10).splitlines())
                    for role, name in self.names.items():
                        if name in known and role not in self.cids:
                            self.cids[role] = json.loads(subprocess.check_output(
                                ["docker", "inspect", name], text=True,
                                timeout=10))[0]["Id"]
                    time.sleep(1)
            finally:
                self.close()
            raise
        return self

    def command(self, role, *arguments):
        return ["docker", "exec", self.cids[role], "/openfoam/run", *arguments]

    def pair(self, step, end, check, guard):
        jobs = {}
        for role in self.cases:
            log = self.output / f"solver_{role}_{step:02d}.log"
            handle = log.open("x")
            jobs[role] = (subprocess.Popen(self.command(role, "pimpleFoam", "-case", "/case"),
                                           stdout=handle, stderr=subprocess.STDOUT), handle, log)
        try:
            while any(process.poll() is None for process, _, _ in jobs.values()):
                guard()
                time.sleep(.25)
            require(all(process.returncode == 0 for process, _, _ in jobs.values()),
                    "paired solver failed")
        finally:
            for process, handle, _ in jobs.values():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                handle.close()
        return {role: check(self.cases[role], log, end)
                for role, (_, _, log) in jobs.items()}

    def close(self):
        errors = []
        for role, cid in reversed(tuple(self.cids.items())):
            try:
                row = json.loads(subprocess.check_output(
                    ["docker", "inspect", cid], text=True, timeout=10))[0]
                mounts = [(item["Source"], item["Destination"], item["RW"])
                          for item in row["Mounts"]]
                require(row["Name"] == "/" + self.names[role]
                        and row["Image"] == self.image_id
                        and row["Config"].get("Labels", {}).get("fluid-control.engineering")
                        == "exploratory-short-h2-real-cfd"
                        and mounts == [(str(self.cases[role]), "/case", True)],
                        "refusing cleanup of unowned container")
            except Exception as error:
                errors.append(repr(error))
                continue
            try:
                subprocess.run(["docker", "stop", "--timeout", "5", cid], check=True,
                               timeout=15, stdout=subprocess.DEVNULL)
            except Exception as error:
                errors.append(repr(error))
            try:
                atomic_json(self.output / f"container_terminal_{cid[:12]}.json",
                            json.loads(subprocess.check_output(
                                ["docker", "inspect", cid], text=True, timeout=10))[0],
                            exclusive=True)
            except Exception as error:
                errors.append(repr(error))
            try:
                subprocess.run(["docker", "rm", "-f", cid], check=True, timeout=15,
                               stdout=subprocess.DEVNULL)
            except Exception as error:
                errors.append(repr(error))
        ids = subprocess.check_output(
            ["docker", "ps", "-aq", "--no-trunc"], text=True, timeout=10).split()
        names = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}"], text=True,
            timeout=10).splitlines()
        require(not any(cid in ids for cid in self.cids.values())
                and not any(name in names for name in self.names.values()) and not errors,
                f"owned cleanup failed: {errors}")

    def __exit__(self, *_):
        self.close()


def execute(spec, repo, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden for CPU inference")
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir(),
            "exclusive output required")
    for relative, expected in spec["source_files"].items():
        path = confined(repo, relative)
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                f"source SHA differs: {relative}")
    inputs = {key: confined(repo, row["path"]) for key, row in spec["inputs"].items()}
    for key, path in inputs.items():
        require(path.is_file() and not path.is_symlink()
                and sha(path) == spec["inputs"][key]["sha256"], f"input SHA differs: {key}")
    host_guard(startup=True)
    validate_runtime(repo)
    output.mkdir()
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]
    import torch
    torch.set_float32_matmul_precision("high")
    # Identity contract remains default-TF32/high even though this bounded run
    # executes the official models on CPU because usable CUDA headroom is absent.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from omegaconf import OmegaConf
    from train_tandem_fno import build_model
    import fluid_control.dual_fno as dual
    import p026_state_history
    from fluid_control.online_current_frame import normalize_current
    from fluid_control.exploratory_short_mpc import (
        five_hold_sequences, load_bound_b00_baseline, load_bound_k1,
        rollout_five_h2, validate_recorded_k1_failure,
    )
    from fluid_control.exploratory_causal_history_mpc import (
        append_actual_endpoint, load_bound_actual_history,
        select_canonical_history_action,
    )
    from fluid_control.canonical_joint_v1 import (
        canonical_force_ledger, canonical_joint_cost_components,
    )
    from fluid_control.openfoam_force_history import actual_causal_prehistory
    from fluid_control.exploratory_paired_causal_history_feedback import run_paired_ten_cycle
    transport = import_file("exploratory_reviewed_transport", inputs["transport"])
    sampler = inputs["sampler"]
    validate_recorded_k1_failure(inputs["k1_formal_receipt"])
    baseline_drag, baseline_rms = load_bound_b00_baseline(inputs["baseline"])
    canonical_baseline = {
        "total_drag": baseline_drag,
        "rear_cl_fluctuation_rms": baseline_rms,
        "source": f"sha256:{spec['inputs']['baseline']['sha256']}",
    }
    cfg = OmegaConf.load(inputs["config"])
    device = torch.device("cpu")
    flow, aero, identity = load_bound_k1(inputs["k1_manifest"], cfg, device,
                                         load_dual_fno=dual.load_dual_fno,
                                         build_model=build_model)
    flow.eval().requires_grad_(False)
    aero.eval().requires_grad_(False)
    norm = read_json(inputs["normalization"])
    force_mean = torch.tensor(norm["all_force_mean"], dtype=torch.float32, device=device)
    force_std = torch.tensor(norm["all_force_std"], dtype=torch.float32, device=device)
    with np.load(inputs["reference_sample"], allow_pickle=False) as packet:
        expected_mask = packet["mask"].copy()
    source = confined(repo, spec["source_restart"])
    source_baseline = {part: tree(source / part) for part in ("constant", "system", "148")}
    require(source_baseline == spec["source_restart_tree_sha256"],
            "source restart tree differs")
    cases = build_pair(source, output)
    require(all({part: tree(case / part) for part in ("constant", "system", "148")}
                == spec["source_restart_tree_sha256"] for case in cases.values()),
            "paired source copies differ")
    for case in cases.values():
        transport.substitute(case / "system/controlDict", "writeInterval", .1)
    image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE,
                                        "--format", "{{.Id}}"], text=True,
                                       timeout=10).strip()
    require(image_id == spec["openfoam_image_id"], "OpenFOAM image ID differs")
    started = time.monotonic()
    history = load_bound_actual_history(
        source, SOURCE_TIME, provenance_root=repo,
        expected_sources=spec["causal_history_sources"],
        actual_causal_prehistory=actual_causal_prehistory,
    )
    restart_observation, restart_force_sources = transport.total_drag_observation_at(
        source, SOURCE_TIME, 0.0)
    require(np.array_equal(history["forces"][-1].astype(np.float32),
                           np.asarray(restart_observation[64:68], dtype=np.float32)),
            "causal force history does not match current restart force")
    def guard():
        host = host_guard()
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps({"elapsed": time.monotonic()-started,
                                     "host": host, "inference_device": "cpu"}) + "\n")
        require(time.monotonic()-started < spec["deadline_seconds"], "deadline exceeded")
    def terminated(*_):
        raise RuntimeError("termination requested")
    old_signals = {item: signal.signal(item, terminated)
                   for item in (signal.SIGTERM, signal.SIGINT)}
    with PairSolvers(cases, output, image_id) as solvers:
        def run_logged(command, log):
            with log.open("x") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=120, check=False)
            require(result.returncode == 0, f"command failed: {command[0]}")
        def observe(role, case, current_time, omega):
            name = f"VTK_current_{role}_{current_time:.1f}".replace(".", "_")
            require(not (case / name).exists(), "stale current export")
            run_logged(solvers.command(role, "foamToVTK", "-case", "/case", "-time",
                                       f"{current_time:g}", "-fields", "(U p)",
                                       "-no-boundary", "-name", name),
                       output / f"export_{role}_{current_time:.1f}.log")
            root = case / name
            sample = output / f"current_{role}_{current_time:.1f}.npz"
            run_logged([str(repo / ".venv-curator-py312/bin/python"), str(sampler),
                        str(root), str(sample)], sample.with_suffix(".log"))
            with np.load(sample, allow_pickle=False) as source_packet:
                packet = {key: source_packet[key].copy() for key in source_packet.files}
            current = normalize_current(packet, expected_time=current_time,
                                        expected_mask=expected_mask,
                                        normalization_bytes=inputs["normalization"].read_bytes(),
                                        expected_normalization_sha256=spec["inputs"]["normalization"]["sha256"])
            return {**current, "role": role, "sampled_before_action": True,
                    "sample_sha256": sha(sample), "applied_omega": omega}
        def plan(packet, previous):
            begin = time.perf_counter()
            predicted, bounds = rollout_five_h2(
                flow, aero, packet["state"].to(device), packet["mask"].to(device),
                previous, force_mean, force_std, build_input=p026_state_history.build_input,
                state_abs_limit=float(spec["state_abs_limit"]))
            decision = select_canonical_history_action(
                predicted, bounds, five_hold_sequences(previous), history,
                current_omega=previous, state_abs_limit=float(spec["state_abs_limit"]),
                baseline=canonical_baseline,
                canonical_force_ledger=canonical_force_ledger,
                canonical_joint_cost_components=canonical_joint_cost_components)
            decision["cpu_inference_wall_seconds"] = time.perf_counter() - begin
            decision["wall_latency_is_observational_not_realtime_claim"] = True
            return decision
        def solve_pair(step, end):
            return solvers.pair(step, end, transport.check_segment, guard)
        def forces(role, case, endpoint, omega):
            nonlocal history
            observation, _ = transport.total_drag_observation_at(case, endpoint, omega)
            actual = observation[64:68]
            if role == "mpc":
                history = append_actual_endpoint(history, endpoint=endpoint,
                                                 actual_force=actual)
            return actual
        def summarize(role, case, begin, end):
            front = transport.read_force_window(case, "forceFront", begin, end)
            rear = transport.read_force_window(case, "forceRear", begin, end)
            expected = round((end-begin)/transport.SOLVER_DT)
            require(len(front) == len(rear) == expected, "actual force sample count differs")
            expected_grid = begin + transport.SOLVER_DT * np.arange(1, expected + 1)
            require(np.allclose(front[:, 0], expected_grid, rtol=0.0, atol=1e-8)
                    and np.allclose(rear[:, 0], expected_grid, rtol=0.0, atol=1e-8),
                    "actual force time grid differs")
            row = transport.force_metrics(front, rear)
            row["actual_cfd_force_samples"] = expected
            return row
        result = run_paired_ten_cycle(
            cases=cases, output=output, latest_time=transport.latest_time,
            observe_current=observe, plan_action=plan,
            configure_interval=transport.configure_interval, solve_pair=solve_pair,
            observe_forces=forces, summarize_actual=summarize, guard=guard,
            identity={"k1_manifest_sha256": identity.manifest_sha256,
                      "flow_model_sha256": identity.flow.model_sha256,
                      "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
                      "k1_original_formal_non_admission": True})
        require({part: tree(source / part) for part in source_baseline} == source_baseline,
                "original restart/config source changed")
        result["source_restart_unchanged"] = True
        result["selector_mode"] = "canonical_causal_history_h2_v1"
        result["causal_history_sources"] = spec["causal_history_sources"]
        result["restart_force_observation_sources"] = restart_force_sources
        result["persistent_history_end_time"] = float(history["times"][-1])
        result["persistent_history_actual_endpoint_only"] = True
        atomic_json(output / "result.json", result, exclusive=True)
    for item, previous_handler in old_signals.items():
        signal.signal(item, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = read_json(args.spec)
    validate_spec(spec, args.spec, args.spec_sha256)
    if not args.execute:
        print("PREPARATION_ONLY_NOT_EXECUTED")
        return
    execute(spec, Path(spec["repo"]).resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
