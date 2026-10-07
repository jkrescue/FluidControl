"""Replay saved H5 decisions on GPU; engineering comparison only.

No CFD, optimizer, checkpoint write, action execution, or scientific admission
is permitted.  The ten saved current fields and actual endpoint forces are the
only trajectory evidence consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path("/workspace/fluid_control")
RESULT = REPO / "artifacts/p064_b_causal_history_h5_real_cfd_20261007/result.json"
OUTPUT = REPO / "artifacts/p064_b_causal_history_h5_gpu_replay_r2_20261007"
STATUS = "P064_B_CAUSAL_HISTORY_H5_GPU_REPLAY_APPROVED"
COMPLETE = "P064_B_CAUSAL_HISTORY_H5_GPU_REPLAY_COMPLETE_NOT_ADMISSION"
RUNTIME = {
    "physicsnemo/models/fno/fno.py": "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9",
    "physicsnemo/utils/checkpoint.py": "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e",
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not path.exists() and not temporary.exists(), "exclusive JSON output required")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def memory() -> dict[str, int]:
    rows = {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "MemFree:", "Cached:"))}
    require("MemAvailable" in rows, "missing available-memory observation")
    return rows


def cgroup_limits() -> dict:
    line = next(row for row in Path("/proc/self/cgroup").read_text().splitlines()
                if row.startswith("0::"))
    group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    require((group / "memory.swap.max").read_text().strip() == "0",
            "MemorySwapMax=0 required")
    value = (group / "memory.max").read_text().strip()
    require(value != "max" and 0 < int(value) <= 12 * 2**30,
            "MemoryMax<=12G required")
    return {"path": str(group), "memory_max": int(value), "memory_swap_max": 0}


def inference_precision(torch) -> dict:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def override_inference_precision(torch) -> dict:
    """Apply the reviewed override only after historical-precision identity load."""
    before = inference_precision(torch)
    require(before == {"float32_matmul_precision": "high",
                       "cuda_matmul_allow_tf32": True,
                       "cudnn_allow_tf32": True},
            "historical load precision differs before override")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    effective = inference_precision(torch)
    require(effective == {"float32_matmul_precision": "highest",
                          "cuda_matmul_allow_tf32": False,
                          "cudnn_allow_tf32": False},
            "post-load inference override not effective")
    return {"scope": "POST_VERIFIED_LOAD_INFERENCE_OVERRIDE_NOT_ORIGINAL_PROTOCOL",
            "before_override": before, "effective": effective}


def compare_decision(cpu: dict, gpu: dict) -> dict:
    """Return raw differences and exact selection/rank evidence."""
    import numpy as np

    cpu_force = np.asarray(cpu["predicted_forces_h5"], dtype=np.float64)
    gpu_force = np.asarray(gpu["predicted_forces_h5"], dtype=np.float64)
    require(cpu_force.shape == gpu_force.shape == (5, 5, 4),
            "fixed ten-by-five-by-five force comparison required")
    difference = gpu_force - cpu_force
    cpu_cost = np.asarray(cpu["h5_cost"], dtype=np.float64)
    gpu_cost = np.asarray(gpu["h5_cost"], dtype=np.float64)
    require(cpu_cost.shape == gpu_cost.shape == (5,), "five costs required")
    component_differences = []
    for cpu_candidate, gpu_candidate in zip(
            cpu["candidate_reports"], gpu["candidate_reports"], strict=True):
        require(len(cpu_candidate["stages"]) == len(gpu_candidate["stages"]) == 5,
                "five canonical stages required")
        for cpu_stage, gpu_stage in zip(
                cpu_candidate["stages"], gpu_candidate["stages"], strict=True):
            require(tuple(cpu_stage["components"]) == tuple(gpu_stage["components"]),
                    "component names differ")
            component_differences.extend(
                float(gpu_stage["components"][key]) - float(cpu_stage["components"][key])
                for key in cpu_stage["components"])
    component_differences = np.asarray(component_differences, dtype=np.float64)
    return {
        "force_difference": {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(np.square(difference)))),
            "values": difference.tolist(),
        },
        "cost_difference": (gpu_cost - cpu_cost).tolist(),
        "component_difference": {
            "max_abs": float(np.max(np.abs(component_differences))),
            "values": component_differences.tolist(),
        },
        "cpu_rank": np.argsort(cpu_cost, kind="stable").tolist(),
        "gpu_rank": np.argsort(gpu_cost, kind="stable").tolist(),
        "ranking_same": bool(np.array_equal(np.argsort(cpu_cost, kind="stable"),
                                             np.argsort(gpu_cost, kind="stable"))),
        "cpu_selected_index": int(cpu["selected_index"]),
        "gpu_selected_index": int(gpu["selected_index"]),
        "cpu_selected_action": float(cpu["selected_action"]),
        "gpu_selected_action": float(gpu["selected_action"]),
        "selection_same": (int(cpu["selected_index"]) == int(gpu["selected_index"])
                           and float(cpu["selected_action"]) == float(gpu["selected_action"])),
    }


def validate_result(payload: dict) -> list[dict]:
    identity = payload.get("identity", {})
    require(identity.get("b_manifest_sha256") == "92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891",
            "exact B CPU result identity required")
    require(payload.get("status") == "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_COMPLETE_NOT_ADMISSION"
            and payload.get("selector_mode") == "canonical_causal_history_h5_v1"
            and payload.get("cycles") == 10 and len(payload.get("rows", [])) == 10,
            "exact completed H5 evidence required")
    require(all(payload.get(key) is False for key in
                ("scientific_admission", "ppo_executed", "hydrogym_solver_used",
                 "original_long_ar_gate_passed")), "H5 scope differs")
    return payload["rows"]


def actual_force(row: dict) -> list[float]:
    force = row["actual_endpoint_forces"]["mpc"]
    return [float(force[key]) for key in ("front_cd", "front_cl", "rear_cd", "rear_cl")]


def worker(approval: dict) -> None:
    result_sha = approval["h5_result_sha256"]
    require(isinstance(result_sha, str) and len(result_sha) == 64
            and sha(RESULT) == result_sha,
            "H5 result identity differs")
    result = json.loads(RESULT.read_text())
    rows = validate_result(result)
    source_files = approval["source_files"]
    for relative, expected in source_files.items():
        require(sha(REPO / relative) == expected, f"source changed: {relative}")
    inputs = {key: REPO / row["path"] for key, row in approval["inputs"].items()}
    for key, path in inputs.items():
        require(sha(path) == approval["inputs"][key]["sha256"], f"input changed: {key}")
    require(Path(sys.prefix).resolve() == (REPO / ".venv-curator-py312").resolve(),
            "wrong runtime environment")
    for package, version in {"torch": "2.14.1", "numpy": "2.5.3",
                             "nvidia-physicsnemo": "2.2.2"}.items():
        require(importlib.metadata.version(package) == version, f"wrong package: {package}")
    for relative, expected in RUNTIME.items():
        require(sha(Path(sys.prefix) / "lib/python3.12/site-packages" / relative) == expected,
                f"official runtime source changed: {relative}")

    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    import numpy as np
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "one CUDA device required")
    torch.cuda.set_per_process_memory_fraction(.06, 0)
    # The official identity loader validates this historical precision.  The
    # engineering replay override is deliberately applied only after loading.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from omegaconf import OmegaConf
    from train_tandem_fno import build_model
    loader_spec = importlib.util.spec_from_file_location("b_cpu_driver", inputs["b_cpu_driver"])
    b_driver = importlib.util.module_from_spec(loader_spec)
    sys.modules[loader_spec.name] = b_driver
    loader_spec.loader.exec_module(b_driver)
    dual = b_driver.import_file("b_replay_dual", inputs["b_consumer"])
    from fluid_control.online_current_frame import normalize_current
    from fluid_control.exploratory_short_mpc import (
        load_bound_b00_baseline, validate_recorded_k1_failure,
    )
    from fluid_control.exploratory_causal_history_mpc import (
        append_actual_endpoint, load_bound_actual_history,
    )
    from fluid_control.exploratory_causal_history_horizon import (
        five_held_sequences, rollout_five_held_horizon,
        select_canonical_history_horizon,
    )
    from fluid_control.canonical_joint_v1 import (
        canonical_force_ledger, canonical_joint_cost_components,
    )
    from fluid_control.openfoam_force_history import actual_causal_prehistory
    from p026_state_history import build_input

    device = torch.device("cuda:0")
    validate_recorded_k1_failure(inputs["k1_formal_receipt"])
    flow, aero, identity = b_driver.load_bound_b(
        inputs["b_manifest"], OmegaConf.load(inputs["config"]), device,
        load_dual_fno=dual.load_dual_fno, build_model=build_model)
    require(result["identity"]["flow_model_sha256"] == identity.flow.model_sha256
            and result["identity"]["aerodynamic_model_sha256"] == identity.aerodynamic.model_sha256,
            "CPU/GPU model identities differ")
    precision = override_inference_precision(torch)
    for model in (flow, aero):
        model.eval().requires_grad_(False)
    def model_digest():
        digest = hashlib.sha256()
        for model in (flow, aero):
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_model_digest = model_digest()
    baseline = load_bound_b00_baseline(inputs["baseline"])
    canonical_baseline = {"total_drag": baseline[0],
                          "rear_cl_fluctuation_rms": baseline[1],
                          "source": f"sha256:{approval['inputs']['baseline']['sha256']}"}
    norm_bytes = inputs["normalization"].read_bytes()
    norm = json.loads(norm_bytes)
    mean = torch.tensor(norm["all_force_mean"], device=device, dtype=torch.float32)
    std = torch.tensor(norm["all_force_std"], device=device, dtype=torch.float32)
    with np.load(inputs["reference_sample"], allow_pickle=False) as packet:
        expected_mask = packet["mask"].copy()
    restart = REPO / approval["source_restart"]
    require(restart.is_dir() and restart.resolve().is_relative_to(REPO.resolve()),
            "restart path differs")
    history = load_bound_actual_history(
        restart, 148.0, provenance_root=REPO,
        expected_sources=approval["causal_history_sources"],
        actual_causal_prehistory=actual_causal_prehistory)
    comparisons = []
    started = time.monotonic()
    for index, row in enumerate(rows):
        require(memory()["MemAvailable"] >= 22 * 2**30, "host Available below 22GiB")
        require(inference_precision(torch) == precision["effective"],
                "inference precision changed")
        sample = inputs[f"sample_{index}"]
        require(sha(sample) == row["current_sample_sha256"]["mpc"],
                "saved current-field identity differs")
        with np.load(sample, allow_pickle=False) as packet:
            arrays = {key: packet[key].copy() for key in packet.files}
        current = normalize_current(
            arrays, expected_time=148.0 + .1 * index, expected_mask=expected_mask,
            normalization_bytes=norm_bytes,
            expected_normalization_sha256=approval["inputs"]["normalization"]["sha256"])
        previous = float(row["previous_omega"])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        predicted, bounds = rollout_five_held_horizon(
            flow, aero, current["state"].to(device), current["mask"].to(device),
            previous, mean, std, build_input=build_input,
            state_abs_limit=float(approval["state_abs_limit"]), horizon=5)
        gpu = select_canonical_history_horizon(
            predicted, bounds, five_held_sequences(previous, horizon=5), history,
            current_omega=previous, state_abs_limit=float(approval["state_abs_limit"]),
            baseline=canonical_baseline, canonical_force_ledger=canonical_force_ledger,
            canonical_joint_cost_components=canonical_joint_cost_components, horizon=5)
        torch.cuda.synchronize()
        comparison = compare_decision(row["decision"], gpu)
        comparison.update({"step": index + 1,
                           "wall_seconds_synchronized": time.perf_counter() - begin,
                           "cuda_free_observational": torch.cuda.mem_get_info()[0]})
        comparisons.append(comparison)
        history = append_actual_endpoint(
            history, endpoint=148.1 + .1 * index, actual_force=actual_force(row))
    selection_count = sum(row["selection_same"] for row in comparisons)
    require(initial_model_digest == model_digest(), "model tensors changed")
    require(all(parameter.grad is None for model in (flow, aero)
                for parameter in model.parameters()), "unexpected model gradient")
    atomic_json(OUTPUT / "result.json", {
        "status": COMPLETE,
        "h5_result_sha256": result_sha,
        "precision_override": precision,
        "manifest_sha256": identity.manifest_sha256,
        "model_tensors_unchanged": True,
        "comparisons": comparisons,
        "selection_same_count": selection_count,
        "ranking_same_count": sum(row["ranking_same"] for row in comparisons),
        "selection_consistency_criterion": "10/10 same selected index and exact action",
        "selection_consistent": selection_count == 10,
        "elapsed_seconds": time.monotonic() - started,
        "optimizer_used": False, "cfd_executed": False, "model_saved": False,
        "scientific_admission": False,
    })


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise(approval: dict, approval_path: Path, approval_sha256: str) -> None:
    require(memory()["MemAvailable"] >= 50 * 2**30, "startup Available below 50GiB")
    limits = cgroup_limits()
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "exclusive output required")
    OUTPUT.mkdir()
    process = None

    def interrupted(*_):
        raise RuntimeError("signal requested shutdown")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    started = time.monotonic()
    try:
        with (OUTPUT / "run.log").open("x") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "--execute",
                 "--approval", str(approval_path), "--approval-sha256", approval_sha256],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                row = {"elapsed": time.monotonic() - started, **memory()}
                with (OUTPUT / "memory.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                require(row["MemAvailable"] >= 22 * 2**30, "runtime Available below 22GiB")
                require(row["elapsed"] < 300, "300 second replay deadline")
                code = process.poll()
                if code is not None:
                    require(code == 0, f"worker exit {code}")
                    break
                time.sleep(.5)
        require((OUTPUT / "result.json").is_file(), "missing replay result")
        atomic_json(OUTPUT / "supervisor_result.json", {
            "exit_code": 0, "limits": limits, "scientific_admission": False,
            "source_sha256": sha(Path(__file__)),
        })
    finally:
        if process is not None:
            stop(process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("PREPARATION_ONLY_NOT_EXECUTED")
        return
    require(args.approval is not None and sha(args.approval) == args.approval_sha256,
            "approval hash differs")
    approval = json.loads(args.approval.read_text())
    require(approval.get("status") == STATUS
            and approval.get("execution_authorized") is True
            and approval.get("source_sha256") == sha(Path(__file__))
            and approval.get("output") == str(OUTPUT)
            and isinstance(approval.get("h5_result_sha256"), str)
            and len(approval["h5_result_sha256"]) == 64,
            "separate exact replay approval required")
    if args.worker:
        cgroup_limits()
        worker(approval)
    else:
        supervise(approval, args.approval, args.approval_sha256)


if __name__ == "__main__":
    main()
