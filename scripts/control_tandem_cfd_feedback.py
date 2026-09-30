#!/usr/bin/env python3
"""Run segmented OpenFOAM feedback control with a frozen PhysicsNeMo MPC."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from control_tandem_mpc import load_composed_config, optimize_actions
from train_tandem_fno import build_model


PROJECT = Path(__file__).resolve().parents[1]
CFD_ROOT = PROJECT / "cfd" / "tandem_cylinders"
sys.path.insert(0, str(CFD_ROOT))
from make_expanded_control_dataset import replace_rear_patch  # noqa: E402


def numeric_time_dirs(case: Path) -> list[tuple[float, Path]]:
    result = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                result.append((float(path.name), path))
            except ValueError:
                pass
    return sorted(result)


def update_control_dict(path: Path, start: float, end: float) -> None:
    text = path.read_text(encoding="utf-8")
    text, first = re.subn(r"(?m)^startTime\s+[^;]+;", f"startTime {start:.10g};", text)
    text, second = re.subn(r"(?m)^endTime\s+[^;]+;", f"endTime {end:.10g};", text)
    if first != 1 or second != 1:
        raise ValueError(f"could not update start/end time in {path}")
    path.write_text(text, encoding="utf-8")


def run_logged(command: list[str], log: Path, append: bool = False) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def latest_rear_force(case: Path, target_time: float) -> tuple[float, float]:
    rows = []
    for path in case.glob("postProcessing/forceRear/*/coefficient.dat"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 5:
                    rows.append((float(fields[0]), float(fields[1]), float(fields[4])))
    if not rows:
        raise FileNotFoundError("no rear force coefficient samples")
    closest = min(rows, key=lambda row: abs(row[0] - target_time))
    if abs(closest[0] - target_time) > 1.0e-6:
        raise ValueError(f"no force sample at {target_time}; closest is {closest[0]}")
    return closest[1], closest[2]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_name")
    parser.add_argument("--config", type=Path, default=Path("conf/tandem_mpc.yaml"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tandem_mpc_feedback"))
    args = parser.parse_args()
    if not args.case_name.startswith("mpc_feedback_"):
        parser.error("case_name must begin with mpc_feedback_")

    # The controller config is a plain YAML; the model config uses Hydra defaults.
    from omegaconf import OmegaConf
    controller_cfg = OmegaConf.load(args.config)
    model_cfg = load_composed_config(Path(controller_cfg.model_config))

    DistributedManager.initialize()
    dist = DistributedManager()
    network = build_model(model_cfg).to(dist.device)
    metadata = {}
    epoch = load_checkpoint(
        controller_cfg.checkpoint_dir, models=network, metadata_dict=metadata, device=dist.device
    )
    if epoch == 0:
        raise FileNotFoundError(f"no checkpoint in {controller_cfg.checkpoint_dir}")
    network.eval().requires_grad_(False)

    data_root = Path(controller_cfg.data_root)
    stats = json.loads((data_root / "normalization.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    action_scale = float(manifest["max_abs_omega"])
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.tensor(stats["force_mean"], device=dist.device)
    force_std = torch.tensor(stats["force_std"], device=dist.device)
    candidates = list(data_root.glob(f"*/{controller_cfg.case}.h5"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one HDF5 for {controller_cfg.case}")
    with h5py.File(candidates[0], "r") as handle:
        index = int(controller_cfg.initial_step)
        physical = torch.from_numpy(handle["state"][index]).float().to(dist.device)
        mask = torch.from_numpy(handle["mask"][index:index + 1]).float().to(dist.device)
        current_omega = torch.tensor(float(handle["omega"][index, 0]), device=dist.device)
        current_time = float(handle["time"][index, 0])
    initial_time = current_time
    initial_omega = float(current_omega.item())
    state = ((physical - state_mean) / state_std * mask[0]).unsqueeze(0)

    case = CFD_ROOT / "cases" / args.case_name
    if not case.is_dir():
        raise FileNotFoundError(case)
    latest_time, _ = numeric_time_dirs(case)[-1]
    if abs(latest_time - current_time) > 1.0e-6:
        raise ValueError(f"case latest time {latest_time} != model initial time {current_time}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    solver_log = case / "log.pimpleFoam.feedback"
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        tick = time.perf_counter()
        optimization_started = time.perf_counter()
        actions, components = optimize_actions(
            network, state, mask, current_omega, action_scale, force_mean, force_std,
            controller_cfg.control, controller_cfg.objective,
        )
        mpc_seconds = time.perf_counter() - optimization_started
        next_omega = actions[0]
        next_time = current_time + 0.1
        _, latest_dir = numeric_time_dirs(case)[-1]
        velocity_path = latest_dir / "U"
        velocity_path.write_text(
            replace_rear_patch(
                velocity_path.read_text(encoding="utf-8"),
                [
                    (current_time, float(current_omega.item())),
                    (next_time, float(next_omega.item())),
                ],
            ),
            encoding="utf-8",
        )
        update_control_dict(case / "system" / "controlDict", current_time, next_time)
        run_logged(
            ["bash", str(CFD_ROOT / "run_openfoam.sh"), "pimpleFoam", "-case", f"/case/cases/{args.case_name}"],
            solver_log, append=True,
        )
        actual_time, _ = numeric_time_dirs(case)[-1]
        if abs(actual_time - next_time) > 2.0e-6:
            raise ValueError(f"solver stopped at {actual_time}, expected {next_time}")

        vtk_name = f"VTK_feedback_{step:03d}"
        vtk_root = case / vtk_name
        vtk_log = output / f"foamToVTK_{step:03d}.log"
        run_logged(
            [
                "bash", str(CFD_ROOT / "run_openfoam.sh"), "foamToVTK",
                "-case", f"/case/cases/{args.case_name}", "-latestTime",
                "-fields", "(U p)", "-no-boundary", "-name", vtk_name,
            ],
            vtk_log,
        )
        sample_path = output / f"frame_{step:03d}.npz"
        sample_log = output / f"sample_{step:03d}.log"
        run_logged(
            [
                str(PROJECT / ".venv-curator" / "bin" / "python"),
                str(PROJECT / "scripts" / "sample_tandem_vtk_frame.py"),
                str(vtk_root), str(sample_path),
            ],
            sample_log,
        )
        with np.load(sample_path) as sampled:
            physical = torch.from_numpy(sampled["state"]).float().to(dist.device)
            next_mask = torch.from_numpy(sampled["mask"][None]).float().to(dist.device)
            sampled_time = float(sampled["time"][0])
        if abs(sampled_time - next_time) > 1.0e-5:
            raise ValueError(f"sampled time {sampled_time} != {next_time}")
        state = ((physical - state_mean) / state_std * next_mask[0]).unsqueeze(0)
        mask = next_mask
        rear_cd, rear_cl = latest_rear_force(case, next_time)
        row = {
            "step": step,
            "time": next_time,
            "omega": float(next_omega),
            "delta_omega": float(next_omega - current_omega),
            "rear_cd": rear_cd,
            "rear_cl": rear_cl,
            "objective": components[0],
            "mpc_seconds": mpc_seconds,
            "step_seconds": time.perf_counter() - tick,
        }
        rows.append(row)
        write_rows(output / "timeseries.csv", rows)
        (output / "progress.json").write_text(
            json.dumps({"completed_steps": step, "latest": row}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "cfd_feedback_step", **row}), flush=True)
        current_time = next_time
        current_omega = next_omega
        shutil.rmtree(vtk_root)

    result = {
        "status": "openfoam_state_feedback",
        "checkpoint_epoch": epoch,
        "action_scale": action_scale,
        "case": args.case_name,
        "steps": args.steps,
        "start_time": rows[0]["time"] - 0.1,
        "end_time": rows[-1]["time"],
        "omega_rms": float(np.sqrt(np.mean(np.square([row["omega"] for row in rows])))),
        "omega_abs_max": float(np.max(np.abs([row["omega"] for row in rows]))),
        "delta_omega_abs_max": float(np.max(np.abs([row["delta_omega"] for row in rows]))),
        "elapsed_seconds": time.perf_counter() - started,
        "mean_mpc_seconds": float(np.mean([row["mpc_seconds"] for row in rows])),
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in rows])),
    }
    case_config_path = case / "case_config.json"
    case_config = json.loads(case_config_path.read_text(encoding="utf-8"))
    case_config.update({
        "control_mode": "state_feedback",
        "feedback_checkpoint_epoch": epoch,
        "feedback_action_scale": action_scale,
        "feedback_action_points": [[initial_time, initial_omega]]
        + [[row["time"], row["omega"]] for row in rows],
    })
    case_config_path.write_text(json.dumps(case_config, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
