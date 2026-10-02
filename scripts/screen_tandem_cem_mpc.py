#!/usr/bin/env python3
"""Fail-closed CEM-MPC screen using a Gate-B-passed PhysicsNeMo FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from fluid_control.cem_mpc import CEMConfig, CEMMPCOptimizer
from fluid_control.stage_c_objective import stage_c_sequence_costs
from train_tandem_fno import build_model


def load_composed_config(path: Path):
    """Resolve the official model configuration tree."""
    with initialize_config_dir(
        config_dir=str(path.parent.resolve()), version_base="1.3"
    ):
        return compose(config_name=path.stem)


def require_gate_b_pass(path: Path) -> dict:
    """Refuse planning unless the exact checkpoint evaluation passed Gate B."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "GATE_B_PASS":
        raise RuntimeError(f"CEM-MPC requires GATE_B_PASS, got {report.get('status')}")
    return report


def rollout_population(
    network: torch.nn.Module,
    initial_state: torch.Tensor,
    mask: torch.Tensor,
    actions: np.ndarray,
    *,
    current_omega: float,
    action_scale: float,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a population and return physical four-force histories and bounds."""
    population = np.asarray(actions, dtype=np.float64)
    if population.ndim != 2 or population.shape[1] < 1:
        raise ValueError("actions must have shape (population, horizon)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(population).all():
        raise ValueError("actions must be finite")
    device = initial_state.device
    height, width = mask.shape[-2:]
    force_chunks = []
    bound_chunks = []
    with torch.inference_mode():
        for start in range(0, len(population), batch_size):
            control = torch.as_tensor(
                population[start : start + batch_size],
                dtype=initial_state.dtype,
                device=device,
            )
            count = control.shape[0]
            state = initial_state.expand(count, -1, -1, -1).clone()
            active_mask = mask.expand(count, -1, -1, -1)
            previous = torch.full(
                (count,), float(current_omega), dtype=state.dtype, device=device
            )
            forces = []
            max_bound = state.abs().flatten(1).amax(dim=1)
            for step in range(control.shape[1]):
                following = control[:, step]
                now_grid = (
                    (previous / action_scale)
                    .reshape(-1, 1, 1, 1)
                    .expand(-1, 1, height, width)
                )
                next_grid = (
                    (following / action_scale)
                    .reshape(-1, 1, 1, 1)
                    .expand(-1, 1, height, width)
                )
                raw = network(
                    torch.cat((state, active_mask, now_grid, next_grid), dim=1)
                )
                if raw.ndim != 4 or raw.shape[1] != 7:
                    raise ValueError("Stage-C CEM requires a seven-output FNO")
                state = (state + raw[:, :3]) * active_mask
                normalized_force = (raw[:, 3:7] * active_mask).sum(dim=(-2, -1)) / (
                    active_mask.sum(dim=(-2, -1)).clamp_min(1)
                )
                forces.append(normalized_force * force_std + force_mean)
                max_bound = torch.maximum(max_bound, state.abs().flatten(1).amax(dim=1))
                previous = following
            force_chunks.append(torch.stack(forces, dim=1).cpu().numpy())
            bound_chunks.append(max_bound.cpu().numpy())
    return np.concatenate(force_chunks), np.concatenate(bound_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/tandem_cem_stage_c.yaml")
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    output = Path(cfg.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    gate = require_gate_b_pass(Path(cfg.gate_b_audit))

    baselines = json.loads(Path(cfg.baseline_manifest).read_text(encoding="utf-8"))
    baseline = baselines["cases"].get(str(cfg.case_key))
    if baseline is None:
        raise KeyError(f"missing phase baseline: {cfg.case_key}")
    data_root = Path(cfg.data_root)
    case_path = data_root / str(cfg.split) / f"{cfg.case}.h5"
    if not case_path.is_file():
        raise FileNotFoundError(case_path)

    DistributedManager.initialize()
    dist = DistributedManager()
    model_cfg = load_composed_config(Path(cfg.model_config))
    if dist.cuda:
        torch.cuda.set_per_process_memory_fraction(
            float(model_cfg.training.gpu_memory_fraction), device=dist.device
        )
    network = build_model(model_cfg).to(dist.device)
    metadata: dict = {}
    epoch = load_checkpoint(
        Path(cfg.checkpoint_dir),
        models=network,
        metadata_dict=metadata,
        device=dist.device,
    )
    if epoch == 0 or int(gate["checkpoint_epoch"]) != epoch:
        raise RuntimeError("Gate-B audit and planning checkpoint do not match")
    network.eval().requires_grad_(False)

    stats = json.loads((data_root / "normalization.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    action_scale = float(manifest["max_abs_omega"])
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.tensor(stats["all_force_mean"], device=dist.device)
    force_std = torch.tensor(stats["all_force_std"], device=dist.device)
    state_guard = 1.25 * float(stats["state_abs_normalized_max_train"])
    with h5py.File(case_path, "r") as handle:
        frame = int(cfg.frame)
        physical = torch.from_numpy(handle["state"][frame]).float().to(dist.device)
        mask = (
            torch.from_numpy(handle["mask"][frame : frame + 1]).float().to(dist.device)
        )
        current_omega = float(handle["omega"][frame, 0])
    initial_state = (((physical - state_mean) / state_std) * mask[0]).unsqueeze(0)

    control = cfg.control
    optimizer = CEMMPCOptimizer(
        CEMConfig(
            horizon=int(control.horizon),
            population_size=int(control.population_size),
            elite_fraction=float(control.elite_fraction),
            iterations=int(control.iterations),
            omega_abs_max=float(control.omega_abs_max),
            max_delta_omega=float(control.max_delta_omega),
            initial_std=float(control.initial_std),
            min_std=float(control.min_std),
            update_rate=float(control.update_rate),
            seed=int(control.seed),
        )
    )

    def objective(population: np.ndarray) -> np.ndarray:
        forces, bounds = rollout_population(
            network,
            initial_state,
            mask,
            population,
            current_omega=current_omega,
            action_scale=action_scale,
            force_mean=force_mean,
            force_std=force_std,
            batch_size=int(control.rollout_batch_size),
        )
        costs, _ = stage_c_sequence_costs(
            forces,
            population,
            current_omega=current_omega,
            baseline=baseline,
            action_scale=action_scale,
            max_delta_omega=float(control.max_delta_omega),
        )
        costs[bounds > state_guard] = np.inf
        return costs

    result = optimizer.optimize(objective, current_omega=current_omega)
    best_forces, best_bounds = rollout_population(
        network,
        initial_state,
        mask,
        result.actions[None],
        current_omega=current_omega,
        action_scale=action_scale,
        force_mean=force_mean,
        force_std=force_std,
        batch_size=1,
    )
    costs, components = stage_c_sequence_costs(
        best_forces,
        result.actions[None],
        current_omega=current_omega,
        baseline=baseline,
        action_scale=action_scale,
        max_delta_omega=float(control.max_delta_omega),
    )
    record = {
        "status": "surrogate_only_cem_screen",
        "scientific_scope": "not_openfoam_control_evidence",
        "gate_b_audit": str(cfg.gate_b_audit),
        "checkpoint_epoch": epoch,
        "case_key": str(cfg.case_key),
        "frame": int(cfg.frame),
        "baseline": baseline,
        "actions": result.actions.tolist(),
        "predicted_forces": best_forces[0].tolist(),
        "cost": float(costs[0]),
        "cost_components": {key: float(value[0]) for key, value in components.items()},
        "best_cost_history": list(result.best_cost_history),
        "evaluations": result.evaluations,
        "max_abs_normalized_state": float(best_bounds[0]),
        "state_guard": state_guard,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
