#!/usr/bin/env python3
"""Project-owned fixed-train diagnostic; no optimization or scientific admission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE_SHA = {
    "scripts/train_fcp013_independent_force_fno.py": "f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7",
    "scripts/train_fcp011_decoder_scope.py": "9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5",
    "scripts/train_tandem_fno.py": "9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a",
    "src/fluid_control/dual_fno.py": "23380355b812d94025071f62feb9dbf51e593244debff6f2b38b4215c644c254",
    "src/fluid_control/augmented_datapipe.py": "d407476e281be6deff46a87fd3f76b4125e8c584e0496b80ef5849ad1267e148",
    "src/fluid_control/tandem_datapipe.py": "c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae",
    "src/fluid_control/calibrated_checkpoint.py": "23f004aefb5ab4acf40523b188e1353d48a3cc04d6b2d886b6cdf435353c8a88",
}
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
MANIFEST_SHA = "23d2917ef038196f066934f51d0be2770c23b26099eb7ab97439ce58488e13bb"
RESULT_SHA = "239f6567d662157e8f0bec8b1277cc219f7580ac3dde3238657dea8d6b4821d0"
WINDOWS = (
    (160, "base20", "matched_start_acquisition_train_b00_zero", 320),
    (816, "train8", "dynamic_train8_b00_prbs", 90),
    (923, "train8", "dynamic_train8_b02_prbs", 100),
    (975, "train8", "dynamic_train8_b04_prbs", 0),
    (1077, "train8", "dynamic_train8_b06_prbs", 0),
    (1233, "train16", "direct_cfd_directppo2048_v1_env0_ep0009_b00", 0),
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_objective(source_root):
    path = Path(source_root) / "scripts/train_fcp013_independent_force_fno.py"
    if sha256(path) != SOURCE_SHA["scripts/train_fcp013_independent_force_fno.py"]:
        raise ValueError("immutable P013 objective differs")
    spec = importlib.util.spec_from_file_location("fcp014_frozen_p013", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_statistics(residual):
    import numpy as np

    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1 or len(values) not in (62, 100) or not np.isfinite(values).all():
        raise ValueError("expected finite H100 or tail62 residual vector")
    bias = float(values.mean())
    mse = float(np.square(values).mean())
    centered = float(np.square(values - bias).mean())
    return {"samples": len(values), "signed_residual_mean": bias, "mse": mse,
            "rmse": float(np.sqrt(mse)), "bias_mse": bias * bias,
            "centered_residual_mse": centered, "centered_residual_rmse": float(np.sqrt(centered)),
            "decomposition_residual": mse - bias * bias - centered}


def capture_force_panel(objective, model, flow_states, h1_states, batch, force_std, predict):
    """Record outputs from the actual immutable mixed-batch objective, unchanged."""
    import torch

    if batch["state"].shape[0] != 1 or force_std.shape != (4,):
        raise ValueError("fixed batch-one/four-force contract differs")
    if not torch.isfinite(force_std).all() or not (force_std > 0).all():
        raise ValueError("force standard deviations must be positive and finite")
    mask = batch["mask"]
    if not torch.all((mask == 0) | (mask == 1)) or not (mask.sum() > 0):
        raise ValueError("analytic bias derivative requires a nonempty binary mask")
    captures = []

    def recording_predict(network, inputs, masks):
        if network is not model or inputs.shape[0] != 20 or masks.shape[0] != 20:
            raise ValueError("P013 mixed H1/AR batch20 schedule differs")
        delta, forces = predict(network, inputs, masks)
        if forces.shape != (20, 4) or not torch.isfinite(forces).all():
            raise ValueError("invalid recorded four-force output")
        captures.append(forces.detach().clone())
        return delta, forces

    forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)
    if any(isinstance(module, forbidden) for module in model.modules()):
        raise ValueError("force train mode cannot use stochastic or running-statistic modules")
    previous_mode = model.training
    try:
        model.train(True)
        with torch.no_grad():
            metrics = objective.chunk_force_objective(
                model, flow_states, h1_states, mask, batch["omega"], batch["target_force"],
                recording_predict, chunk_size=10, backward=False,
            )
    finally:
        model.train(previous_mode)
    if len(captures) != 10:
        raise ValueError("expected exactly ten mixed-batch force forwards")
    domains = {}
    for name, part in (("h1", slice(0, 10)), ("ar", slice(10, 20))):
        predicted = torch.cat([value[part] for value in captures], dim=0)
        error = predicted - batch["target_force"][0]
        normalized = error[:, 3].cpu().numpy()
        physical = (error[:, 3] * force_std[3]).cpu().numpy()
        domains[name] = {
            "rear_cl_normalized_residual": normalized.tolist(),
            "rear_cl_physical_residual": physical.tolist(),
            "h100": residual_statistics(physical), "tail62": residual_statistics(physical[38:]),
            "analytic_normalized_rear_cl_bias_derivative": 1.25 * float(normalized.astype("float64").mean()),
        }
    return {"objective": metrics, "domains": domains,
            "analytic_combined_rear_cl_bias_derivative": 0.5 * sum(
                row["analytic_normalized_rear_cl_bias_derivative"] for row in domains.values()),
            "force_forward_batch_sizes": [20] * 10, "force_predictor_training_mode": True,
            "autograd_enabled": False, "physical_residual_formula": "(pred_norm-target_norm)*std in float32"}


def compare_panels(parent, terminal):
    result = {"objective_delta": {key: terminal["objective"][key] - parent["objective"][key]
                                  for key in ("h1_balanced", "ar_balanced", "total")}, "domains": {}}
    for domain in ("h1", "ar"):
        result["domains"][domain] = {}
        for window in ("h100", "tail62"):
            a, b = parent["domains"][domain][window], terminal["domains"][domain][window]
            result["domains"][domain][window] = {
                key + "_delta": b[key] - a[key] for key in
                ("signed_residual_mean", "mse", "bias_mse", "centered_residual_mse")}
    return result


def validate_windows(items):
    observed = tuple((x["global_index"], x["family"], x["identity"]["case"], x["identity"]["start"])
                     for x in items)
    if observed != WINDOWS or any(x["identity"].get("split") != "train" or
                                 x["identity"].get("rollout_steps") != 100 for x in items):
        raise ValueError("fixed six train-window contract differs")


def preflight(args):
    root = args.source_root.resolve()
    for name, expected in SOURCE_SHA.items():
        if sha256(root / name) != expected:
            raise ValueError("immutable source differs: " + name)
    for path, expected in ((args.config, CONFIG_SHA), (args.dual_manifest, MANIFEST_SHA),
                           (args.training_result, RESULT_SHA)):
        if sha256(path) != expected:
            raise ValueError("fixed P013 identity differs: " + str(path))
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from fluid_control import dual_fno

    if Path(dual_fno.__file__).resolve() != root / "src/fluid_control/dual_fno.py":
        raise ValueError("dual loader was imported from another source tree")
    identity = dual_fno.validate_dual_fno_manifest(args.dual_manifest, expected_sha256=MANIFEST_SHA)
    if args.output.exists():
        raise FileExistsError(args.output)
    return identity


def execute(args, identity):
    resource_checks = [check_memory()]
    objective = load_objective(args.source_root)
    trainer = objective.load_frozen_trainer(args.source_root / "scripts/train_fcp011_decoder_scope.py")
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.dual_fno import load_dual_fno, validate_dual_runtime_files, validate_runtime_precision
    from train_tandem_fno import build_model, configured_force_indices, predict

    cfg = OmegaConf.load(args.config)
    input_sha = trainer.validate_data_contract(cfg)
    validate_dual_runtime_files(identity, config_path=args.config,
                               normalization_path=Path(cfg.data.root) / "normalization.json")
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single CUDA device required for eventual approved execution")
    torch.cuda.set_per_process_memory_fraction(0.15, dist.device)
    precision = validate_runtime_precision()
    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)
    indices = configured_force_indices(cfg)
    if tuple(indices) != (0, 1, 2, 3):
        raise ValueError("four-force channel order differs")
    base = TandemRolloutDataset(cfg.data.root, "train", 100, stride=int(cfg.training.train_stride),
                               num_workers=cfg.training.workers, force_indices=indices)
    train = None
    try:
        train, _ = compose_training_data(base, [Path(x) for x in cfg.data.additional_train_roots],
            rollout_steps=100, stride=int(cfg.training.additional_train_stride),
            workers=cfg.training.workers, force_indices=indices)
        items = trainer.diagnostic_windows(train)
        validate_windows(items)
        dual, _ = load_dual_fno(args.dual_manifest, cfg, dist.device, build_model=build_model,
                               expected_manifest_sha256=MANIFEST_SHA)
        before = objective.tensor_state_sha256(dual)
        rows = []
        with torch.no_grad():
            for item in items:
                resource_checks.append(check_memory())
                batch = {key: value[None].to(dist.device) for key, value in item["sample"].items()}
                states = objective.frozen_flow_states(dual.flow_model, batch["state"], batch["mask"], batch["omega"], predict)
                h1 = objective.true_state_inputs(batch["state"], batch["target_state"])
                if not torch.isfinite(states).all() or not torch.isfinite(h1).all():
                    raise FloatingPointError("nonfinite fixed input states")
                panels = {label: capture_force_panel(objective, model, states, h1, batch,
                    base.force_std.to(dist.device), predict) for label, model in
                    (("p009_parent", dual.flow_model), ("p013_terminal", dual.aerodynamic_model))}
                rows.append({**{key: item[key] for key in ("global_index", "family", "identity")},
                             "panels": panels, "comparison": compare_panels(*panels.values())})
                print(json.dumps({"event": "fixed_window_complete", "global_index": item["global_index"]}), flush=True)
        after = objective.tensor_state_sha256(dual)
        if before != after or any(p.grad is not None for p in dual.parameters()):
            raise RuntimeError("read-only diagnostic changed model tensors or created gradients")
        return {"status": "FC_P014_TRAIN_OBJECTIVE_DIAGNOSTIC_COMPLETE_NOT_ADMISSION",
                "rows": rows, "source_sha256": SOURCE_SHA, "input_sha256": input_sha,
                "dual_manifest_sha256": MANIFEST_SHA, "training_result_sha256": RESULT_SHA,
                "config_sha256": CONFIG_SHA, "diagnostic_script_sha256": sha256(Path(__file__)),
                "precision": precision, "tensor_sha256_before": before, "tensor_sha256_after": after,
                "resource_checks": resource_checks, "frozen_flow_training_mode": False,
                "earlier_fixed_six_force_mode": "eval; this diagnostic explicitly uses train mode",
                "optimizer_created": False, "optimizer_steps": 0, "backward_performed": False,
                "candidate_saved": False, "selection_performed": False, "validation_accessed": False,
                "frozen_test_accessed": False, "ppo_executed": False,
                "bias_derivative_method": "analytic affine-bias coordinate; not measured autograd or stationarity",
                "comparison_scope": "fixed six train windows, no admission thresholds or convergence claim"}
    finally:
        (train if train is not None else base).close()


def check_memory(path=Path("/proc/meminfo")):
    values = {line.split()[0].rstrip(":"): int(line.split()[1]) for line in path.read_text().splitlines()
              if line.startswith(("MemFree:", "MemAvailable:"))}
    if any(values.get(key, 0) < 20 * 1024 ** 2 for key in ("MemFree", "MemAvailable")):
        raise RuntimeError("MemFree and MemAvailable must both be >=20 GiB")
    return values


def write_exclusive(path, result):
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "config", "dual-manifest", "training-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    identity = preflight(args)
    if not args.execute:
        print("FC_P014_IDENTITIES_VERIFIED_NO_GPU_NO_DIAGNOSTIC")
        return
    write_exclusive(args.output, execute(args, identity))


if __name__ == "__main__":
    main()
