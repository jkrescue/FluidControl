"""Project adapter for an official-FNO total+pressure auxiliary head.

The original seven outputs remain the deployment interface. Four appended
outputs predict train-only normalized physical pressure coefficients.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader

PRESSURE_MEAN = torch.tensor(
    [1.0419116437149978, -0.00006205397649474777,
     0.6305951529062531, -0.00013873081349062121], dtype=torch.float32)
PRESSURE_STD = torch.tensor(
    [0.009185585794470026, 0.2768732706035754,
     0.18729255004083134, 1.1097941813343266], dtype=torch.float32)
AUX_WEIGHT = 0.1
FINAL_WEIGHT = "decoder_net.final_layer.linear.weight"
FINAL_BIAS = "decoder_net.final_layer.linear.bias"


def require(value, message):
    if not value:
        raise ValueError(message)


def initialize_out11_from_out7(parent, candidate) -> None:
    """Copy old outputs exactly and initialize standardized pressure as total."""
    old = dict(parent.named_parameters())
    new = dict(candidate.named_parameters())
    require(old.keys() == new.keys(), "parameter names differ")
    with torch.no_grad():
        for name, value in old.items():
            target = new[name]
            if name in (FINAL_WEIGHT, FINAL_BIAS):
                require(value.shape[0] == 7 and target.shape[0] == 11, "final rows")
                target[:7].copy_(value)
                target[7:11].copy_(value[3:7])
            else:
                require(target.shape == value.shape, f"shape differs: {name}")
                target.copy_(value)


def extract_total_pressure(raw: torch.Tensor, mask: torch.Tensor):
    require(raw.ndim == 4 and raw.shape[1] == 11, "B11HW output required")
    require(mask.shape == (raw.shape[0], 1, raw.shape[2], raw.shape[3]), "mask shape")
    require(torch.isfinite(raw).all() and torch.isfinite(mask).all(), "nonfinite")
    denominator = mask.sum((-2, -1)).clamp_min(1)
    total = (raw[:, 3:7] * mask).sum((-2, -1)) / denominator
    pressure = (raw[:, 7:11] * mask).sum((-2, -1)) / denominator
    return raw[:, :3], total, pressure


def normalize_pressure(physical: torch.Tensor) -> torch.Tensor:
    require(physical.shape[-1] == 4 and torch.isfinite(physical).all(), "pressure target")
    mean = PRESSURE_MEAN.to(device=physical.device, dtype=physical.dtype)
    std = PRESSURE_STD.to(device=physical.device, dtype=physical.dtype)
    return (physical - mean) / std


def pressure_h1_mse(predicted: torch.Tensor, physical_target: torch.Tensor) -> torch.Tensor:
    target = normalize_pressure(physical_target)
    require(predicted.shape == target.shape, "pressure prediction/target shape")
    return (predicted - target).square().mean()


def chunk_total_pressure_objective(
    aerodynamic_model, flow_states, h1_states, mask, omega, target_force,
    target_pressure_physical, preceding_states, preceding_actions,
    raw_forward, history_module, original_objective, *, backward,
):
    """Original ten-chunk total objective plus fixed H1 pressure auxiliary."""
    require(flow_states.shape == h1_states.shape, "H1/AR state shape")
    require(target_force.shape == target_pressure_physical.shape ==
            (flow_states.shape[0], 100, 4), "force/pressure target alignment")
    totals = {key: torch.zeros((), device=flow_states.device)
              for key in ("h1_balanced", "ar_balanced", "original_total",
                          "pressure_h1", "pressure_aux_weighted", "training_objective")}
    channel_h1 = torch.zeros(4, device=flow_states.device)
    channel_ar = torch.zeros(4, device=flow_states.device)
    batch = flow_states.shape[0]
    captured = {"h1": [], "ar": []}
    for begin in range(0, 100, 10):
        end = begin + 10
        h1, h1_masks = history_module.chunk_inputs(
            h1_states, preceding_states, mask, omega, preceding_actions, begin, end)
        ar, ar_masks = history_module.chunk_inputs(
            flow_states, preceding_states, mask, omega, preceding_actions, begin, end)
        raw = raw_forward(aerodynamic_model, torch.cat((h1, ar), 0))
        _, total, pressure = extract_total_pressure(raw, torch.cat((h1_masks, ar_masks), 0))
        h1_total, ar_total = total[:batch * 10], total[batch * 10:]
        targets = target_force[:, begin:end].reshape(batch * 10, 4)
        hl = original_objective.balanced_force_objective(h1_total[:, None], targets[:, None])
        al = original_objective.balanced_force_objective(ar_total[:, None], targets[:, None])
        old = 0.5 * hl["balanced"] + 0.5 * al["balanced"]
        pressure_loss = pressure_h1_mse(
            pressure[:batch * 10],
            target_pressure_physical[:, begin:end].reshape(batch * 10, 4))
        objective = old + AUX_WEIGHT * pressure_loss
        if backward:
            (0.1 * objective).backward()
        for key, value in (("h1_balanced", hl["balanced"]),
                           ("ar_balanced", al["balanced"]),
                           ("original_total", old),
                           ("pressure_h1", pressure_loss),
                           ("pressure_aux_weighted", AUX_WEIGHT * pressure_loss),
                           ("training_objective", objective)):
            totals[key] += 0.1 * value.detach()
        channel_h1 += 0.1 * hl["channel_mse"].detach()
        channel_ar += 0.1 * al["channel_mse"].detach()
        captured["h1"].append(h1_total.detach().reshape(batch, 10, 4))
        captured["ar"].append(ar_total.detach().reshape(batch, 10, 4))
    return dict(**{key: float(value) for key, value in totals.items()},
                total=float(totals["original_total"]),
                h1_channel_mse=[float(x) for x in channel_h1],
                ar_channel_mse=[float(x) for x in channel_ar],
                chunk_size=10, chunks=10,
                normalized_predictions={domain: torch.cat(parts, 1)
                                        for domain, parts in captured.items()})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PressureSidecars:
    """Small official-Reader in-memory lookup; no state/action/field payload."""
    def __init__(self, root: Path, manifest_path: Path):
        self.root = Path(root)
        self.data = {}
        manifest = json.loads(Path(manifest_path).read_text())
        require(manifest["cases"] == 45 and manifest["frames"] == 20493 and
                len(manifest["rows"]) == 45, "sidecar manifest counts")
        for item in manifest["rows"]:
            path = self.root / item["family"] / (item["case"] + ".components.h5")
            require(_sha(path) == item["sidecar_sha256"], "sidecar SHA")
            reader = HDF5Reader(path, fields=["time", "pressure"])
            try:
                rows = [reader[index][0] for index in range(len(reader))]
            finally:
                reader.close()
            training_case = Path(item["hdf_path"]).stem
            require(training_case not in self.data, "duplicate training case")
            self.data[training_case] = {
                "time": torch.stack([row["time"].float().reshape(1) for row in rows]),
                "pressure": torch.stack([row["pressure"].float() for row in rows]),
            }

    def target(self, identity: dict, initial_time: torch.Tensor) -> torch.Tensor:
        case, start = identity["case"], int(identity["start"])
        require(case in self.data, f"missing sidecar: {case}")
        row = self.data[case]
        require(0 <= start and start + 100 < len(row["time"]), "sidecar H100 range")
        observed = float(initial_time.detach().cpu().reshape(()))
        expected = float(row["time"][start, 0])
        require(abs(observed - expected) <= 2e-5, "sidecar initial time differs")
        clocks = row["time"][start:start + 101, 0]
        expected_clocks = clocks[0] + torch.arange(101, dtype=clocks.dtype) * 0.1
        require(torch.allclose(clocks, expected_clocks, atol=2e-5, rtol=0),
                "sidecar target clock differs")
        target = row["pressure"][start + 1:start + 101]
        require(target.shape == (100, 4), "sidecar target shape")
        return target
