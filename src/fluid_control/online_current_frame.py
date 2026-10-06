"""Project-only CPU current-frame bridge; no reader, model or control execution."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

HISTORY_SHA = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def normalize_current(packet, *, expected_time, expected_mask, normalization_bytes,
                      expected_normalization_sha256):
    """Validate caller-bound current packet; never fetch next-frame labels."""
    require(set(packet) == {"state", "mask", "time", "x", "y"}, "packet keys")
    require(type(expected_time) in (int, float) and math.isfinite(expected_time), "expected time")
    require(type(normalization_bytes) is bytes, "normalization bytes")
    digest = hashlib.sha256(normalization_bytes).hexdigest()
    require(digest == expected_normalization_sha256, "normalization SHA")
    stats = json.loads(normalization_bytes)
    require(stats["state_channels"] == ["u", "v", "gauge_pressure"], "state channels")
    for key in ("state_mean", "state_std"):
        require(isinstance(stats[key], list) and len(stats[key]) == 3, "normalization shape")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in stats[key]), "normalization values")
    mean = torch.tensor(stats["state_mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(stats["state_std"], dtype=torch.float32)[:, None, None]
    require(bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()), "FP32 normalization")
    for name, shape, dtype in (("state", (3,128,256), np.float32),
                               ("mask", (1,128,256), np.uint8),
                               ("x", (256,), np.float32), ("y", (128,), np.float32)):
        value = packet[name]
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype, name + " shape/dtype")
        require(bool(np.isfinite(value).all()), name + " finite")
    mask = packet["mask"]
    require(bool(np.isin(mask, [0,1]).all() and mask.any()), "binary nonempty mask")
    require(isinstance(expected_mask, np.ndarray) and expected_mask.shape == mask.shape
            and expected_mask.dtype == mask.dtype and np.array_equal(mask, expected_mask), "expected mask")
    require(np.array_equal(packet["x"], np.linspace(8,25,256,dtype=np.float32))
            and np.array_equal(packet["y"], np.linspace(4,11,128,dtype=np.float32)), "grid coordinates")
    require(bool((packet["state"][:, mask[0] == 0] == 0).all()), "solid state must be zero")
    time = packet["time"]
    require(isinstance(time, np.ndarray) and time.shape == (1,) and time.dtype.kind == "f"
            and bool(np.isfinite(time).all()), "time shape/finite")
    require(abs(float(time[0]) - expected_time) <= 1e-5, "current time mismatch")
    physical = torch.from_numpy(packet["state"].copy())
    mask_tensor = torch.from_numpy(mask.copy()).float()
    state = (physical - mean) / std
    state *= mask_tensor
    require(bool(torch.isfinite(state).all()), "normalized state finite")
    return {"state": state[None], "mask": mask_tensor[None], "time": float(time[0]),
            "normalization_sha256": digest}


def build_current_input(current, *, applied_omega_now, constrained_omega_next):
    """K1 packing only. Reject illegal commands; never choose or clamp actions."""
    import p026_state_history as history
    require(hashlib.sha256(Path(history.__file__).read_bytes()).hexdigest() == HISTORY_SHA,
            "canonical history helper SHA")
    for value in (applied_omega_now, constrained_omega_next):
        require(type(value) in (int, float) and math.isfinite(value) and abs(value) <= .75,
                "finite physical action within .75")
    require(abs(constrained_omega_next - applied_omega_now) <= .1 + 1e-12, "action rate")
    state, mask = current["state"], current["mask"]
    require(state.shape == (1,3,128,256) and mask.shape == (1,1,128,256), "current shapes")
    require(state.device.type == mask.device.type == "cpu" and state.dtype == mask.dtype == torch.float32,
            "CPU float32 only")
    require(not state.requires_grad and not mask.requires_grad, "no input gradients")
    now = torch.tensor([applied_omega_now], dtype=torch.float32) / .75
    nxt = torch.tensor(constrained_omega_next, dtype=torch.float32) / .75
    return history.build_input(state, mask[0], now, nxt)[None]
