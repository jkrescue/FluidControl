#!/usr/bin/env python3
"""Compare FC-P003C true-state H1 force errors on three fixed windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

MODEL_SHA = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
STATE_SHA = "a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a"
TRAIN8_SHA = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
BASE_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
DYNAMIC_SHA = "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae"
PAIR_SHA = "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
NORM_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
SEGMENTS_SHA = "626611c7fddca8046a18d884cfa676b4d2defae91605b32df8ab5673904070d9"
POSTEVAL_SHA = "0ee2468b193b25e09cf2bc1b2c71a43fc918888788db92b2f115149f21cb8338"
CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
WINDOWS = {
    "train_paired_window": (1, 100),
    "train_late_window": (100, 200),
    "validation_late_window": (100, 200),
}


def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def endpoint_indices(panel):
    if panel not in WINDOWS:
        raise ValueError(f"unknown panel: {panel}")
    a, b = WINDOWS[panel]
    return list(range(a, b + 1))


def summarize(values):
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("finite nonempty values required")
    return {
        "count": len(values),
        "mae": math.fsum(abs(v) for v in values) / len(values),
        "rmse": math.sqrt(math.fsum(v * v for v in values) / len(values)),
        "bias": math.fsum(values) / len(values),
    }


def summarize_rows(action_rows, zero_rows):
    ag, zg = defaultdict(list), defaultdict(list)
    for r in action_rows:
        for c in CHANNELS:
            ag[(r["panel"], r["phase"], r["profile"], c)].append(r["channels"][c])
    for r in zero_rows:
        for c in CHANNELS:
            zg[(r["panel"], r["phase"], c)].append(r["channels"][c]["zero_error"])
    by_panel = {}
    for panel in sorted({r["panel"] for r in action_rows}):
        actions = [r for r in action_rows if r["panel"] == panel]
        zeros = [r for r in zero_rows if r["panel"] == panel]
        absolute = actions if panel == "validation_late_window" else actions + zeros
        nonzero = [r for r in actions if r["profile"] != "zero"]
        absolute_values = [
            r["channels"][c]["action_error" if "profile" in r else "zero_error"]
            for r in absolute
            for c in CHANNELS
        ]
        delta_values = [
            r["channels"][c]["delta_error"] for r in nonzero for c in CHANNELS
        ]
        by_panel[panel] = {
            "absolute_micro": {
                **summarize(absolute_values),
                "endpoint_count": len(absolute),
            },
            "nonzero_action_delta_micro": {
                **summarize(delta_values),
                "endpoint_count": len(nonzero),
            },
        }
    return {
        "panel_micro": by_panel,
        "action_profiles": {
            "/".join(k): {
                "absolute_error": summarize([v["action_error"] for v in vs]),
                "delta_error": summarize([v["delta_error"] for v in vs]),
            }
            for k, vs in sorted(ag.items())
        },
        "deduplicated_zero_phases": {
            "/".join(k): summarize(vs) for k, vs in sorted(zg.items())
        },
    }


def validate_existing_segments(payload, rows):
    if payload.get("split") != "validation" or payload.get("action_mode") != "observed":
        raise ValueError("requires observed validation segments")
    lookup = {}
    for r in payload.get("segments", []):
        key = (r.get("case"), r.get("horizon"), r.get("start"))
        if key in lookup:
            raise ValueError("duplicate existing segment")
        lookup[key] = r
    diffs = []
    for r in rows:
        case = f"full40_dynamic_validation_{r['phase']}_{r['profile']}"
        old = lookup.get((case, 1, r["target_index"] - 1))
        if old is None or set(old.get("force_channel_absolute_error", {})) != set(
            CHANNELS
        ):
            raise ValueError("missing/malformed H1 segment")
        for c in CHANNELS:
            diffs.append(
                abs(
                    abs(r["channels"][c]["action_error"])
                    - float(old["force_channel_absolute_error"][c])
                )
            )
    if len(rows) != 606 or len(diffs) != 2424:
        raise ValueError("requires exact dynamic6 late panel")
    return {
        "compared_endpoint_channels": len(diffs),
        "max_absolute_error_difference": max(diffs),
        "note": "Floating-point reproduction check only; existing segments lack signed forces.",
    }


def _manifest(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"manifest SHA differs: {path}")
    return json.loads(Path(path).read_text())


def _load_hdf(path, digest, full_frames):
    import h5py
    import numpy as np

    if sha256(path) != digest:
        raise ValueError(f"HDF SHA differs: {path}")
    with h5py.File(path, "r") as h:
        if {len(h[k]) for k in ("state", "mask", "omega", "force", "time")} != {
            full_frames
        }:
            raise ValueError(f"full trajectory length differs: {path}")
        data = {
            k: np.asarray(h[k][:201])
            for k in ("state", "mask", "omega", "force", "time")
        }
    if {len(v) for v in data.values()} != {201} or not all(
        np.isfinite(v).all() for v in data.values()
    ):
        raise ValueError("requires 201 aligned finite frames")
    times, omega = data["time"].reshape(-1), data["omega"].reshape(-1)
    if not np.allclose(np.diff(times), 0.1, rtol=0, atol=2e-5):
        raise ValueError("time grid differs")
    if np.max(abs(omega)) > 0.75 + 1e-7 or np.max(abs(np.diff(omega))) > 0.1 + 2e-5:
        raise ValueError("action limits differ")
    if not np.array_equal(
        data["mask"], np.broadcast_to(data["mask"][0], data["mask"].shape)
    ):
        raise ValueError("mask changes")
    return data


def _normalized_pair(action, zero, endpoints, stats, device):
    import numpy as np
    import torch

    first, last = endpoints[0] - 1, endpoints[-1]
    idx = np.arange(first, last + 1)
    if not np.allclose(action["time"][idx], zero["time"][idx], rtol=0, atol=2e-5):
        raise ValueError("paired times differ")
    if not np.array_equal(action["mask"][idx], zero["mask"][idx]):
        raise ValueError("paired masks differ")
    mean = torch.tensor(stats["state_mean"], device=device)[None, None, :, None, None]
    std = torch.tensor(stats["state_std"], device=device)[None, None, :, None, None]

    def branch(src):
        state = torch.as_tensor(src["state"][idx], dtype=torch.float32, device=device)[
            None
        ]
        masks = torch.as_tensor(src["mask"][idx], dtype=torch.float32, device=device)[
            None
        ]
        omega = (
            torch.as_tensor(src["omega"][idx], dtype=torch.float32, device=device)[None]
            / 0.75
        )
        return (state - mean) / std * masks, omega

    ast, ao = branch(action)
    zst, zo = branch(zero)
    return {
        "action_state": ast,
        "zero_state": zst,
        "action_omega": ao,
        "zero_omega": zo,
        "mask": torch.as_tensor(
            action["mask"][first], dtype=torch.float32, device=device
        )[None],
    }


def _predict_pair(model, pair, chunk):
    import torch
    from train_tandem_fno import predict

    from fluid_control.paired_step_training import predict_true_state_force_chunk

    aa, zz = [], []
    total = pair["action_state"].shape[1] - 1
    with torch.no_grad():
        for start in range(0, total, chunk):
            pa, pz = predict_true_state_force_chunk(
                model,
                pair,
                start,
                min(start + chunk, total),
                lambda n, x, m: predict(n, x, m)[1],
            )
            aa.append(pa)
            zz.append(pz)
    return torch.cat(aa, 1)[0], torch.cat(zz, 1)[0]


def _rows(panel, phase, profile, endpoints, action, zero, pa, pz):
    out = []
    for i, target in enumerate(endpoints):
        channels = {}
        for j, c in enumerate(CHANNELS):
            ae = float(pa[i, j] - action["force"][target, j])
            ze = float(pz[i, j] - zero["force"][target, j])
            td = float(action["force"][target, j] - zero["force"][target, j])
            pd = float(pa[i, j] - pz[i, j])
            channels[c] = {
                "action_error": ae,
                "zero_error": ze,
                "target_delta": td,
                "predicted_delta": pd,
                "delta_error": pd - td,
            }
        out.append(
            {
                "panel": panel,
                "source_split": "validation"
                if panel == "validation_late_window"
                else "train",
                "phase": phase,
                "profile": profile,
                "target_index": target,
                "start_index": target - 1,
                "target_time": float(action["time"][target]),
                "extra_paired_supervision": panel.startswith("train_")
                and target <= 100,
                "regular_training_coverage": panel.startswith("train_"),
                "channels": channels,
            }
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train8",
        "base",
        "dynamic-validation",
        "pair-manifest",
        "normalization",
        "config",
        "checkpoint-dir",
        "posteval-receipt",
        "existing-validation-segments",
        "output",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=10)
    a = p.parse_args()
    if (
        a.output.exists()
        or isinstance(a.chunk_size, bool)
        or not 1 <= a.chunk_size <= 101
    ):
        raise ValueError("new output and chunk-size [1,101] required")
    tm = _manifest(a.train8 / "manifest.json", TRAIN8_SHA)
    _manifest(a.base / "manifest.json", BASE_SHA)
    dm = _manifest(a.dynamic_validation / "manifest.json", DYNAMIC_SHA)
    pm = _manifest(a.pair_manifest, PAIR_SHA)
    if sha256(a.normalization) != NORM_SHA:
        raise ValueError("normalization SHA differs")
    if (
        sha256(a.config) != CONFIG_SHA
        or sha256(a.existing_validation_segments) != SEGMENTS_SHA
        or sha256(a.posteval_receipt) != POSTEVAL_SHA
    ):
        raise ValueError("fixed C config/posteval evidence differs")
    posteval = json.loads(a.posteval_receipt.read_text())
    if (
        posteval.get("checkpoint_sha256") != MODEL_SHA
        or posteval.get("status") != "FC_P003C_POSTEVAL_COMPLETE"
        or posteval.get("frozen_test_accessed")
        or posteval.get("ppo_auto_launched")
        or posteval.get("sha256", {}).get("dynamic6/segments.json") != SEGMENTS_SHA
    ):
        raise ValueError("C posteval receipt contract differs")
    stats = json.loads(a.normalization.read_text())
    if tuple(stats.get("all_force_channels", ())) != CHANNELS:
        raise ValueError("force-channel order differs")
    if tm.get("max_abs_omega") != 0.75:
        raise ValueError("train8 action scale differs")
    if tm.get("trajectory_counts") != {"train": 8, "validation": 0, "frozen_test": 0}:
        raise ValueError("train8 split differs")
    if dm.get("trajectory_counts") != {"train": 0, "validation": 6, "frozen_test": 0}:
        raise ValueError("dynamic6 split differs")
    if pm.get("sequence_length") != 101 or pm.get("pair_count") != 8:
        raise ValueError("paired prefix differs")
    models = list(a.checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = list(a.checkpoint_dir.glob("checkpoint.0.*.pt"))
    if (
        len(models) != 1
        or sha256(models[0]) != MODEL_SHA
        or len(states) != 1
        or sha256(states[0]) != STATE_SHA
    ):
        raise ValueError("checkpoint differs")
    import torch
    from evaluate_tandem_fno import load_composed_config
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA diagnostic required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    cfg = load_composed_config(a.config)
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("four force outputs required")
    model = build_model(cfg).to(dist.device)
    if load_checkpoint(a.checkpoint_dir, models=model, device=dist.device) != 2:
        raise ValueError("expected epoch2")
    model.eval()
    fm = torch.tensor(stats["all_force_mean"], device=dist.device)
    fs = torch.tensor(stats["all_force_std"], device=dist.device)
    action_rows = []
    zero_cache = {}
    pairs = pm["pairs"]
    if {(x["phase"], x["profile"]) for x in pairs} != {
        (f"b{i:02d}", p) for i in (0, 2, 4, 6) for p in ("multisine", "prbs")
    }:
        raise ValueError("dynamic8 identities differ")

    def consume(panel, phase, profile, action, zero):
        endpoints = endpoint_indices(panel)
        pair = _normalized_pair(action, zero, endpoints, stats, dist.device)
        pa, pz = _predict_pair(model, pair, a.chunk_size)
        pa = (pa * fs + fm).cpu().numpy()
        pz = (pz * fs + fm).cpu().numpy()
        rows = _rows(panel, phase, profile, endpoints, action, zero, pa, pz)
        action_rows.extend(rows)
        key = (panel, phase)
        values = [(r["target_index"], r["target_time"], r["channels"]) for r in rows]
        if key in zero_cache:
            if any(
                left[:2] != right[:2]
                or any(
                    not math.isclose(
                        left[2][c]["zero_error"],
                        right[2][c]["zero_error"],
                        rel_tol=0,
                        abs_tol=2e-6,
                    )
                    for c in CHANNELS
                )
                for left, right in zip(zero_cache[key], values, strict=True)
            ):
                raise ValueError("repeated zero differs")
        else:
            zero_cache[key] = values

    for panel in ("train_paired_window", "train_late_window"):
        for x in pairs:
            consume(
                panel,
                x["phase"],
                x["profile"],
                _load_hdf(a.train8 / x["action_file"], x["action_hdf_sha256"], 201),
                _load_hdf(a.base / x["zero_file"], x["zero_hdf_sha256"], 801),
            )
    val = {
        Path(name).stem: _load_hdf(
            a.dynamic_validation / "validation" / name, digest, 201
        )
        for name, digest in dm["hdf_sha256"].items()
    }
    for phase in ("b01", "b05"):
        zero = val[f"full40_dynamic_validation_{phase}_zero"]
        for profile in ("minus", "zero", "plus"):
            consume(
                "validation_late_window",
                phase,
                profile,
                val[f"full40_dynamic_validation_{phase}_{profile}"],
                zero,
            )
    zero_rows = [
        {
            "panel": panel,
            "source_split": "train" if panel.startswith("train_") else "validation",
            "phase": phase,
            "target_index": target,
            "target_time": time,
            "channels": channels,
        }
        for (panel, phase), values in sorted(zero_cache.items())
        for target, time, channels in values
    ]
    counts = {panel: sum(r["panel"] == panel for r in action_rows) for panel in WINDOWS}
    if (
        counts
        != {
            "train_paired_window": 800,
            "train_late_window": 808,
            "validation_late_window": 606,
        }
        or len(zero_rows) != 1006
    ):
        raise ValueError("panel counts differ")
    sb = a.existing_validation_segments.read_bytes()
    vrows = [r for r in action_rows if r["panel"] == "validation_late_window"]
    result = {
        "status": "FCP003C_TRAIN_VS_VALIDATION_TRUE_STATE_H1_DIAGNOSTIC_COMPLETE",
        "scope": "diagnostic only; not admission, selection, training, or control evidence",
        "checkpoint_epoch": 2,
        "model_sha256": MODEL_SHA,
        "training_state_sha256": STATE_SHA,
        "force_channels": list(CHANNELS),
        "panels": {
            "train_paired_window": {
                "targets": [1, 100],
                "action_endpoint_count": 800,
                "deduplicated_absolute_endpoint_count": 1200,
                "nonzero_delta_endpoint_count": 800,
                "extra_paired_supervision": "all targets; regular training also covers this window",
            },
            "train_late_window": {
                "targets": [100, 200],
                "action_endpoint_count": 808,
                "deduplicated_absolute_endpoint_count": 1212,
                "nonzero_delta_endpoint_count": 808,
                "target100_overlap": "target100 is extra-paired; targets101..200 are not; all have regular coverage",
            },
            "validation_late_window": {
                "targets": [100, 200],
                "action_endpoint_count": 606,
                "deduplicated_absolute_endpoint_count": 606,
                "nonzero_delta_endpoint_count": 404,
                "split": "validation",
            },
        },
        "action_endpoint_rows": action_rows,
        "deduplicated_zero_endpoint_rows": zero_rows,
        "summaries": summarize_rows(action_rows, zero_rows),
        "existing_dynamic6_h1_reproduction": validate_existing_segments(
            json.loads(sb), vrows
        ),
        "input_sha256": {
            "train8_manifest": TRAIN8_SHA,
            "base_manifest": BASE_SHA,
            "dynamic_validation_manifest": DYNAMIC_SHA,
            "dynamic_pair_manifest": PAIR_SHA,
            "normalization": NORM_SHA,
            "config": CONFIG_SHA,
            "posteval_receipt": POSTEVAL_SHA,
            "existing_validation_segments": SEGMENTS_SHA,
        },
        "implementation_sha256": sha256(Path(__file__)),
        "chunk_size": a.chunk_size,
        "zero_prediction_accounting": "same-phase zero retained once; repeated computations agree within 2e-6",
        "frozen_test_accessed": False,
        "optimizer_steps": 0,
        "candidate_saved": False,
        "ppo_executed": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(result["status"], counts)


if __name__ == "__main__":
    main()
