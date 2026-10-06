#!/usr/bin/env python3
"""Project core for fixed saved-CFD H1--H5 replay; no model/runtime imports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

START_INDICES = (0, 100, 200, 300, 400, 500, 600, 700)
HORIZON = 5
BRANCHES = ("mpc", "zero")
FORCE_SLICE = slice(64, 68)
EXPECTED_ROWS = 800
EXPECTED_FRAMES = 96
EXPECTED_ENDPOINTS = 80


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def finite_number(value, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), name)
    return float(value)


def time_name(value: float) -> str:
    return f"{value:.10g}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(row: dict, key: str) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float64)
    require(value.shape == (69,) and np.isfinite(value).all(), key)
    return value


def validate_progress(document: dict) -> list[dict]:
    """Validate the actual 800-cycle record and executed endpoint convention."""
    rows = document.get("rows")
    require(type(document.get("completed_cycles")) is int, "completed_cycles type")
    require(document["completed_cycles"] == EXPECTED_ROWS, "completed_cycles")
    require(isinstance(rows, list) and len(rows) == EXPECTED_ROWS, "rows")
    previous = 0.0
    for index, row in enumerate(rows):
        require(row.get("step") == index + 1, "row step")
        start = finite_number(row.get("start_time"), "start_time")
        end = finite_number(row.get("end_time"), "end_time")
        requested = finite_number(row.get("requested_omega"), "requested_omega")
        applied = finite_number(row.get("applied_omega"), "applied_omega")
        require(abs(start - (148.0 + 0.1 * index)) <= 1e-8, "start grid")
        require(abs(end - (148.0 + 0.1 * (index + 1))) <= 1e-8, "end grid")
        require(abs(applied) <= 0.75 + 1e-12 and abs(applied - previous) <= 0.1 + 1e-12,
                "executed action bounds")
        require(abs(requested) <= 0.75 + 1e-12, "requested action bounds")
        input_obs = _observation(row, "input_observation")
        output_obs = _observation(row, "output_observation")
        zero_obs = _observation(row, "zero_observation")
        require(abs(float(input_obs[68]) - previous) <= 1e-7, "input action continuity")
        require(abs(float(output_obs[68]) - applied) <= 1e-7, "output action endpoint")
        require(abs(float(zero_obs[68])) <= 1e-7, "zero action endpoint")
        previous = applied
    return rows


def make_selection(document: dict, source_root: Path, *, require_files: bool = True) -> dict:
    """Select the predeclared 96 frames and bind five executed transitions."""
    rows = validate_progress(document)
    records = []
    for branch in BRANCHES:
        case = source_root / f"case_{branch}"
        for start in START_INDICES:
            frames = []
            for offset in range(HORIZON + 1):
                time = 148.0 + 0.1 * (start + offset)
                directory = case / time_name(time)
                paths = {name: directory / name for name in ("U", "p")}
                if require_files:
                    require(directory.is_dir() and all(path.is_file() for path in paths.values()),
                            "selected saved field missing")
                frames.append({"global_index": start + offset, "time": time,
                               "source": str(directory.relative_to(source_root)),
                               "files": {name: str(path.relative_to(source_root))
                                         for name, path in paths.items()}})
            transitions = []
            if branch == "mpc":
                initial_force = _observation(rows[start], "input_observation")[FORCE_SLICE]
            elif start == 0:
                initial_force = _observation(rows[0], "input_observation")[FORCE_SLICE]
            else:
                initial_force = _observation(rows[start - 1], "zero_observation")[FORCE_SLICE]
            for offset in range(HORIZON):
                index = start + offset
                if branch == "mpc":
                    now = 0.0 if index == 0 else float(rows[index - 1]["applied_omega"])
                    nxt = float(rows[index]["applied_omega"])
                    truth = _observation(rows[index], "output_observation")[FORCE_SLICE]
                else:
                    now = nxt = 0.0
                    truth = _observation(rows[index], "zero_observation")[FORCE_SLICE]
                transitions.append({"lead": offset + 1, "omega_now": now,
                                    "omega_next": nxt, "truth_force": truth.tolist()})
            records.append({"branch": branch, "start_index": start,
                            "start_time": frames[0]["time"], "frames": frames,
                            "initial_force": initial_force.tolist(),
                            "transitions": transitions})
    require(len(records) == 16 and sum(len(row["frames"]) for row in records) == EXPECTED_FRAMES,
            "selection cardinality")
    require(sum(len(row["transitions"]) for row in records) == EXPECTED_ENDPOINTS,
            "endpoint cardinality")
    return {"status": "PROJECTED_POLICY_H1_H5_SELECTION_PREPARED_NOT_EXECUTED",
            "starts": list(START_INDICES), "horizon": HORIZON,
            "frames": EXPECTED_FRAMES, "endpoints": EXPECTED_ENDPOINTS,
            "records": records, "scientific_admission": False}


def validate_packet(packet: dict, *, expected_time: float,
                    expected_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate one output from the exact reviewed Curator sampler."""
    require(set(packet) == {"state", "mask", "time", "x", "y"}, "packet keys")
    state, mask = packet["state"], packet["mask"]
    require(isinstance(state, np.ndarray) and state.shape == (3, 128, 256)
            and state.dtype == np.float32 and np.isfinite(state).all(), "state")
    require(isinstance(mask, np.ndarray) and mask.shape == (1, 128, 256)
            and mask.dtype == np.uint8 and np.isin(mask, (0, 1)).all() and mask.any(), "mask")
    require(np.array_equal(packet["x"], np.linspace(8, 25, 256, dtype=np.float32)), "x")
    require(np.array_equal(packet["y"], np.linspace(4, 11, 128, dtype=np.float32)), "y")
    time = packet["time"]
    require(isinstance(time, np.ndarray) and time.shape == (1,)
            and np.isfinite(time).all() and abs(float(time[0]) - expected_time) <= 1e-5,
            "sample time")
    require(np.all(state[:, mask[0] == 0] == 0), "invalid state must be zero")
    valid = mask[0].astype(bool)
    require(abs(float(state[2, valid].mean(dtype=np.float64))) <= 2e-6,
            "pressure must be ROI centered")
    if expected_mask is not None:
        require(np.array_equal(mask, expected_mask), "mask changed")
    return state, mask


def pack_and_verify_mini_hdf(record: dict, packets: list[dict], output: Path,
                             read_trajectory: Callable) -> dict:
    """Pack six converted frames, then reread them through official HDF5Reader."""
    import h5py

    require(len(packets) == HORIZON + 1 and not output.exists(), "mini-HDF inputs/output")
    states, masks, times = [], [], []
    anchor_mask = anchor_x = anchor_y = None
    for index, (packet, frame) in enumerate(zip(packets, record["frames"])):
        state, mask = validate_packet(packet, expected_time=frame["time"],
                                      expected_mask=anchor_mask)
        if index == 0:
            anchor_mask, anchor_x, anchor_y = mask.copy(), packet["x"].copy(), packet["y"].copy()
        else:
            require(np.array_equal(packet["x"], anchor_x)
                    and np.array_equal(packet["y"], anchor_y), "grid changed")
        states.append(state); masks.append(mask); times.append([float(packet["time"][0])])
    omega = [record["transitions"][0]["omega_now"]]
    omega.extend(row["omega_next"] for row in record["transitions"])
    forces = [record["initial_force"]]
    forces.extend(row["truth_force"] for row in record["transitions"])
    arrays = {"state": np.asarray(states, dtype=np.float32),
              "mask": np.asarray(masks, dtype=np.uint8),
              "omega": np.asarray(omega, dtype=np.float32)[:, None],
              "force": np.asarray(forces, dtype=np.float32),
              "time": np.asarray(times, dtype=np.float64),
              "x": anchor_x, "y": anchor_y}
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as handle:
        for key, value in arrays.items():
            handle.create_dataset(key, data=value)
    fields, static = read_trajectory(output)
    for key in ("state", "mask", "omega", "force", "time"):
        actual = fields[key].detach().cpu().numpy()
        require(actual.dtype == arrays[key].dtype and np.array_equal(actual, arrays[key]),
                f"official reader changed {key}")
    require(np.array_equal(static["x"], anchor_x) and np.array_equal(static["y"], anchor_y),
            "official reader changed coordinates")
    return {"path": str(output), "sha256": file_sha256(output),
            "frames": HORIZON + 1, "official_reader_verified": True,
            "mask_sha256": hashlib.sha256(anchor_mask.tobytes()).hexdigest(),
            "x_sha256": hashlib.sha256(anchor_x.tobytes()).hexdigest(),
            "y_sha256": hashlib.sha256(anchor_y.tobytes()).hexdigest()}


def replay_one(states: list[np.ndarray], mask: np.ndarray, transitions: list[dict],
               predict_step: Callable[[np.ndarray, np.ndarray, np.float32, np.float32],
                                      tuple[np.ndarray, np.ndarray]]) -> dict:
    """Run one free-AR H5 chain. Only states[0] is exposed to the predictor."""
    require(len(states) == HORIZON + 1 and len(transitions) == HORIZON, "H5 inputs")
    for state in states:
        require(state.shape == (3, 128, 256) and state.dtype == np.float32
                and np.isfinite(state).all(), "truth state")
    predicted_state = states[0].copy()
    predicted_states, predicted_forces = [], []
    for transition in transitions:
        now = np.float32(transition["omega_now"])
        nxt = np.float32(transition["omega_next"])
        result_state, result_force = predict_step(predicted_state, mask, now, nxt)
        result_state, result_force = np.asarray(result_state), np.asarray(result_force)
        require(result_state.shape == predicted_state.shape and result_state.dtype == np.float32
                and np.isfinite(result_state).all(), "predicted state")
        require(result_force.shape == (4,) and np.isfinite(result_force).all(), "predicted force")
        predicted_state = result_state
        predicted_states.append(result_state.copy())
        predicted_forces.append(result_force.astype(np.float64).copy())
    return {"predicted_states": predicted_states,
            "predicted_forces": predicted_forces,
            "truth_states": [state.copy() for state in states[1:]]}


def sufficient_statistics(records: list[dict]) -> dict:
    """Pool physical SSE/reference and force absolute errors by endpoint lead."""
    require(records, "records")
    output = {}
    for lead in range(1, HORIZON + 1):
        field_sse = np.zeros(3, dtype=np.float64)
        field_ref = np.zeros(3, dtype=np.float64)
        persistence_sse = np.zeros(3, dtype=np.float64)
        force_abs = np.zeros(4, dtype=np.float64)
        persistence_force_abs = np.zeros(4, dtype=np.float64)
        total_drag_abs = persistence_total_drag_abs = 0.0
        count = 0
        for row in records:
            mask = np.asarray(row["mask"], dtype=bool)[0]
            pred = np.asarray(row["predicted_states"][lead - 1], dtype=np.float64)
            truth = np.asarray(row["truth_states"][lead - 1], dtype=np.float64)
            initial = np.asarray(row["truth_states_initial"], dtype=np.float64)
            pforce = np.asarray(row["predicted_forces"][lead - 1], dtype=np.float64)
            tforce = np.asarray(row["truth_forces"][lead - 1], dtype=np.float64)
            iforce = np.asarray(row["truth_force_initial"], dtype=np.float64)
            require(pred.shape == truth.shape == initial.shape == (3, 128, 256), "metric state")
            require(pforce.shape == tforce.shape == iforce.shape == (4,), "metric force")
            require(all(np.isfinite(value).all() for value in
                        (pred, truth, initial, pforce, tforce, iforce)), "metric finite")
            for channel in range(3):
                field_sse[channel] += np.square(pred[channel, mask] - truth[channel, mask]).sum()
                field_ref[channel] += np.square(truth[channel, mask]).sum()
                persistence_sse[channel] += np.square(
                    initial[channel, mask] - truth[channel, mask]).sum()
            force_abs += np.abs(pforce - tforce)
            persistence_force_abs += np.abs(iforce - tforce)
            total_drag_abs += abs(float(pforce[0] + pforce[2] - tforce[0] - tforce[2]))
            persistence_total_drag_abs += abs(
                float(iforce[0] + iforce[2] - tforce[0] - tforce[2]))
            count += 1
        require(count == len(records) and np.all(field_ref > 0), "metric count/reference")
        output[str(lead)] = {
            "field_relative_l2_u_v_p": np.sqrt(field_sse / field_ref).tolist(),
            "velocity_relative_l2": float(math.sqrt(
                (field_sse[0] + field_sse[1]) / (field_ref[0] + field_ref[1]))),
            "persistence_field_relative_l2_u_v_p":
                np.sqrt(persistence_sse / field_ref).tolist(),
            "force_channel_mae": (force_abs / count).tolist(),
            "persistence_force_channel_mae": (persistence_force_abs / count).tolist(),
            "total_drag_mae": total_drag_abs / count,
            "persistence_total_drag_mae": persistence_total_drag_abs / count,
            "field_squared_error_sums_u_v_p": field_sse.tolist(),
            "field_reference_squared_sums_u_v_p": field_ref.tolist(),
            "persistence_field_squared_error_sums_u_v_p": persistence_sse.tolist(),
            "segments": count,
        }
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(args.prepare_only, "only metadata preparation is implemented/authorized")
    require(not args.output.exists(), "exclusive output required")
    document = json.loads(args.progress.read_text())
    selection = make_selection(document, args.source_root.resolve())
    selection["progress_sha256"] = file_sha256(args.progress)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "selection.json"
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": selection["status"], "selection_sha256": file_sha256(path)}))


if __name__ == "__main__":
    main()
