"""Metadata-only non-authorizing draft, using an actual 24-packet CPU receipt."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from train_exploratory_diverse_h5_ppo import PROTOCOL, fixed_panel, require

ORIGINAL_APPROVAL_SHA = "8aa44f5f177d6c7d831a1efc55abf9d8db4b700d640cb640c5fcbb709cc44569"
REPO = Path("/workspace/fluid_control")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    require(sha(args.original_approval) == ORIGINAL_APPROVAL_SHA, "original runtime contract differs")
    require(sha(args.packet_proof) == args.packet_proof_sha256, "actual packet receipt differs")
    previous = json.loads(args.original_approval.read_text())
    proof = json.loads(args.packet_proof.read_text())
    require(proof["status"] == "DIVERSE_H5_REAL_RESET_PACKETS_VERIFIED_CPU_NOT_TRAINING"
            and proof["packet_count"] == 24 and proof["no_model_loaded"] is True
            and proof["no_gpu"] is True and proof["scientific_admission"] is False,
            "actual CPU-only packet verification required")
    require([{k: p[k] for k in ("case", "frame")} for p in proof["packets"]] == fixed_panel(),
            "fixed packet order differs")
    require(sha(args.adapter) in proof["source_sha256"].values(), "adapter not verified by CPU receipt")
    spec = copy.deepcopy(previous)
    spec.update(status="PREPARATION_ONLY_NOT_EXECUTION_APPROVAL", execution_authorized=False,
                protocol=PROTOCOL, reset_panel=fixed_panel(),
                packet_verification={"path": str(args.packet_proof.resolve()), "sha256": args.packet_proof_sha256},
                interpretation="Reset distribution alone changes; no training or CFD authorized by this draft")
    # Never inherit the earlier run's human authorization text/decision.
    for key in ("reviewed_by_lead", "lead_approval", "approval_statement", "execution_approval",
                "lead_statement", "preparation_spec_sha256", "source_root", "source_manifest_sha256"):
        spec.pop(key, None)
    spec["parent_source_manifest_sha256"] = previous["source_manifest_sha256"]
    spec["reviewed_by_lead"] = False
    data, cases = Path(spec["data_root"]), Path(spec["cases_root"])
    split = json.loads((data / "splits/train.json").read_text())
    case_order = [f"matched_start_acquisition_train_b{p}_{f}"
                  for p in ("00", "02", "04", "06")
                  for f in ("m075", "m0375", "zero", "p0375", "p075")]
    require(set(case_order) == set(split["cases"]), "exact original base-train20 membership")
    spec["train_hdf"] = [{"path": str(data / "train" / f"{c}.h5"),
                          "sha256": split["hdf5_sha256"][c]} for c in case_order]
    provenance = {r["path"]: r for r in previous["train_provenance_files"]}
    for packet in proof["packets"]:
        p = packet["provenance"]
        require(p["hdf_sha256"] == split["hdf5_sha256"][packet["case"]], "receipt HDF identity differs")
        config = cases / packet["case"] / "case_config.json"
        require(sha(config) == p["case_config_sha256"], "case metadata changed")
        provenance[str(config)] = {"path": str(config), "sha256": p["case_config_sha256"]}
        for rows in p["raw_sources"].values():
            for row in rows:
                raw = cases / row["path"]
                require(raw.resolve().is_relative_to(cases.resolve()), "raw provenance escaped cases")
                entry = {"path": str(raw), "sha256": row["sha256"]}
                require(str(raw) not in provenance or provenance[str(raw)] == entry,
                        "conflicting raw provenance")
                provenance[str(raw)] = entry
    spec["train_provenance_files"] = list(provenance.values())
    spec["case_metadata"] = [{k: p[k] for k in ("case", "frame", "time", "omega")}
                             for p in proof["packets"]]
    stage = Path(__file__).resolve().parent
    runner, supervisor = stage / "train_exploratory_diverse_h5_ppo.py", stage / "supervise_exploratory_diverse_h5_ppo.py"
    for path in (runner, supervisor, args.adapter.resolve(), Path(__file__).resolve()):
        spec["source_files"][str(path)] = sha(path)
    spec["import_bindings"]["exploratory_diverse_h5_resets"] = str(args.adapter.resolve())
    # Bind the actual official reader without importing it or reading HDF payloads.
    fno_source = Path(spec["import_bindings"]["physicsnemo.models.fno.fno"])
    reader = fno_source.parents[2] / "datapipes/readers/hdf5.py"
    require(sha(reader) in proof["source_sha256"].values(), "CPU official-reader proof differs")
    spec["runtime_sources"][str(reader)] = sha(reader)
    spec["import_bindings"]["physicsnemo.datapipes.readers.hdf5"] = str(reader)
    flow_source = Path(spec["import_bindings"]["fluid_control.tandem_hydrogym"])
    history = flow_source.with_name("openfoam_force_history.py")
    require(sha(history) in proof["source_sha256"].values(), "CPU raw-history source differs")
    spec["import_bindings"]["fluid_control.openfoam_force_history"] = str(history)
    spec["runner"] = str(runner)
    spec["pythonpath"] = list(dict.fromkeys([str(stage), str(args.adapter.resolve().parent), *previous["pythonpath"]]))
    spec["supervision_output"] = str(REPO / "artifacts/exploratory_diverse_h5_ppo_training_20261006")
    spec["output"] = str(Path(spec["supervision_output"]) / "payload")
    spec["parent_experiment_approval_sha256"] = ORIGINAL_APPROVAL_SHA
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-approval", type=Path, required=True)
    parser.add_argument("--packet-proof", type=Path, required=True)
    parser.add_argument("--packet-proof-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
