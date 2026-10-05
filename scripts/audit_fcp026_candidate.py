#!/usr/bin/env python3
"""P026 terminal integrity only. CPU archive inspection; no model/forward/admission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

TRAINER_SHA = "562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e"
LOADER_SHA = "3343dba367dd6e45fdc914fc321e90b94efc2d8a00553c61b025ef7d776fc2a8"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
FILES = ("result.json", "training_protocol.json", "dual_model_manifest.json",
         "flow/FNO.0.0.mdlus", "flow/checkpoint.0.0.pt",
         "aerodynamic/FNO.0.1.mdlus", "aerodynamic/checkpoint.0.1.pt")
LIFT = "spec_encoder.lift_network.0.conv.weight"
FROZEN = ("spec_encoder.lift_network.0.conv.bias", "spec_encoder.lift_network.2.conv.bias")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, label):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), label)


def fields(actual, expected, label):
    require(isinstance(actual, dict), label)
    equal({key: actual.get(key) for key in expected}, expected, label)


def validate_approved_protocol(actual, expected):
    """Accept only the approved clip norm's equivalent JSON integer spelling."""
    require(isinstance(actual, dict), "approved protocol must be an object")
    value = actual.get("gradient_clip_norm")
    wanted = expected["gradient_clip_norm"]
    require(type(value) in (int, float) and math.isfinite(value)
            and value == wanted, "approved protocol gradient_clip_norm differs")
    normalized = dict(actual)
    normalized["gradient_clip_norm"] = wanted
    equal(normalized, expected, "approved protocol")


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, label)
    return value


def checked(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected,
            f"byte identity differs: {path}")


def load_pinned(path, digest, name):
    checked(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_files(root):
    root = Path(root).resolve()
    result = {}
    for name in FILES:
        path = root / "candidate" / name
        require(path.resolve().is_relative_to(root / "candidate") and path.is_file(), "candidate file escaped/missing")
        result["candidate/" + name] = sha(path)
    return result


def validate_terminal(props, invocation):
    fields(props, {"LoadState": "loaded", "ActiveState": "active", "SubState": "exited",
                   "Result": "success", "ExecMainCode": "1", "ExecMainStatus": "0",
                   "MainPID": "0", "InvocationID": invocation}, "retained terminal service required")


def validate_progress(text, k, inventory):
    events = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    ordered = inventory["ordered_identities"]
    keys = [(r["dataset_index"], r["case"], r["start"]) for r in ordered]
    index = {key: i for i, key in enumerate(sorted(keys))}
    require(len(index) == 1368, "inventory uniqueness")
    windows = [r for r in events if r.get("event") == "training_window_complete"]
    updates = [r for r in events if r.get("event") == "accumulation_update_complete"]
    require(len(windows) == 1368 and len(updates) == 171, "actual log progress incomplete")
    progress = [r["event"] for r in events if r.get("event") in
                ("training_window_complete", "accumulation_update_complete")]
    equal(progress, (["training_window_complete"]*8+["accumulation_update_complete"])*171,
          "actual eight-window update ordering")
    for consumed, (row, key) in enumerate(zip(windows, keys, strict=True), 1):
        fields(row, dict(history_k=k, consumed=consumed, global_index=index[key]), "actual log window order")
    for update, row in enumerate(updates, 1):
        fields(row, dict(history_k=k, update=update), "actual log optimizer update")


def terminal_properties(unit):
    keys = ("LoadState", "ActiveState", "SubState", "Result", "ExecMainCode",
            "ExecMainStatus", "MainPID", "InvocationID", "ExecStart")
    text = subprocess.check_output(["systemctl", "--user", "show", unit,
        *["--property=" + key for key in keys]], text=True, timeout=15)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def validate_history(row, k):
    ident = row["identity"]
    fields(ident, {"split": "train", "rollout_steps": 100}, "window identity")
    start = ident["start"]
    require(type(start) is int and start >= 0 and type(ident["dataset_index"]) is int and
            ident["dataset_index"] in (0, 1, 2), "window start/family")
    indices = list(range(start-k+1, start+1))
    equal(row["history"], dict(frame_indices=[max(0, i) for i in indices],
          padding_mask=[i < 0 for i in indices], full_observed_history=all(i >= 0 for i in indices)), "causal history/padding differs")
    value = row.get("flow_history_sha256")
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "flow history digest")


def validate_objective(row):
    fields(row, {"chunk_size": 10, "chunks": 10}, "mixed20 chunk schedule")
    for domain in ("h1", "ar"):
        channels = row[domain + "_channel_mse"]
        require(isinstance(channels, list) and len(channels) == 4, "four force channels")
        for value in channels:
            number(value, "channel MSE")
        require(math.isclose(number(row[domain+"_balanced"], "balanced objective"),
                            .125*sum(channels)+.5*channels[3], rel_tol=2e-6, abs_tol=1e-10), "balanced objective differs")
    require(math.isclose(number(row["total"], "combined objective"),
                        .5*(row["h1_balanced"]+row["ar_balanced"]), rel_tol=2e-6, abs_tol=1e-10), "H1/AR mixing differs")


def validate_records(records, k, inventory):
    require(isinstance(records, list) and len(records) == 171, "171 update records required")
    rows = []
    for update, group in enumerate(records, 1):
        fields(group, dict(update=update, consumed_windows=update*8, windows=8, optimizer_steps=1), "update boundary")
        norm = number(group["preclip_mean_gradient_norm"], "gradient norm")
        require(group["applied_clip_scale"] == min(1., 1./(norm+1e-6)), "clip scale")
        for key in ("gradient_audit", "parameter_update", "cumulative_displacement"):
            values = group[key]
            require(set(values) == {"new_history_l2", "inherited_l2"}, "partition norm keys")
            for value in values.values():
                number(value, "partition norm")
            require(k == 4 or values["new_history_l2"] == 0, "K1 cannot update history coefficients")
        batch = group["records"]
        require(isinstance(batch, list) and len(batch) == 8, "eight windows per update")
        for row in batch:
            validate_history(row, k)
            validate_objective(row)
        equal(group["mean_objective"], {key: sum(row[key] for row in batch)/8
              for key in ("h1_balanced", "ar_balanced", "total")}, "update objective average")
        rows.extend(batch)
    actual = [{key: row["identity"][key] for key in ("dataset_index", "case", "start", "split")} for row in rows]
    expected = [{key: row[key] for key in ("dataset_index", "case", "start", "split")} for row in inventory["ordered_identities"]]
    equal(actual, expected, "actual consumed order differs from independently enumerated inventory")
    return rows


def validate_tensor_pair(parent, terminal, k):
    import torch
    require(set(parent) == set(terminal), "official state tensor names differ")
    initial = {name: value.clone() for name, value in parent.items()}
    if k == 4:
        old = parent[LIFT]
        require(tuple(old.shape) == (24, 8, 1, 1), "parent lifting shape")
        mapped = old.new_zeros((24, 20, 1, 1))
        mapped[:, 9:13] = old[:, :4]
        mapped[:, 16:20] = old[:, 4:8]
        initial[LIFT] = mapped
    changed = []
    for name, before in initial.items():
        after = terminal[name]
        require(torch.is_tensor(after) and before.shape == after.shape and before.dtype == after.dtype
                and bool(torch.isfinite(after).all()), "tensor shape/dtype/finiteness")
        if not torch.equal(before, after):
            require(name not in FROZEN and before.numel() > 0, "frozen tensor changed")
            changed.append(name)
    require(changed, "no aerodynamic tensor update")
    return initial, changed


def validate_optimizer(state, terminal):
    # Same fresh AdamW numerical protocol as P018; independent P026 metadata below.
    from audit_fcp018_candidate import validate_optimizer as validate_adam
    validate_adam(state)
    parameters = [tensor for name, tensor in terminal.items() if name not in FROZEN and tensor.numel()]
    require(len(parameters) == 28, "28 persisted trainable tensors")
    optimizer = state["optimizer_state_dict"]
    ids = optimizer["param_groups"][0]["params"]
    for pid, parameter in zip(ids, parameters, strict=True):
        for name in ("exp_avg", "exp_avg_sq"):
            moment = optimizer["state"][pid][name]
            require(moment.shape == parameter.shape and moment.dtype == parameter.dtype,
                    "Adam moments do not match actual parameter order/shapes")


def validate_panels(result, trainer, p020, inventory, k):
    panels = result["fixed_train_panels"]
    require(len(panels) == 4, "four fixed diagnostic snapshots")
    panel_ids = [160, 816, 923, 975, 1077, 1233]
    identity_by_global = sorted(inventory["ordered_identities"], key=lambda row: (row["dataset_index"], row["case"], row["start"]))
    reference_flow = None
    for panel, consumed in zip(panels, (0, 456, 912, 1368), strict=True):
        require(type(panel["consumed_windows"]) is int and panel["consumed_windows"] == consumed, "diagnostic schedule")
        validate_panel(panel, trainer, p020, identity_by_global, k, panel_ids)
        flows = [row["flow_history_sha256"] for row in panel["rows"]]
        if reference_flow is None:
            reference_flow = flows
        equal(flows, reference_flow, "frozen flow diagnostic history changed")
    ablation = result["terminal_history_zero_ablation"]
    if k == 1:
        require(ablation is None, "K1 has no history-block ablation")
    else:
        validate_panel(ablation, trainer, p020, identity_by_global, k, panel_ids)
        equal([row["flow_history_sha256"] for row in ablation["rows"]], reference_flow, "ablation flow changed")


def validate_panel(panel, trainer, p020, identities, k, indices):
    rows = panel["rows"]
    require([row["global_index"] for row in rows] == indices, "fixed six panel ordering")
    for row in rows:
        validate_history(row, k)
        fields(row["identity"], {key: identities[row["global_index"]][key]
               for key in ("dataset_index", "case", "start", "split")}, "fixed panel identity")
        validate_objective(row["panel"]["objective"])
        for domain in ("h1", "ar"):
            values = row["panel"]["domains"][domain]
            for key in ("bias_mse", "rms_error_mse", "absolute_rms_error", "centered_residual_mse", "predicted_tail_rms", "truth_tail_rms"):
                number(values[key], key)
            require(math.isclose(values["bias_mse"], values["signed_mean_error"]**2, rel_tol=1e-10, abs_tol=1e-14), "bias statistic algebra")
            require(math.isclose(values["rms_error_mse"], (values["predicted_tail_rms"]-values["truth_tail_rms"])**2, rel_tol=1e-9, abs_tol=1e-14), "RMS statistic algebra")
    equal(panel["aggregate"], p020.aggregate(rows), "panel aggregate recomputation")
    equal(panel["history_subgroups"], trainer.grouped_panel(rows), "warm/padded aggregate recomputation")


def validate_execution(repo, root, k, approval_sha, observation_path, observation_sha, unit, invocation):
    from audit_fcp013_dual_candidate import validate_guard
    from audit_fcp015_candidate import validate_resource_watch
    approval_path = root / "execution_approval.json"
    checked(approval_path, approval_sha)
    checked(observation_path, observation_sha)
    approval, observation = read(approval_path), read(observation_path)
    fields(approval, {"status": "FC_P026_APPROVED_MATCHED_HISTORY_TRAINING_NOT_ADMISSION", "history_k": k, "image": IMAGE}, "training approval")
    fields(observation, {"status": f"FC_P026_K{k}_TRAINING_RUNNING_NOT_ADMISSION", "unit": unit,
           "invocation": invocation, "approval_sha256": approval_sha, "image": IMAGE,
           "observed_container_state": "running", "planned_windows": 1368,
           "planned_optimizer_updates": 171, "training_completed": False, "scientific_admission": False}, "external running observation")
    require(unit == f"fluid-control-fcp026-history-k{k}-20261005.service" and len(invocation) == 32, "exact arm unit/invocation")
    props = terminal_properties(unit)
    validate_terminal(props, invocation)  # Before archive/HDF reads, especially while still running.
    frozen = repo / "artifacts/fcp026_history_training_source_20261005_immutable"
    command = f"/bin/bash {frozen}/scripts/run_fcp026_history_training_spark.sh --execute {k}"
    require("argv[]="+command+" ;" in props["ExecStart"], "actual terminal launcher command")
    for base, mapping in ((frozen, approval["source_sha256"]), (repo, approval["dependency_sha256"])):
        for name, digest in mapping.items():
            path = base/name
            require(path.resolve().is_relative_to(base.resolve()), "source path confinement")
            checked(path, digest)
    checked(root/"immutable_launcher.sh", approval["launcher_sha256"])
    checked(frozen/"scripts/train_fcp026_history.py", TRAINER_SHA)
    for marker in ("resource_violation", "watcher_failure"):
        require(not (root/marker).exists(), "external watchdog failure")
    require((root/"container_exit_code").read_text().strip() == "0", "container pipeline did not succeed")
    runtime = read(root/"runtime_container.json")
    require(runtime["Id"] == observation["container_id"] and runtime["Image"] == IMAGE, "actual container identity")
    require(runtime["Config"]["Labels"].get("fc.p026.approval") == approval_sha, "actual approval label")
    fields(runtime["HostConfig"], {"ReadonlyRootfs": True, "NetworkMode": "none", "Memory": 12*1024**3}, "actual container restrictions")
    require(runtime["Config"]["User"] == "1000:1000", "container user")
    cmd = runtime["Config"]["Cmd"]
    for flag, value in (("--history-k", str(k)), ("--flow-parent", "/workspace/candidate/flow"),
                        ("--aerodynamic-parent", "/workspace/candidate/aerodynamic"),
                        ("--output", "/workspace/output/candidate")):
        require(cmd.count(flag) == 1 and cmd[cmd.index(flag)+1] == value, "actual trainer argument "+flag)
    require(cmd[-1] == "--execute" and "/workspace/probe/train_fcp026_history.py" in cmd, "actual trainer execute command")
    expected_cmd = ["python", "-u", "scripts/spark_gpu_guard.py", "--min-free-gib", "20",
        "--allocator-fraction", ".06", "--margin-gib", "4", "--poll-seconds", "2", "--",
        "timeout", "-k", "20", "14400", "python", "-u", "/workspace/probe/train_fcp026_history.py"]
    options = [("source-root", "/workspace/project"),
        ("diagnostic-script", "/workspace/helpers/diagnose_fcp014_train_objective.py"),
        ("gradient-helper", "/workspace/gradient_helper.py"),
        ("history-module", "/workspace/probe/p026_state_history.py"),
        ("history-objective", "/workspace/probe/p026_history_objective.py"),
        ("config", "/workspace/config.yaml"),
        ("history-inference-module", "/workspace/probe/p026_history_inference.py"),
        ("resource-helper", "/workspace/probe/probe_fcp026_history_resource.py"),
        ("accumulation-helper", "/workspace/accumulation_helper.py"),
        ("comparison-helper", "/workspace/comparison_helper.py"),
        ("flow-parent", "/workspace/candidate/flow"),
        ("aerodynamic-parent", "/workspace/candidate/aerodynamic"),
        ("parent-manifest", "/workspace/candidate/dual_model_manifest.json"),
        ("candidate-audit", "/workspace/candidate_audit.json"),
        ("history-k", str(k)), ("output", "/workspace/output/candidate")]
    for flag, value in options:
        expected_cmd.extend(["--"+flag, value])
    equal(cmd, expected_cmd+["--execute"], "exact actual approved trainer/guard command")
    mounts = runtime["Mounts"]
    require([m["Destination"] for m in mounts if m["RW"]] == ["/workspace/output"], "unexpected writable bind mount")
    require(any(m["Destination"] == "/workspace/output" and m["Source"] == str(root) for m in mounts), "actual output bind")
    expected_mounts = {
        "/workspace/project": repo/"artifacts/fcp013_training_source_1634c05_immutable",
        "/workspace/probe": frozen/"scripts",
        "/workspace/helpers": repo/"artifacts/fcp014_train_objective_source_20261005_immutable/scripts",
        "/workspace/gradient_helper.py": repo/"artifacts/fcp019_gradient_alignment_source_20261005_immutable/scripts/probe_fcp019_gradient_alignment.py",
        "/workspace/accumulation_helper.py": repo/"artifacts/fcp015_window_accumulation_source_20261005_immutable/scripts/train_fcp015_window_accumulation.py",
        "/workspace/comparison_helper.py": repo/"artifacts/fcp020_symmetric_statistics_source_20261005_immutable/scripts/probe_fcp020_symmetric_statistics.py",
        "/workspace/candidate": repo/"artifacts/fcp018_reduced_rate_training_20261005/candidate",
        "/workspace/candidate_audit.json": repo/"artifacts/fcp018_reduced_rate_training_20261005/candidate_audit.json",
        "/workspace/config.yaml": repo/"artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml",
        "/workspace/output": root,
    }
    for alias, family in (("base", "tandem_cylinders_matched_start_full40_dev30_v1"),
                          ("train8", "tandem_cylinders_dynamic_train8_v1"),
                          ("train16", "tandem_cylinders_directppo_train16_v1")):
        expected_mounts["/workspace/"+alias] = repo/"artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view"/alias
        expected_mounts["/workspace/"+alias+"/train"] = repo/"data/curated"/family/"train"
    equal({m["Destination"]: m["Source"] for m in mounts},
          {key: str(value) for key, value in expected_mounts.items()}, "actual training-only mount set")
    require(len(mounts) == len(expected_mounts), "duplicate mount destination")
    devices = runtime["HostConfig"]["DeviceRequests"]
    require(len(devices) == 1 and devices[0]["DeviceIDs"] == ["0"] and devices[0]["Capabilities"] == [["gpu"]], "actual GPU0 assignment")
    guard = validate_guard((root/"run.log").read_text())
    number(guard["min_observed_cuda_free_gib"], "CUDA free guard minimum", 20)
    return approval, props, guard, validate_resource_watch((root/"resource_watch.jsonl").read_text())


def validate_candidate(args):
    import torch
    from audit_fcp011_candidate import INPUT_SHA, CONFIG_SHA, NORMALIZATION_SHA, load_model_state
    from audit_fcp013_dual_candidate import state_digest, observed_order_from_hdf
    from audit_fcp015_candidate import require_finite
    import fluid_control.dual_fno as dual
    repo, root, k = args.repo.resolve(), args.root.resolve(), args.history_k
    require(root == repo/f"artifacts/fcp026_history_training_k{k}_20261005", "arm output root")
    approval, terminal, guard, host = validate_execution(repo, root, k, args.approval_sha256,
        args.observation, args.observation_sha256, args.unit, args.invocation)
    frozen = repo/"artifacts/fcp026_history_training_source_20261005_immutable/scripts"
    trainer = load_pinned(frozen/"train_fcp026_history.py", TRAINER_SHA, "p026_audit_trainer")
    checked(Path(dual.__file__), LOADER_SHA)
    candidate = root/"candidate"
    expected_protocol = trainer.protocol(k)
    protocol_sha = trainer.canonical_sha(expected_protocol)
    checked(candidate/"training_protocol.json", protocol_sha)
    equal(read(candidate/"training_protocol.json"), expected_protocol, "effective protocol")
    validate_approved_protocol(approval["effective_protocol"], expected_protocol)
    require(approval["effective_protocol_sha256"] == protocol_sha, "approved protocol SHA")
    inv_path = repo/approval["inventory_receipt"]
    checked(inv_path, approval["inventory_receipt_sha256"])
    inventory = read(inv_path)
    equal(inventory["inventory"], expected_protocol["inventory"], "independent inventory")
    fields(inventory, dict(status="FC_P026_CPU_TRAIN_INVENTORY_ORDER_VERIFIED_NOT_TRAINING_APPROVAL",
           trainer_sha256=TRAINER_SHA, sampler_order_sha256=trainer.ORDER_SHA), "inventory producer identity")
    identity = dual.validate_dual_fno_manifest(candidate/"dual_model_manifest.json")
    require(identity.payload["kind"] == f"FC_P026_K{k}_HISTORY_FORCE_FNO", "candidate history kind")
    result = read(candidate/"result.json")
    require_finite(result)
    fields(result, dict(status=f"FC_P026_K{k}_TRAINING_COMPLETE_NOT_ADMISSION", history_k=k,
        protocol_sha256=protocol_sha, history_inventory=expected_protocol["inventory"],
        sampler_order_sha256=trainer.ORDER_SHA, optimizer_steps=171, training_windows=1368,
        official_fresh_reload_verified=True, scientific_admission=False, trainer_sha256=TRAINER_SHA,
        replay_rtol=1e-5, replay_atol=1e-6, input_sha256=INPUT_SHA,
        precision=dual.PRECISION_PROTOCOL), "actual terminal result contract")
    expected_sources = dict(history_module=trainer.HISTORY_SHA, history_objective=trainer.OBJECTIVE_SHA,
        history_inference_module=trainer.INFERENCE_SHA, resource_helper=trainer.RESOURCE_SHA,
        accumulation_helper=trainer.P015_SHA, gradient_helper=trainer.P019_SHA,
        comparison_helper=trainer.P020_SHA, diagnostic_script="849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d",
        config=CONFIG_SHA, parent_manifest=trainer.MANIFEST_SHA,
        candidate_audit="03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9")
    equal(result["source_sha256"], expected_sources, "actual numerical source map")
    equal(identity.payload["source_sha256"], expected_sources, "manifest source map")
    rows = validate_records(result["records"], k, inventory)
    validate_progress((root/"run.log").read_text(), k, inventory)
    order, hdf_hashes = observed_order_from_hdf(repo, rows)
    p020 = load_pinned(repo/"artifacts/fcp020_symmetric_statistics_source_20261005_immutable/scripts/probe_fcp020_symmetric_statistics.py", trainer.P020_SHA, "p026_audit_panel")
    validate_panels(result, trainer, p020, inventory, k)
    parent_root = repo/"artifacts/fcp018_reduced_rate_training_20261005/candidate"
    checked(parent_root/"aerodynamic/FNO.0.1.mdlus", dual.P026_AERO_INITIAL_MODEL_SHA256)
    parent = load_model_state(parent_root/"aerodynamic/FNO.0.1.mdlus")
    aero = load_model_state(identity.aerodynamic.model)
    initial, changed = validate_tensor_pair(parent, aero, k)
    flow = load_model_state(identity.flow.model)
    fields(result, dict(flow_tensor_sha256=state_digest(flow),
        aerodynamic_initial_tensor_sha256=state_digest(initial), aerodynamic_terminal_tensor_sha256=state_digest(aero)), "persisted model tensors")
    state = torch.load(identity.aerodynamic.state, map_location="cpu", weights_only=True)
    validate_optimizer(state, aero)
    metadata = dict(status=f"FC_P026_K{k}_HISTORY_AERODYNAMIC_CHECKPOINT", checkpoint_epoch=1,
        training_experiment="FC-P026", history_profile=f"p026_k{k}", history_k=k,
        model_in_channels=6 if k == 1 else 18, training_protocol_sha256=protocol_sha,
        training_protocol_file="training_protocol.json", history_state_module_sha256=trainer.HISTORY_SHA,
        history_inference_module_sha256=trainer.INFERENCE_SHA, training_windows=1368,
        optimizer_steps=171, accumulation_windows=8, actual_learning_rate=1.5625e-7,
        sampler_order_sha256=trainer.ORDER_SHA, history_inventory=expected_protocol["inventory"],
        selection_performed=False, validation_accessed=False, frozen_test_accessed=False, ppo_executed=False,
        flow_parent_model_sha256=identity.flow.model_sha256, flow_parent_state_sha256=identity.flow.state_sha256,
        aerodynamic_initial_model_sha256=dual.P026_AERO_INITIAL_MODEL_SHA256,
        aerodynamic_initial_state_sha256=dual.P026_AERO_INITIAL_STATE_SHA256)
    equal(state["metadata"], metadata, "official terminal metadata")
    samples = result["resources"]
    require(isinstance(samples, list) and samples, "internal resource samples missing")
    number(samples[0]["MemFree"], "startup MemFree", 30)
    number(samples[0]["MemAvailable"], "startup MemAvailable", 50)
    previous = -1
    for sample in samples:
        number(sample["MemFree"], "internal MemFree", 20)
        number(sample["MemAvailable"], "internal MemAvailable", 20)
        elapsed = number(sample["elapsed_seconds"], "internal elapsed")
        require(previous <= elapsed <= 14400, "internal deadline/order")
        previous = elapsed
    for domain in ("h1", "ar"):
        number(result["replay_max_abs"][domain], "initial replay difference")
    files = candidate_files(root)
    validate_terminal(terminal_properties(args.unit), args.invocation)
    return dict(status=f"FC_P026_K{k}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION", history_k=k,
        training_protocol_sha256=protocol_sha, dual_manifest_sha256=identity.manifest_sha256,
        candidate_result_sha256=files["candidate/result.json"], candidate_sha256=files,
        training_unit=args.unit, training_invocation=args.invocation, terminal_evidence=terminal,
        training_approval_sha256=args.approval_sha256, execution_observation_sha256=args.observation_sha256,
        actual_optimizer_steps=171, actual_training_windows=1368, accumulation_windows=8,
        train_order_sha256=order, train_hdf_sha256=hdf_hashes,
        tensor_sha256=dict(flow=state_digest(flow), aerodynamic=state_digest(aero)),
        changed_aerodynamic_tensor_names=changed, guard=guard, resource_watch=host,
        internal_resources=dict(samples=len(samples), min_mem_free_gib=min(r["MemFree"] for r in samples),
                                min_mem_available_gib=min(r["MemAvailable"] for r in samples)),
        dual_adapter_fresh_reload_verified=False, scientific_admission=False, ppo_authorized=False,
        auditor_sha256=sha(__file__), role_loader_sha256=sha(dual.__file__),
        execution_sha256={name: sha(root/name) for name in ("execution_approval.json", "immutable_launcher.sh",
             "runtime_container.json", "container_exit_code", "run.log", "resource_watch.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "root", "observation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("approval-sha256", "observation-sha256", "unit", "invocation"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--history-k", type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output already exists")
    result = validate_candidate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
