"""P029 project orchestration; inert until separately approved per-mode execution.

Official FNO/DataPipe/checkpoint APIs and reviewed P028 optimizer safeguards are
reused explicitly. No old module globals or numerical functions are patched.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

BASE_SHA = "4ad0dfa41fcfeb93481e313d2f0ca24685ea3ea183a074a7b017284f8c5ee057"
PARENT_SHA = "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
AUDIT_SHA = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
MODES = ("scales", "resource-probe", "train")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(record):
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError("file identity differs: " + str(path))
    return path


def protocol():
    return dict(experiment="FC-P029", optimized_role="flow", fixed_role="aerodynamic",
                horizon=10, original_window_horizon=100, training_windows=1368,
                optimizer_steps=171, accumulation_windows=8, seed=20261003,
                learning_rate=1e-5, betas=[.9, .999], eps=1e-8, weight_decay=1e-4,
                gradient_clip_norm=1.0, sampler_order_sha256=ORDER_SHA,
                objective="half_parent_scaled_field_MSE_plus_half_parent_scaled_four_force_MSE",
                field_weight=.5, force_weight=.5, scale_windows=1368, force_loss=True,
                force_timing="aero_current_state_and_current_next_action_predicts_next_force",
                terminal_selection=False, future_truth_inputs=False)


def validate_spec(spec, mode):
    if (mode not in MODES or spec.get("status") != "FC_P029_EXECUTION_APPROVED"
            or spec.get("mode") != mode or spec.get("protocol") != protocol()):
        raise ValueError("separate P029 mode/protocol approval required")
    for key, digest in (("parent_manifest", PARENT_SHA), ("config", CONFIG_SHA),
                        ("train_audit", AUDIT_SHA)):
        if spec[key]["sha256"] != digest:
            raise ValueError("fixed " + key + " differs")
    fraction = spec["resources"]["allocator_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= .45:
        raise ValueError("explicit bounded allocator required")
    if mode != "scales" and not spec.get("scales_receipt"):
        raise ValueError("actual fixed scales receipt required")
    if mode == "train" and not spec.get("resource_probe_receipt"):
        raise ValueError("actual resource receipt required")


def hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_scales(values):
    if not isinstance(values, dict) or set(values) != {"field", "force"}:
        raise ValueError("exact two fixed scales required")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("finite positive fixed scales required")
    return dict(values)


def summarize_scales(records):
    if len(records) != 1368:
        raise ValueError("all1368 parent windows required")
    values = {}
    for role in ("field", "force"):
        raw = [r["raw_" + role] for r in records]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in raw):
            raise ValueError("invalid raw parent objective")
        values[role] = math.fsum(raw) / 1368
    return validate_scales(values)


def binding(spec):
    keys = ("parent_manifest", "config", "train_audit", "source_sha256")
    if not isinstance(spec, dict) or any(key not in spec for key in keys):
        raise ValueError("receipt source binding missing")
    return {key: spec[key] for key in keys}


def validate_receipt(receipt, spec, mode):
    expected_status = {"scales": "FC_P029_PARENT_SCALES_COMPLETE_NOT_ADMISSION",
                       "resource-probe": "FC_P029_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION"}[mode]
    initial = receipt.get("flow_initial_tensor_sha256")
    if (receipt.get("status") != expected_status or receipt.get("mode") != mode
            or receipt.get("protocol") != protocol()
            or receipt.get("optimizer_steps") != 0
            or receipt.get("optimizer_created") is not False
            or receipt.get("model_saved") is not False
            or receipt.get("scientific_admission") is not False
            or not hex_digest(initial) or initial != receipt.get("flow_terminal_tensor_sha256")
            or not hex_digest(receipt.get("frozen_aerodynamic_tensor_sha256"))
            or binding(receipt.get("source_spec", {})) != binding(spec)):
        raise ValueError("actual immutable parent receipt differs")
    if mode == "scales":
        if (receipt.get("training_windows") != 1368
                or receipt.get("sampler_order_sha256") != ORDER_SHA
                or receipt.get("fixed_scales") != summarize_scales(receipt.get("records", []))):
            raise ValueError("full parent scales/order/recomputation differs")
    elif (receipt.get("training_windows") != 1
          or receipt.get("source_spec", {}).get("scales_receipt") != spec["scales_receipt"]
          or receipt.get("fixed_scales") != spec["fixed_scales"]):
        raise ValueError("resource proof scales/window differs")
    return validate_scales(receipt["fixed_scales"])


def load_module(name, path):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    sys.modules[name] = module
    try:
        definition.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def dependencies(spec):
    root = Path(spec["source_root"])
    sources = spec["source_sha256"]
    for name in ("train_p029_control_aware_flow", "p029_control_aware_flow_objective"):
        relative = "scripts/" + name + ".py"
        checked(dict(path=str(root / relative), sha256=sources[relative]))
    if sha(__file__) != sources["scripts/train_p029_control_aware_flow.py"]:
        raise ValueError("P029 runner self identity differs")
    base_path = checked(dict(path=str(root / "scripts/train_p028_flow_rollout.py"), sha256=BASE_SHA))
    base = load_module("p029_pinned_p028_runner", base_path)
    identity, history, official, accumulation, inputs = base.dependencies(spec)
    objective_path = root / "scripts/p029_control_aware_flow_objective.py"
    objective = load_module("p029_pinned_objective", objective_path)
    return base, identity, history, official, accumulation, inputs, objective


def resource_limits(memory, cuda_free, elapsed, mode, startup=False):
    if elapsed > (14400 if mode == "train" else 900):
        raise RuntimeError("whole mode deadline")
    if (memory.get("MemFree", 0) < (30 if startup else 20)
            or memory.get("MemAvailable", 0) < (50 if startup else 20)
            or cuda_free < 20):
        raise RuntimeError("physical/CUDA memory floor")


def execute(spec, mode, output):
    import torch
    import numpy as np
    validate_spec(spec, mode)
    if output.exists():
        raise FileExistsError(output)
    config = checked(spec["config"])
    parent_path = checked(spec["parent_manifest"])
    audit = json.loads(checked(spec["train_audit"]).read_text())
    scales = {"field": 1., "force": 1.}
    receipts = []
    if mode != "scales":
        receipt = json.loads(checked(spec["scales_receipt"]).read_text())
        receipts.append(receipt)
        scales = validate_receipt(receipt, spec, "scales")
        if scales != validate_scales(spec["fixed_scales"]):
            raise ValueError("approved scales differ from actual parent receipt")
    if mode == "train":
        receipt = json.loads(checked(spec["resource_probe_receipt"]).read_text())
        validate_receipt(receipt, spec, "resource-probe")
        receipts.append(receipt)
    base, identity_helper, history, official, accumulation, inputs, objective = dependencies(spec)
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from fluid_control.dual_fno import load_dual_fno, validate_runtime_precision
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.augmented_datapipe import compose_training_data

    started, resources = time.monotonic(), []
    def guard(startup=False):
        memory = {line.split(":")[0]: int(line.split()[1]) / 2**20
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemFree:", "MemAvailable:"))}
        free = torch.cuda.mem_get_info()[0] / 2**30
        resource_limits(memory, free, time.monotonic() - started, mode, startup)
        resources.append(dict(elapsed=time.monotonic() - started, cuda_free_gib=free, **memory))

    cfg = OmegaConf.load(config)
    identity_helper.validate_data_contract(cfg)
    if any(sha(inspect.getfile(f)) != base.CHECKPOINT_SHA for f in (save_checkpoint, load_checkpoint)):
        raise ValueError("official checkpoint source differs")
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count() != 1:
        raise RuntimeError("one approved GPU required")
    guard(True)
    torch.cuda.set_per_process_memory_fraction(spec["resources"]["allocator_fraction"], dist.device)
    precision = base.configure_precision(torch, validate_runtime_precision,
                                        json.loads(parent_path.read_text())["precision_protocol"])
    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)
    adapter, identity = load_dual_fno(parent_path, cfg, "cpu", build_model=official.build_model,
                                    expected_manifest_sha256=PARENT_SHA)
    if identity.payload["kind"] != "FC_P026_K1_HISTORY_FORCE_FNO":
        raise ValueError("same original P009 flow/P026 K1 aero parent required")
    flow, aero = adapter.flow_model, adapter.aerodynamic_model
    if len(list(flow.parameters())) != 30 or sha(inspect.getfile(type(flow))) != base.FNO_SHA:
        raise ValueError("official full30 flow parameter scope required")
    flow_initial_sha = identity_helper.tensor_state_sha256(flow)
    aero_sha = identity_helper.tensor_state_sha256(aero)
    if any(r["flow_initial_tensor_sha256"] != flow_initial_sha
           or r["frozen_aerodynamic_tensor_sha256"] != aero_sha for r in receipts):
        raise ValueError("actual loaded parent tensors differ from preparatory receipts")
    flow.to(dist.device).train().requires_grad_(True)
    aero.to(dist.device).eval().requires_grad_(False)
    saved = base.preserve_rows(flow)
    frozen_proof = lambda: base.assert_frozen(aero, aero_sha, identity_helper.tensor_state_sha256)
    aero_versions = [(v, v._version) for v in (*aero.parameters(), *aero.buffers())]
    dataset = TandemRolloutDataset(cfg.data.root, "train", 100, stride=20,
                                  num_workers=cfg.training.workers, force_indices=(0, 1, 2, 3))
    train = None
    try:
        train, _ = compose_training_data(dataset, [Path(x) for x in cfg.data.additional_train_roots],
                                        rollout_steps=100, stride=2, workers=cfg.training.workers,
                                        force_indices=(0, 1, 2, 3))
        inventory = history.inventory(train)
        indices = identity_helper.identity_index(train)
        actual = {}
        for child in train._datasets:
            for path in child.paths:
                if path.is_symlink() or path.parent.name != "train" or path.name in actual:
                    raise ValueError("unique train-only regular HDF required")
                actual[path.name] = sha(path)
        expected_files = {Path(k).name: v for k, v in audit["train_hdf_sha256"].items()}
        if len(actual) != 44 or actual != expected_files:
            raise ValueError("exact44 training bytes differ")
        sampler = DataLoader(train, batch_size=1, shuffle=True, prefetch_factor=0,
                             use_streams=False, seed=20261003)
        expected = list(iter(sampler.sampler))
        if len(expected) != 1368 or identity_helper.sequence_sha(expected) != ORDER_SHA:
            raise ValueError("original1368 sampler order differs")
        loader = DataLoader(train, batch_size=1, shuffle=True, collate_metadata=True,
                            prefetch_factor=int(cfg.data.prefetch_factor), num_streams=int(cfg.data.num_streams),
                            use_streams=True, seed=20261003)
        def run(batch, ident):
            guard()
            x = {key: value.to(dist.device) for key, value in batch.items()}
            def predict(*args):
                guard()
                return official.predict(*args)
            with torch.set_grad_enabled(mode != "scales"):
                value = objective.field_force_rollout_objective(
                    flow, aero, x["state"], x["target_state"], x["mask"], x["omega"],
                    x["target_force"], predict, inputs.make_inputs,
                    field_scale=scales["field"], force_scale=scales["force"], backward=mode != "scales")
            value["identity"] = ident
            print(json.dumps(dict(event="window_complete", mode=mode, **ident)), flush=True)
            return value
        records, observed, gradients, optimizer = [], [], None, None
        if mode == "resource-probe":
            sample, metadata = train[816]
            ident = identity_helper.training_identity([metadata])
            flow.zero_grad(set_to_none=True)
            records.append(run({key: value[None] for key, value in sample.items()}, ident))
            base.mask_rows(flow, None, saved)
            gradients = base.finite_gradients(flow)
            observed = [816]
        else:
            iterator = iter(loader)
            if mode == "train":
                optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
                base.check_optimizer(flow, optimizer, 0)
            def windows():
                for _ in range(8):
                    batch, metadata = next(iterator)
                    ident = identity_helper.training_identity(metadata)
                    observed.append(indices[(ident["case"], ident["start"], ident["dataset_index"])])
                    yield batch, ident
            for step in range(1, 172):
                if mode == "scales":
                    records.extend(run(*item) for item in windows())
                else:
                    record = accumulation.accumulate_eight_window_gradients(flow, optimizer, windows(), lambda item: run(*item))
                    for key in ("raw_field", "raw_force", "normalized_field_contribution", "normalized_force_contribution"):
                        record["mean_" + key] = math.fsum(row[key] for row in record["records"]) / 8
                    record.update(base.optimizer_update(flow, optimizer, saved, step))
                    record.update(update=step, consumed_windows=len(observed))
                    records.append(record)
                frozen_proof()
                guard()
                print(json.dumps(dict(event="group_complete", mode=mode, group=step)), flush=True)
            if observed != expected or next(iterator, None) is not None:
                raise RuntimeError("observed original order differs")
        frozen_proof()
        if any(v._version != version for v, version in aero_versions):
            raise RuntimeError("frozen aerodynamic version changed")
        flow_terminal_sha = identity_helper.tensor_state_sha256(flow)
        if mode != "train" and flow_terminal_sha != flow_initial_sha:
            raise RuntimeError("no-update parent tensors changed")
        if mode == "scales":
            if any(p.grad is not None for p in flow.parameters()):
                raise RuntimeError("scales pass created a gradient")
            scales = summarize_scales(records)
        guard()
        output.mkdir()
        status = {"scales": "PARENT_SCALES_COMPLETE", "resource-probe": "RESOURCE_PROBE_COMPLETE",
                  "train": "TRAINING_COMPLETE"}[mode]
        result = dict(status="FC_P029_" + status + "_NOT_ADMISSION", mode=mode,
                      protocol=protocol(), fixed_scales=scales, training_windows=len(observed),
                      optimizer_steps=171 if optimizer is not None else 0, optimizer_created=optimizer is not None,
                      model_saved=False, records=records, probe_gradient_norms=gradients,
                      flow_initial_tensor_sha256=flow_initial_sha, flow_terminal_tensor_sha256=flow_terminal_sha,
                      frozen_aerodynamic_tensor_sha256=aero_sha, sampler_order_sha256=identity_helper.sequence_sha(observed),
                      inventory=inventory, resources=resources, precision=precision, source_spec=spec,
                      scientific_admission=False, validation_accessed=False, frozen_test_accessed=False,
                      unused_force_rows_preserved=True, unused_force_outputs_preserved=False,
                      optimizer_memory_accounting=base.optimizer_memory_estimate(flow),
                      cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
        if mode == "train":
            result["optimizer_memory_accounting"].update(optimizer_created=True, optimizer_peak_measured=True)
            save_terminal(output, flow, aero, optimizer, cfg, identity, result, official.build_model,
                          save_checkpoint, load_checkpoint, identity_helper.tensor_state_sha256)
            result["model_saved"] = True
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    finally:
        frozen_proof()
        (train if train is not None else dataset).close()


def save_terminal(output, flow, aero, optimizer, cfg, identity, result, build, save, load, tensor_sha):
    """Explicit P029 trained-flow identity; old P028 admission is not inherited."""
    effective = dict(protocol(), fixed_scales=result["fixed_scales"],
                     scales_receipt_sha256=result["source_spec"]["scales_receipt"]["sha256"])
    (output / "training_protocol.json").write_text(json.dumps(effective, sort_keys=True, separators=(",", ":")))
    protocol_sha = sha(output / "training_protocol.json")
    metadata = dict(status="FC_P029_CONTROL_AWARE_FLOW_CHECKPOINT", training_experiment="FC-P029",
                    checkpoint_epoch=1, training_protocol_sha256=protocol_sha,
                    training_protocol_file="training_protocol.json", accumulation_windows=8,
                    training_windows=1368, optimizer_steps=171, actual_learning_rate=1e-5,
                    parent_manifest_sha256=PARENT_SHA, fixed_scales=result["fixed_scales"],
                    scales_receipt_sha256=effective["scales_receipt_sha256"],
                    source_sha256=result["source_spec"]["source_sha256"])
    for role_name in ("flow", "aerodynamic"):
        for kind in ("model", "state"):
            metadata[role_name + "_parent_" + kind + "_sha256"] = identity.payload[role_name][kind + "_sha256"]
    save(output / "flow", models=flow, optimizer=optimizer, epoch=1, metadata=metadata)
    flow.cpu()
    fresh = build(cfg)
    actual_meta = {}
    if (load(output / "flow", models=fresh, metadata_dict=actual_meta, device="cpu") != 1
            or actual_meta != metadata or tensor_sha(fresh) != result["flow_terminal_tensor_sha256"]):
        raise RuntimeError("official flow fresh reload differs")
    del fresh
    aero.cpu()
    (output / "aerodynamic").mkdir()
    for name in (identity.payload["aerodynamic"]["model_file"], identity.payload["aerodynamic"]["state_file"]):
        shutil.copy2(identity.aerodynamic.directory / name, output / "aerodynamic" / name)
        if sha(output / "aerodynamic" / name) != sha(identity.aerodynamic.directory / name):
            raise RuntimeError("frozen aerodynamic copy differs")
    fresh = build(cfg)
    if load(output / "aerodynamic", models=fresh, metadata_dict={}, device="cpu") != 1 or tensor_sha(fresh) != tensor_sha(aero):
        raise RuntimeError("official frozen aerodynamic fresh reload differs")
    inherited = ("schema_version", "architecture", "flow_architecture", "aerodynamic_architecture",
                 "config_sha256", "normalization_sha256", "precision_protocol", "input_sha256",
                 "history_input", "history_inventory", "history_state_module_sha256",
                 "history_inference_module_sha256", "aerodynamic_initial_model_sha256", "aerodynamic_initial_state_sha256")
    manifest = {key: copy.deepcopy(identity.payload[key]) for key in inherited}
    manifest.update(status="FC_P029_DUAL_FNO_MANIFEST_VERIFIED", kind="FC_P029_CONTROL_AWARE_FLOW_REPAIR",
                    training_experiment="FC-P029", parent_manifest_sha256=PARENT_SHA,
                    training_protocol_file="training_protocol.json", training_protocol_sha256=protocol_sha,
                    training_windows=1368, optimizer_steps=171, accumulation_windows=8,
                    actual_learning_rate=1e-5, training_semantics=effective,
                    fixed_scales=result["fixed_scales"], scales_receipt_sha256=effective["scales_receipt_sha256"],
                    config_role="base_architecture_data_only_effective_P029_protocol", base_config_sha256=CONFIG_SHA,
                    scientific_admission=False, source_sha256=result["source_spec"]["source_sha256"],
                    checkpoint_sha256={str(p.relative_to(output)): sha(p) for role in ("flow", "aerodynamic") for p in sorted((output / role).iterdir())})
    for role_name in ("flow", "aerodynamic"):
        record = copy.deepcopy(identity.payload[role_name])
        record.update(frozen=role_name == "aerodynamic", checkpoint_epoch=1)
        if role_name == "flow":
            record.update(model_file="FNO.0.1.mdlus", state_file="checkpoint.0.1.pt", metadata_kind=metadata["status"])
        for kind in ("model", "state"):
            record[kind + "_sha256"] = sha(output / role_name / record[kind + "_file"])
            manifest[role_name + "_parent_" + kind + "_sha256"] = metadata[role_name + "_parent_" + kind + "_sha256"]
        manifest[role_name] = record
    (output / "dual_model_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    result.update(official_fresh_reload_verified=True, training_protocol_sha256=protocol_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("FC_P029_PREPARATION_ONLY_NO_MODEL_OR_DATA_ACCESS")
        return
    spec = json.loads(checked(dict(path=str(args.spec), sha256=args.spec_sha256)).read_text())
    execute(spec, args.mode, args.output)


if __name__ == "__main__":
    main()
