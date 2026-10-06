"""FC-P028 project orchestration; official FNO, DataPipe and checkpoints.

Preparation is inert. Execute requires a separately SHA-bound approval spec.
The new terminal profile deliberately is NOT accepted by old dual loaders.
"""
import argparse
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import random
import shutil
import sys
import time

PARENT_SHA = "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
AUDIT_SHA = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
FNO_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
FINAL = ("decoder_net.final_layer.linear.weight", "decoder_net.final_layer.linear.bias")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol():
    return dict(experiment="FC-P028", optimized_role="flow", fixed_role="aerodynamic",
                horizon=10, original_window_horizon=100, training_windows=1368,
                optimizer_steps=171, accumulation_windows=8, seed=20261003,
                learning_rate=1e-5, betas=[0.9, 0.999], eps=1e-8,
                weight_decay=1e-4, gradient_clip_norm=1.0,
                objective="ten_equal_masked_normalized_state_MSE",
                sampler_order_sha256=ORDER_SHA, force_loss=False,
                terminal_selection=False, future_truth_inputs=False)


def checked(record):
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError("file SHA/confinement mismatch: " + str(path))
    return path


def validate_spec(spec, mode):
    if mode not in ("resource-probe", "train") or spec.get("status") != "FC_P028_EXECUTION_APPROVED":
        raise ValueError("separate execution approval required")
    if spec.get("mode") != mode or spec.get("protocol") != protocol():
        raise ValueError("mode/protocol differs")
    for key, digest in (("parent_manifest", PARENT_SHA), ("config", CONFIG_SHA), ("train_audit", AUDIT_SHA)):
        if spec[key]["sha256"] != digest:
            raise ValueError(key + " fixed identity differs")
    fraction = spec["resources"]["allocator_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= 0.45:
        raise ValueError("resource-probe-approved allocator required")
    if mode == "train" and not spec.get("resource_probe_receipt"):
        raise ValueError("actual resource receipt required before training")


def resource_limits(memory, cuda_free, elapsed, mode, startup=False):
    if elapsed > (900 if mode == "resource-probe" else 14400):
        raise RuntimeError("wall deadline")
    if (memory.get("MemFree", 0) < (30 if startup else 20)
            or memory.get("MemAvailable", 0) < (50 if startup else 20)
            or cuda_free < 20):
        raise RuntimeError("physical/CUDA memory floor")


def configure_precision(torch_module, validate, parent_precision):
    torch_module.set_float32_matmul_precision("high")
    actual = validate()
    if actual != parent_precision:
        raise ValueError("parent/runtime precision protocol differs")
    return actual


def validate_probe_receipt(probe, spec):
    digest = probe.get("flow_initial_tensor_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or probe.get("status") != "FC_P028_RESOURCE_PROBE_COMPLETE_NOT_ADMISSION"
            or probe.get("mode") != "resource-probe" or probe.get("training_windows") != 1
            or probe.get("protocol") != protocol() or probe.get("optimizer_steps") != 0
            or digest != probe.get("flow_terminal_tensor_sha256")
            or probe.get("source_spec", {}).get("parent_manifest") != spec["parent_manifest"]
            or probe.get("source_spec", {}).get("source_sha256") != spec["source_sha256"]):
        raise ValueError("actual no-update resource-probe proof differs")


def preserve_rows(model):
    params = dict(model.named_parameters())
    if any(name not in params or params[name].shape[0] != 7 for name in FINAL):
        raise ValueError("official seven-channel final layer required")
    return {name: params[name][3:].detach().clone() for name in FINAL}


def optimizer_memory_estimate(model):
    """Accounting only, not an allocation or upper bound on AdamW peak memory."""
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return dict(parameter_bytes=parameter_bytes, gradient_bytes=parameter_bytes,
                adam_first_second_moment_bytes=2 * parameter_bytes,
                scalar_step_count=len(list(model.parameters())),
                optimizer_created=False, optimizer_peak_measured=False,
                warning="No-update probe excludes Adam allocation/foreach temporaries; two moments are a lower bound, not a training-capacity proof")


def mask_rows(model, optimizer, saved, restore=False):
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in saved.items():
            parameter = params[name]
            if parameter.grad is not None:
                parameter.grad[3:].zero_()
            if optimizer is not None:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = optimizer.state.get(parameter, {}).get(key)
                    if value is not None:
                        value[3:].zero_()
            if restore:
                parameter[3:].copy_(original)
            if not torch.equal(parameter[3:], original):
                raise RuntimeError("unused flow-force parameter rows changed")


def finite_gradients(model):
    import torch
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("missing/nonfinite flow gradient: " + name)
        # abs handles complex spectral tensors without discarding imaginary parts.
        rows[name] = float(parameter.grad.detach().abs().double().square().sum().sqrt())
    return rows


def check_optimizer(model, optimizer, step):
    import torch
    params = list(model.parameters())
    if len(optimizer.param_groups) != 1:
        raise ValueError("single optimizer group required")
    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [id(p) for p in params]:
        raise ValueError("optimizer must own only all flow parameters")
    if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]) != (1e-5, (0.9, 0.999), 1e-8, 1e-4):
        raise ValueError("AdamW settings differ")
    if (step == 0 and optimizer.state) or (step and set(optimizer.state) != set(params)):
        raise ValueError("fresh/all flow Adam states required")
    for state in optimizer.state.values():
        if float(state["step"]) != step:
            raise ValueError("Adam step differs")
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
            raise FloatingPointError("nonfinite Adam state")
    if any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError("nonfinite flow parameters")


def optimizer_update(model, optimizer, saved, step):
    import torch
    mask_rows(model, optimizer, saved)
    audit = finite_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    mask_rows(model, optimizer, saved, restore=True)
    check_optimizer(model, optimizer, step)
    return dict(preclip_mean_gradient_norm=float(norm), gradient_norms=audit,
                applied_clip_scale=min(1.0, 1.0 / (float(norm) + 1e-6)))


def assert_frozen(model, expected_sha, tensor_sha):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen aerodynamic scope differs")
    if tensor_sha(model) != expected_sha:
        raise RuntimeError("frozen aerodynamic tensors changed")


def dependencies(spec):
    root = Path(spec["source_root"])
    sources = spec["source_sha256"]
    required = {"scripts/" + name + ".py" for name in (
        "train_p028_flow_rollout", "train_fcp011_decoder_scope", "train_fcp026_history",
        "train_tandem_fno", "p028_flow_h10_objective", "train_fcp013_independent_force_fno",
        "p026_history_inference", "p026_state_history")}
    required.update("src/fluid_control/" + name + ".py" for name in (
        "dual_fno", "calibrated_checkpoint", "tandem_datapipe", "augmented_datapipe"))
    if not required.issubset(sources) or sha(__file__) != sources["scripts/train_p028_flow_rollout.py"]:
        raise ValueError("approved source closure/self identity incomplete")
    for relative, digest in sources.items():
        p = root / relative
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("source escape")
        checked(dict(path=str(p), sha256=digest))
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    names = ("train_fcp011_decoder_scope", "train_fcp026_history", "train_tandem_fno",
             "p028_flow_h10_objective", "train_fcp013_independent_force_fno")
    modules = [importlib.import_module(name) for name in names]
    for module in modules:
        p = Path(module.__file__).resolve()
        rel = str(p.relative_to(root.resolve()))
        if rel not in sources or sha(p) != sources[rel]:
            raise ValueError("unexpected imported source")
    return modules


def execute(spec, mode, output):
    import torch
    import numpy as np
    validate_spec(spec, mode)
    if output.exists():
        raise FileExistsError(output)
    config = checked(spec["config"])
    parent_path = checked(spec["parent_manifest"])
    audit = json.loads(checked(spec["train_audit"]).read_text())
    if mode == "train":
        probe = json.loads(checked(spec["resource_probe_receipt"]).read_text())
        validate_probe_receipt(probe, spec)
    identity_helper, p026, official, objective, p013 = dependencies(spec)
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from fluid_control.dual_fno import load_dual_fno, validate_runtime_precision
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.augmented_datapipe import compose_training_data

    started = time.monotonic()
    resources = []
    def guard(startup=False):
        memory = {line.split(":")[0]: int(line.split()[1]) / 2**20
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemFree:", "MemAvailable:"))}
        free = torch.cuda.mem_get_info()[0] / 2**30
        resource_limits(memory, free, time.monotonic() - started, mode, startup)
        resources.append(dict(elapsed=time.monotonic() - started, cuda_free_gib=free, **memory))

    cfg = OmegaConf.load(config)
    identity_helper.validate_data_contract(cfg)
    if any(sha(inspect.getfile(f)) != CHECKPOINT_SHA for f in (save_checkpoint, load_checkpoint)):
        raise ValueError("official checkpoint source differs")
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count() != 1:
        raise RuntimeError("one approved GPU required")
    guard(True)
    torch.cuda.set_per_process_memory_fraction(spec["resources"]["allocator_fraction"], dist.device)
    precision = configure_precision(torch, validate_runtime_precision,
                                    json.loads(parent_path.read_text())["precision_protocol"])
    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)
    adapter, identity = load_dual_fno(parent_path, cfg, "cpu", build_model=official.build_model,
                                    expected_manifest_sha256=PARENT_SHA)
    if identity.payload["kind"] != "FC_P026_K1_HISTORY_FORCE_FNO":
        raise ValueError("only matched K1 parent")
    flow, aero = adapter.flow_model, adapter.aerodynamic_model
    if len(list(flow.parameters())) != 30:
        raise ValueError("exact official flow30 parameter scope required")
    if sha(inspect.getfile(type(flow))) != FNO_SHA:
        raise ValueError("official FNO source differs")
    aero.eval().requires_grad_(False)
    aero_sha = identity_helper.tensor_state_sha256(aero)
    flow_initial_sha = identity_helper.tensor_state_sha256(flow)
    flow.to(dist.device).train().requires_grad_(True)
    saved = preserve_rows(flow)
    frozen_proof = lambda: assert_frozen(aero, aero_sha, identity_helper.tensor_state_sha256)
    base = TandemRolloutDataset(cfg.data.root, "train", 100, stride=20,
                               num_workers=cfg.training.workers, force_indices=(0, 1, 2, 3))
    train = None
    try:
        train, _ = compose_training_data(base, [Path(x) for x in cfg.data.additional_train_roots],
                                        rollout_steps=100, stride=2, workers=cfg.training.workers,
                                        force_indices=(0, 1, 2, 3))
        inventory = p026.inventory(train)
        indices = identity_helper.identity_index(train)
        # Reuse the already approved audit map, never enumerate heldout roots.
        actual = {}
        for child in train._datasets:
            for path in child.paths:
                if path.is_symlink() or path.parent.name != "train":
                    raise ValueError("train-only regular HDF required")
                actual[path.name] = sha(path)
        expected_files = {Path(k).name: v for k, v in audit["train_hdf_sha256"].items()}
        if len(actual) != 44 or actual != expected_files:
            raise ValueError("exact44 training bytes differ")
        sampler = DataLoader(train, batch_size=1, shuffle=True, prefetch_factor=0,
                             use_streams=False, seed=20261003)
        expected = list(iter(sampler.sampler))
        if identity_helper.sequence_sha(expected) != ORDER_SHA:
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
            result = objective.flow_rollout_objective(flow, x["state"], x["target_state"], x["mask"],
                                                  x["omega"], predict, p013.make_inputs, backward=True)
            result.pop("prediction", None)
            result.pop("predictions", None)
            result.pop("normalized_predictions", None)
            result["identity"] = ident
            print(json.dumps(dict(event="training_window_complete", **ident)), flush=True)
            return result
        records = []
        observed = []
        if mode == "resource-probe":
            sample, metadata = train[816]
            ident = identity_helper.training_identity([metadata])
            flow.zero_grad(set_to_none=True)
            records.append(run({k: v[None] for k, v in sample.items()}, ident))
            mask_rows(flow, None, saved)
            gradients = finite_gradients(flow)
            if identity_helper.tensor_state_sha256(flow) != flow_initial_sha:
                raise RuntimeError("no-update probe modified flow")
            observed = [816]
            optimizer = None
        else:
            optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
            check_optimizer(flow, optimizer, 0)
            iterator = iter(loader)
            for step in range(1, 172):
                def windows():
                    for _ in range(8):
                        batch, metadata = next(iterator)
                        ident = identity_helper.training_identity(metadata)
                        observed.append(indices[(ident["case"], ident["start"], ident["dataset_index"])])
                        yield batch, ident
                record = objective.accumulate_eight_window_gradients(flow, optimizer, windows(), lambda item: run(*item))
                record.update(optimizer_update(flow, optimizer, saved, step))
                record.update(update=step, consumed_windows=len(observed))
                records.append(record)
                frozen_proof()
                guard()
                print(json.dumps(dict(event="accumulation_update_complete", update=step)), flush=True)
            if observed != expected or next(iterator, None) is not None:
                raise RuntimeError("observed original order differs")
            gradients = None
        frozen_proof()
        guard()
        flow_terminal_sha = identity_helper.tensor_state_sha256(flow)
        output.mkdir()
        result = dict(status="FC_P028_" + ("RESOURCE_PROBE_COMPLETE" if mode == "resource-probe" else "TRAINING_COMPLETE") + "_NOT_ADMISSION",
                      protocol=protocol(), mode=mode, training_windows=len(observed), optimizer_steps=0 if optimizer is None else 171,
                      records=records, probe_gradient_norms=gradients, flow_initial_tensor_sha256=flow_initial_sha,
                      flow_terminal_tensor_sha256=flow_terminal_sha, frozen_aerodynamic_tensor_sha256=aero_sha,
                      sampler_order_sha256=identity_helper.sequence_sha(observed), inventory=inventory,
                      resources=resources, precision=precision, source_spec=spec, scientific_admission=False,
                      optimizer_memory_accounting=optimizer_memory_estimate(flow),
                      validation_accessed=False, frozen_test_accessed=False,
                      unused_force_rows_preserved=True, unused_force_outputs_preserved=False,
                      cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
        if optimizer is not None:
            result["optimizer_memory_accounting"]["optimizer_created"] = True
            result["optimizer_memory_accounting"]["optimizer_peak_measured"] = True
            save_terminal(output, flow, aero, optimizer, cfg, identity, result, official.build_model,
                          save_checkpoint, load_checkpoint, identity_helper.tensor_state_sha256)
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    finally:
        frozen_proof()
        (train if train is not None else base).close()


def save_terminal(output, flow, aero, optimizer, cfg, identity, result, build, save, load, tensor_sha):
    """New explicit role schema; no claim of current downstream compatibility."""
    data = json.dumps(protocol(), sort_keys=True, separators=(",", ":"))
    (output / "training_protocol.json").write_text(data)
    protocol_sha = sha(output / "training_protocol.json")
    metadata = dict(status="FC_P028_FLOW_ROLLOUT_CHECKPOINT", training_experiment="FC-P028",
                    checkpoint_epoch=1, training_protocol_sha256=protocol_sha,
                    training_protocol_file="training_protocol.json", accumulation_windows=8,
                    training_windows=1368, optimizer_steps=171, actual_learning_rate=1e-5,
                    parent_manifest_sha256=PARENT_SHA, source_sha256=result["source_spec"]["source_sha256"])
    for role_name in ("flow", "aerodynamic"):
        for kind in ("model", "state"):
            metadata[role_name + "_parent_" + kind + "_sha256"] = identity.payload[role_name][kind + "_sha256"]
    save(output / "flow", models=flow, optimizer=optimizer, epoch=1, metadata=metadata)
    flow.cpu()
    fresh = build(cfg)
    actual_meta = {}
    if load(output / "flow", models=fresh, metadata_dict=actual_meta, device="cpu") != 1 or actual_meta != metadata or tensor_sha(fresh) != result["flow_terminal_tensor_sha256"]:
        raise RuntimeError("official flow fresh reload differs")
    del fresh
    (output / "aerodynamic").mkdir()
    role = identity.aerodynamic
    for name in (identity.payload["aerodynamic"]["model_file"], identity.payload["aerodynamic"]["state_file"]):
        shutil.copy2(role.directory / name, output / "aerodynamic" / name)
        if sha(output / "aerodynamic" / name) != sha(role.directory / name):
            raise RuntimeError("frozen aerodynamic copy differs")
    fresh = build(cfg)
    if load(output / "aerodynamic", models=fresh, metadata_dict={}, device="cpu") != 1 or tensor_sha(fresh) != tensor_sha(aero):
        raise RuntimeError("official frozen aerodynamic fresh reload differs")
    inherited_keys = ("schema_version", "architecture", "flow_architecture", "aerodynamic_architecture",
                      "config_sha256", "normalization_sha256", "precision_protocol", "input_sha256",
                      "history_input", "history_inventory", "history_state_module_sha256",
                      "history_inference_module_sha256", "aerodynamic_initial_model_sha256",
                      "aerodynamic_initial_state_sha256")
    manifest = {key: copy.deepcopy(identity.payload[key]) for key in inherited_keys}
    manifest.update(status="FC_P028_DUAL_FNO_MANIFEST_VERIFIED", kind="FC_P028_FLOW_ROLLOUT_REPAIR",
                    training_experiment="FC-P028", parent_manifest_sha256=PARENT_SHA,
                    training_protocol_file="training_protocol.json", training_protocol_sha256=protocol_sha,
                    training_windows=1368, optimizer_steps=171, accumulation_windows=8,
                    actual_learning_rate=1e-5, training_semantics=protocol(),
                    config_role="base_architecture_data_only_effective_P028_protocol",
                    base_config_sha256=CONFIG_SHA, scientific_admission=False,
                    source_sha256=result["source_spec"]["source_sha256"],
                    checkpoint_sha256={str(p.relative_to(output)): sha(p) for role_name in ("flow", "aerodynamic") for p in sorted((output / role_name).iterdir())})
    for role_name in ("flow", "aerodynamic"):
        role_record = copy.deepcopy(identity.payload[role_name])
        role_record.update(frozen=role_name == "aerodynamic", checkpoint_epoch=1)
        if role_name == "flow":
            role_record.update(model_file="FNO.0.1.mdlus", state_file="checkpoint.0.1.pt",
                               metadata_kind=metadata["status"])
        for kind in ("model", "state"):
            role_record[kind + "_sha256"] = sha(output / role_name / role_record[kind + "_file"])
            manifest[role_name + "_parent_" + kind + "_sha256"] = metadata[role_name + "_parent_" + kind + "_sha256"]
        manifest[role_name] = role_record
    (output / "dual_model_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    result["official_fresh_reload_verified"] = True
    result["training_protocol_sha256"] = protocol_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--mode", choices=("resource-probe", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("FC_P028_PREPARATION_ONLY_NO_MODEL_OR_DATA_ACCESS")
        return
    spec = json.loads(checked(dict(path=str(args.spec), sha256=args.spec_sha256)).read_text())
    execute(spec, args.mode, args.output)


if __name__ == "__main__":
    main()
