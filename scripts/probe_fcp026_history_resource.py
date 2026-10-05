"""One-window K1/K4 production-size resource probe; no optimizer or save."""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time

P019_SHA = "03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64"
HISTORY_SHA = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"
OBJECTIVE_SHA = "4d27fb53e05df73ba94d84bf42ba8205d78ebe6f91de832a68659870ea7d77c0"
OFFICIAL_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
FROZEN = (
    "spec_encoder.lift_network.0.conv.bias",
    "spec_encoder.lift_network.2.conv.bias",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, digest, name):
    if sha(path) != digest:
        raise ValueError(name + " source differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits(memory, elapsed, startup=False):
    if elapsed > 900:
        raise RuntimeError("whole-probe900second deadline")
    if memory.get("MemFree", 0) < (30 if startup else 20) or memory.get(
        "MemAvailable", 0
    ) < (50 if startup else 20):
        raise RuntimeError("physical memory floor")


def validate_metadata(role, epoch, metadata, manifest):
    expected = manifest[role]
    if (
        epoch != expected["checkpoint_epoch"]
        or metadata.get("status") != expected["metadata_kind"]
    ):
        raise ValueError("separate parent epoch/kind differs")
    if role == "flow":
        required = dict(
            candidate_checkpoint_epoch=0,
            parent_checkpoint_epoch=2,
            calibration_generation=1,
            alpha=0.0,
            domain_mix={"free_ar": 0.5, "matched_weight_h1": 0.5},
            calibration_fit_performed=True,
            optimizer_training_performed=False,
            validation_accessed=False,
            frozen_test_accessed=False,
            ppo_executed=False,
        )
        if any(metadata.get(key) != value for key, value in required.items()):
            raise ValueError("P009 calibrated flow metadata differs")
    if role == "aerodynamic":
        keys = (
            "training_experiment",
            "accumulation_windows",
            "training_windows",
            "optimizer_steps",
            "actual_learning_rate",
            "training_protocol_sha256",
            "training_protocol_file",
            "flow_parent_model_sha256",
            "flow_parent_state_sha256",
            "aerodynamic_initial_model_sha256",
            "aerodynamic_initial_state_sha256",
        )
        if any(metadata.get(key) != manifest[key] for key in keys):
            raise ValueError("P018 aerodynamic metadata differs")
        if any(
            metadata.get(key) is not False
            for key in (
                "selection_performed",
                "validation_accessed",
                "frozen_test_accessed",
                "ppo_executed",
            )
        ):
            raise ValueError("P018 scope metadata differs")


def trainable_scope(model):
    names = dict(model.named_parameters())
    if not all(name in names for name in FROZEN):
        raise ValueError("two original frozen biases missing")
    for name, parameter in names.items():
        parameter.requires_grad_(name not in FROZEN)
    if sum(p.requires_grad for p in names.values()) != 28:
        raise ValueError("expected28 trainable tensors")


def validate_arm_schema(model, parent, k, lift):
    actual = dict(model.named_parameters())
    expected = dict(parent.named_parameters())
    if actual.keys() != expected.keys():
        raise ValueError("arm parameter names differ from parent")
    if expected[lift].shape != (24, 8, 1, 1) or actual[lift].shape != (
        24,
        8 if k == 1 else 20,
        1,
        1,
    ):
        raise ValueError("official lifting24x8 to24x20 shape differs")
    if {name for name, param in actual.items() if param.requires_grad} != set(
        expected
    ) - set(FROZEN):
        raise ValueError("exact trainable name set differs")


def gradient_report(model, lift, k):
    import torch

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if len(params) != 28 or any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()
    ):
        raise RuntimeError("28 finite gradients required")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("frozen bias gradient")
    extra = (
        torch.cat((params[lift].grad[:, :9], params[lift].grad[:, 13:16]), dim=1)
        if k == 4
        else None
    )
    if extra is not None and (
        extra.numel() != 288
        or not torch.isfinite(extra).all()
        or extra.norm().item() == 0
    ):
        raise RuntimeError("finite nonzero new288 gradient required")
    count = sum(p.numel() * p.element_size() for p in params.values())
    return dict(
        trainable_tensors=28,
        new_history_scalars=288 if k == 4 else 0,
        new_history_gradient_norm=(
            float(extra.double().norm()) if extra is not None else None
        ),
        trainable_parameter_bytes=count,
        projected_two_adam_moments_bytes=2 * count,
        optimizer_projection_only=True,
    )


def execute(args, history, chunk, helper, objective, started):
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model, predict

    resources = []
    phase = "preflight"

    def guard(startup=False):
        values = {
            line.split(":")[0]: int(line.split()[1]) / 2**20
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemFree:", "MemAvailable:"))
        }
        elapsed = time.monotonic() - started
        limits(values, elapsed, startup)
        resources.append(dict(elapsed_seconds=elapsed, phase=phase, **values))

    cfg = OmegaConf.load(args.config)
    trainer = objective.load_frozen_trainer(
        args.source_root / "scripts/train_fcp011_decoder_scope.py"
    )
    data_identity = trainer.validate_data_contract(cfg)
    manifest = json.loads((args.candidate / "dual_model_manifest.json").read_text())
    guard(True)
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count() != 1:
        raise RuntimeError("one approved CUDA device required")
    torch.cuda.set_per_process_memory_fraction(0.06, dist.device)
    precision = validate_runtime_precision()
    if precision != manifest["precision_protocol"]:
        raise ValueError("parent runtime precision differs")
    parents = {}
    metadata_records = {}
    for role in ("flow", "aerodynamic"):
        model = build_model(cfg).to(dist.device)
        if sha(inspect.getfile(type(model))) != OFFICIAL_SHA:
            raise ValueError("official model source differs")
        metadata = {}
        epoch = load_checkpoint(
            args.candidate / role,
            models=model,
            metadata_dict=metadata,
            device=dist.device,
        )
        validate_metadata(role, epoch, metadata, manifest)
        if objective.tensor_state_sha256(model) != helper.TENSORS[role]:
            raise ValueError("separate parent tensors differ")
        for p in model.parameters():
            p.requires_grad_(False)
        parents[role] = model
        metadata_records[role] = metadata
    flow, legacy = parents["flow"], parents["aerodynamic"]
    flow.eval()
    legacy.cpu()
    dataset = TandemRolloutDataset(
        cfg.data.additional_train_roots[0],
        "train",
        100,
        stride=2,
        num_workers=1,
        force_indices=(0, 1, 2, 3),
    )
    active = None
    active_sha = None
    try:
        file_index = next(
            i
            for i, p in enumerate(dataset.paths)
            if p.stem == "dynamic_train8_b00_prbs"
        )
        local_index = dataset.index.index((file_index, 90))
        if local_index != 96:
            raise ValueError("original global816/local96 index differs")
        sample, identity, past = history.HistoryWindowAdapter(dataset, 4)[local_index]
        if identity != dict(
            case="dynamic_train8_b00_prbs", step=90, rollout_steps=100, split="train"
        ) or past["metadata"]["frame_indices"] != [87, 88, 89, 90]:
            raise ValueError("warm window identity differs")
        batch = {key: value[None].to(dist.device) for key, value in sample.items()}
        calls = dict(flow=0)

        def flow_predict(*items):
            guard()
            calls["flow"] += 1
            return predict(*items)

        phase = "frozen_flow"
        flow_states = objective.frozen_flow_states(
            flow, batch["state"], batch["mask"], batch["omega"], flow_predict
        ).detach()
        h1_states = objective.true_state_inputs(
            batch["state"], batch["target_state"]
        ).detach()
        frozen_hash = objective.tensor_sha256(flow_states)
        flow.cpu()
        pre = past["states"][:-1][None].to(dist.device)
        pre_actions = past["actions"][:-1][None].to(dist.device)
        arms = []
        reference = None
        for k in (1, 4):
            phase = "K" + str(k)
            arm_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            arm_cfg.model.in_channels = 6 if k == 1 else 18
            active = build_model(arm_cfg)
            active.load_state_dict(
                history.history_warmstart_state(
                    legacy.state_dict(), active.state_dict(), k
                ),
                strict=True,
            )
            trainable_scope(active)
            validate_arm_schema(active, legacy, k, history.LIFT)
            active.to(dist.device).train()
            active_sha = objective.tensor_state_sha256(active)
            forward_calls = 0

            def counted(*items):
                nonlocal forward_calls
                guard()
                forward_calls += 1
                return predict(*items)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.monotonic()
            result = chunk.chunk_force_objective(
                active,
                flow_states,
                h1_states,
                batch["mask"],
                batch["omega"],
                batch["target_force"],
                pre[:, : k - 1],
                pre_actions[:, : k - 1],
                counted,
                objective,
                backward=True,
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            gradients = gradient_report(active, history.LIFT, k)
            predictions = {
                d: x.cpu() for d, x in result.pop("normalized_predictions").items()
            }
            if k == 1:
                reference = predictions
            else:
                for domain in ("h1", "ar"):
                    torch.testing.assert_close(
                        predictions[domain], reference[domain], rtol=1e-5, atol=1e-6
                    )
            if (
                forward_calls != 10
                or objective.tensor_sha256(flow_states) != frozen_hash
            ):
                raise RuntimeError("chunk calls/frozen history changed")
            arms.append(
                dict(
                    k=k,
                    elapsed_seconds=elapsed,
                    forward_calls=forward_calls,
                    backward_chunks=10,
                    gradients=gradients,
                    objective=result,
                    normalized_predictions={
                        d: x.tolist() for d, x in predictions.items()
                    },
                    replay_max_abs={
                        d: float((predictions[d] - reference[d]).abs().max())
                        for d in predictions
                    },
                    expanded_initial_sha256=active_sha,
                    shared_flow_history_sha256=frozen_hash,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            )
            active.zero_grad(set_to_none=True)
            if objective.tensor_state_sha256(active) != active_sha:
                raise RuntimeError("arm model changed without update")
            active.cpu()
            active = None
            active_sha = None
            torch.cuda.empty_cache()
            guard()
            print(
                json.dumps(dict(event="arm_complete", k=k, elapsed_seconds=elapsed)),
                flush=True,
            )
        if calls["flow"] != 100:
            raise RuntimeError("flow call count differs")
        return dict(
            status="FC_P026_HISTORY_RESOURCE_COMPLETE_NOT_ADMISSION",
            arms=arms,
            identity=identity,
            global_index=816,
            history_metadata=past["metadata"],
            flow_calls=100,
            parent_tensors=helper.TENSORS,
            parent_metadata=metadata_records,
            data_identity=data_identity,
            precision=precision,
            resources=resources,
            replay_rtol=1e-5,
            replay_atol=1e-6,
            allocator_fraction=0.06,
            optimizer_steps=0,
            candidate_saved=False,
            heldout_accessed=False,
            scientific_admission=False,
        )
    finally:
        if active is not None:
            active.zero_grad(set_to_none=True)
            if (
                active_sha is not None
                and objective.tensor_state_sha256(active) != active_sha
            ):
                raise RuntimeError("failure cleanup arm tensor changed")
        for role, parent in parents.items():
            if objective.tensor_state_sha256(parent) != helper.TENSORS[role] or any(
                p.grad is not None for p in parent.parameters()
            ):
                raise RuntimeError("original parent changed")
        dataset.close()


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "source-root",
        "diagnostic-script",
        "gradient-helper",
        "history-module",
        "history-objective",
        "config",
        "candidate",
        "candidate-audit",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if not args.execute or args.output.exists():
        raise RuntimeError("explicit approval and exclusive output required")
    try:
        history = load(args.history_module, HISTORY_SHA, "p026_history")
        chunk = load(args.history_objective, OBJECTIVE_SHA, "p026_objective")
        helper = load(args.gradient_helper, P019_SHA, "p026_p019")
        _, objective = helper.load_dependencies(args)
        result = execute(args, history, chunk, helper, objective, started)
        result.update(
            harness_sha256=sha(__file__),
            elapsed_seconds=time.monotonic() - started,
            source_sha256={
                name: sha(getattr(args, name))
                for name in (
                    "history_module",
                    "history_objective",
                    "gradient_helper",
                    "diagnostic_script",
                    "config",
                    "candidate_audit",
                )
            },
        )
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    except Exception as error:
        with args.output.with_name(args.output.name + ".failure.json").open(
            "x"
        ) as stream:
            json.dump(
                dict(
                    status="FC_P026_RESOURCE_FAILED_NOT_ADMISSION",
                    error=repr(error),
                    elapsed_seconds=time.monotonic() - started,
                ),
                stream,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
