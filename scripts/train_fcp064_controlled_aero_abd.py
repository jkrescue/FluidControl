"""Fixed-budget FC-P064 aero-data A/B/C/D continuation from the K1 terminal."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

P015_SHA = "2f5de1a946b040c42c21f8164da41a5afc642fd6174e7bb35372bb7ba98eb996"
P019_SHA = "03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64"
P020_SHA = "139dee2c3dcd4ee97de78a7a0e343ca9ab9d9708a1cafab447f0d9e4988b5d8c"
P014_SHA = "849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d"
AUDIT_SHA = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
HISTORY_SHA = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"
OBJECTIVE_SHA = "4d27fb53e05df73ba94d84bf42ba8205d78ebe6f91de832a68659870ea7d77c0"
RESOURCE_SHA = "b806ded8258c787807e67ccb42b5166dbd06e36fc40eba5025d9bda0769eedc6"
INFERENCE_SHA = "fd568f6457b980046a0419be96562291e9f96736270d45f20dd2adb2ffc5878c"
CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
MANIFEST_SHA = "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"
ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
FLOW_TENSOR_SHA = "89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb"
AERO_TENSOR_SHA = "b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def history_input(k):
    if k not in (1, 4):
        raise ValueError("fixed K1/K4 only")
    return dict(
        schema_version=1,
        profile=f"p026_k{k}",
        history_length=k,
        flow_input_channels=6,
        aerodynamic_input_channels=6 if k == 1 else 18,
        left_padding="trajectory_frame0",
        autoregressive_state_source="frozen_flow_prediction",
        future_state_inputs=False,
        future_force_inputs=False,
    )


def protocol(arm, schedule, parent_order):
    controlled_windows = {"A": 0, "B": 64, "C": 128, "D": 64}[arm]
    replacement_positions = {
        "A": [], "B": [0, 4], "C": [0, 2, 4, 6], "D": [0, 4]
    }[arm]
    result = dict(
        training_experiment="FC-P064",
        arm=arm,
        parent_experiment="FC-P026-K1",
        history_input=history_input(1),
        training_windows=256,
        accumulation_windows=8,
        optimizer_steps=32,
        learning_rate=1.5625e-7,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        seed=20261003,
        chunk_size=10,
        rollout_steps=100,
        parent_sampler_order_sha256=ORDER_SHA,
        schedule_sha256=schedule.schedule_sha256(schedule.compile_schedule(parent_order, arm)),
        diagnostic_counts=[0, 256],
        objective="equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
        history_state_module_sha256=HISTORY_SHA,
        history_objective_sha256=OBJECTIVE_SHA,
        history_inference_module_sha256=INFERENCE_SHA,
        action_semantics="stored_prescribed_action_samples_not_exact_nominal_time_commands",
        controlled_b00_action_semantics=(
            "actual_closed_loop_applied_endpoint_omega_samples"
            if arm in ("B", "C")
            else "not_applicable_no_b00_windows"
        ),
        b00_windows=controlled_windows,
        b00_weight=controlled_windows / 256,
        replacement_within_each_update=replacement_positions,
        allocator_fraction=0.06,
        wall_seconds=3600,
        validation_accessed=False,
        frozen_test_accessed=False,
        selection_performed=False,
    )
    if arm == "D":
        result.update(
            controlled_b00_action_semantics="actual_k1_projected_closed_loop_applied_endpoint_omega_samples",
            b00_windows=32,
            b00_weight=0.125,
            b02_windows=32,
            b02_weight=0.125,
            controlled_b02_action_semantics="actual_symmetry_canonical_closed_loop_applied_endpoint_omega_samples",
            controlled_source_profile="32_b00_k1_projected_plus_32_b02_canonical_policy",
        )
    return result


def inventory(dataset):
    children = getattr(dataset, "_datasets", None)
    if not isinstance(children, list) or len(children) != 3:
        raise ValueError("official three-family MultiDataset required")
    counts = [len(x.index) for x in children]
    padded = [sum(int(start) < 3 for _, start in x.index) for x in children]
    result = dict(
        windows=sum(counts),
        warm=sum(counts) - sum(padded),
        padded=sum(padded),
        family_windows=counts,
        family_padded=padded,
    )
    if result != dict(
        windows=1368,
        warm=1300,
        padded=68,
        family_windows=[720, 408, 240],
        family_padded=[20, 16, 32],
    ):
        raise ValueError("real1368/1300/68 inventory differs")
    return result


def preceding(dataset, identity, sample, k, history):
    """Use original batch; only preceding frames touch its official reader."""
    import torch

    child = dataset._datasets[identity["dataset_index"]]
    matches = [i for i, p in enumerate(child.paths) if p.stem == identity["case"]]
    if len(matches) != 1 or identity["split"] != "train":
        raise ValueError("history source identity differs")
    indices, padded = history.history_indices(identity["start"], k)
    mask = sample["mask"]
    current = sample["state"]
    states = []
    actions = []
    cache = {}
    for index in indices[:-1]:
        if index == identity["start"]:
            state, action = current, sample["omega"][0]
        else:
            if index not in cache:
                frame, _ = child._reader(matches[0])[index]
                if not torch.equal(frame["mask"].float().to(mask.device), mask):
                    raise ValueError("preceding mask differs")
                cache[index] = (
                    ((frame["state"].float() - child.state_mean) / child.state_std)
                    * frame["mask"].float(),
                    frame["omega"].float().reshape(1) / child.action_scale,
                )
            state, action = cache[index]
        states.append(state.to(current.device))
        actions.append(action.to(current.device))
    return (
        (
            torch.stack(states)[None]
            if states
            else current.new_empty((1, 0, *current.shape))
        ),
        (
            torch.stack(actions).reshape(1, k - 1, 1)
            if actions
            else current.new_empty((1, 0, 1))
        ),
        dict(
            frame_indices=indices,
            padding_mask=padded,
            full_observed_history=not any(padded),
        ),
    )


def snapshot(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def split_norm(values, k, lift):
    """CPU-double reductions, no additional full-size GPU optimizer buffers."""
    import torch

    inherited = 0.0
    new = 0.0
    for name, value in values.items():
        x = value.detach().cpu().double()
        if k == 4 and name == lift:
            new += float(x[:, :9].square().sum() + x[:, 13:16].square().sum())
            inherited += float(x[:, 9:13].square().sum() + x[:, 16:].square().sum())
        else:
            inherited += float(x.square().sum())
    return dict(new_history_l2=new**0.5, inherited_l2=inherited**0.5)


def displacement(model, reference, k, lift):
    return split_norm(
        {
            name: p.detach().cpu() - reference[name]
            for name, p in model.named_parameters()
            if p.requires_grad
        },
        k,
        lift,
    )


def check_optimizer(model, optimizer, steps):
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    if (
        len(trainable) != 28
        or len(optimizer.param_groups) != 1
        or [id(x) for x in optimizer.param_groups[0]["params"]]
        != [id(x) for x in trainable]
    ):
        raise ValueError("exact28 optimizer parameter ownership differs")
    g = optimizer.param_groups[0]
    if (g["lr"], tuple(g["betas"]), g["eps"], g["weight_decay"]) != (
        1.5625e-7,
        (0.9, 0.999),
        1e-8,
        1e-4,
    ):
        raise ValueError("optimizer protocol differs")
    if steps == 0:
        if optimizer.state:
            raise ValueError("fresh optimizer required")
    elif set(optimizer.state) != set(trainable) or any(
        float(state["step"]) != steps for state in optimizer.state.values()
    ):
        raise ValueError("28 Adam state step counts differ")
    if any(
        torch.is_tensor(v) and not torch.isfinite(v).all()
        for s in optimizer.state.values()
        for v in s.values()
    ):
        raise FloatingPointError("nonfinite Adam state")


def validate_b00_view(args):
    """Bind the one-file read-only training view before any model is loaded."""
    root = args.b00_data_root.resolve()
    manifest_path = args.b00_manifest.resolve()
    receipt_path = args.b00_conversion_receipt.resolve()
    if manifest_path != root / "manifest.json":
        raise ValueError("b00 manifest confinement differs")
    if sha(manifest_path) != args.b00_manifest_sha256:
        raise ValueError("b00 view manifest SHA differs")
    if sha(receipt_path) != args.b00_conversion_receipt_sha256:
        raise ValueError("b00 conversion receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        float(manifest.get("max_abs_omega", -1)) != 0.75
        or manifest.get("conversion_receipt_sha256")
        != args.b00_conversion_receipt_sha256
        or manifest.get("validation_accessed") is not False
        or manifest.get("frozen_test_accessed") is not False
    ):
        raise ValueError("b00 train-only view contract differs")
    normalization = root / "normalization.json"
    if sha(normalization) != manifest.get("normalization_sha256"):
        raise ValueError("b00 normalization bytes differ")
    files = sorted((root / "train").glob("*.h5"))
    declared = manifest.get("hdf_sha256", {})
    if len(files) != 1 or set(declared) != {files[0].name}:
        raise ValueError("b00 view must expose exactly one HDF")
    if sha(files[0]) != declared[files[0].name]:
        raise ValueError("b00 HDF bytes differ")
    receipt_exact = {
        "status": "B00_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING",
        "frames": 801,
        "trajectories": 1,
        "split": "train",
        "normalization_sha256": manifest["normalization_sha256"],
        "normalization_refit": False,
        "official_reader_verified": True,
        "source_unchanged": True,
        "model_loaded": False,
        "cfd_executed": False,
        "optimizer_steps": 0,
        "scientific_admission": False,
        "owned_containers_cleaned": True,
    }
    if any(receipt.get(key) != value for key, value in receipt_exact.items()):
        raise ValueError("b00 conversion did not complete the reviewed official-reader contract")
    if receipt.get("hdf") != {
        "path": str(args.b00_source_hdf),
        "sha256": declared[files[0].name],
    }:
        raise ValueError("b00 conversion result and view HDF identity differ")
    return manifest


def validate_b02_view(args):
    """Bind the future b02 one-file train-only view without reading arrays."""
    root = args.b02_data_root.resolve()
    manifest_path = args.b02_manifest.resolve()
    receipt_path = args.b02_conversion_receipt.resolve()
    if manifest_path != root / "manifest.json":
        raise ValueError("b02 manifest confinement differs")
    if sha(manifest_path) != args.b02_manifest_sha256:
        raise ValueError("b02 view manifest SHA differs")
    if sha(receipt_path) != args.b02_conversion_receipt_sha256:
        raise ValueError("b02 conversion receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        float(manifest.get("max_abs_omega", -1)) != 0.75
        or manifest.get("conversion_receipt_sha256")
        != args.b02_conversion_receipt_sha256
        or manifest.get("validation_accessed") is not False
        or manifest.get("frozen_test_accessed") is not False
    ):
        raise ValueError("b02 train-only view contract differs")
    normalization = root / "normalization.json"
    if sha(normalization) != manifest.get("normalization_sha256"):
        raise ValueError("b02 normalization bytes differ")
    files = sorted((root / "train").glob("*.h5"))
    declared = manifest.get("hdf_sha256", {})
    if len(files) != 1 or set(declared) != {files[0].name}:
        raise ValueError("b02 view must expose exactly one HDF")
    if sha(files[0]) != declared[files[0].name]:
        raise ValueError("b02 HDF bytes differ")
    receipt_exact = {
        "status": "B02_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING",
        "frames": 801, "trajectories": 1, "split": "train",
        "normalization_sha256": manifest["normalization_sha256"],
        "normalization_refit": False, "official_reader_verified": True,
        "source_unchanged": True, "model_loaded": False,
        "cfd_executed": False, "optimizer_steps": 0,
        "scientific_admission": False, "owned_containers_cleaned": True,
    }
    if any(receipt.get(key) != value for key, value in receipt_exact.items()):
        raise ValueError("b02 conversion did not complete official-reader contract")
    if receipt.get("hdf") != {
        "path": str(args.b02_source_hdf), "sha256": declared[files[0].name]
    }:
        raise ValueError("b02 conversion result and view HDF identity differ")
    return manifest


def validate_training_roots(args):
    """Bind the three approved train-only roots without mutating the frozen config."""
    records = {}
    for name in ("base", "train8", "train16"):
        root = getattr(args, f"{name}_data_root").resolve()
        manifest = root / "manifest.json"
        normalization = root / "normalization.json"
        if (
            not (root / "train").is_dir()
            or sha(manifest) != getattr(args, f"{name}_manifest_sha256")
            or sha(normalization) != getattr(args, f"{name}_normalization_sha256")
        ):
            raise ValueError(f"{name} train-only root identity differs")
        records[name] = {
            "root": str(root),
            "manifest_sha256": sha(manifest),
            "normalization_sha256": sha(normalization),
        }
    if len({row["normalization_sha256"] for row in records.values()}) != 1:
        raise ValueError("three-family normalization bytes differ")
    return records


def optimizer_signature(optimizer, objective):
    import torch

    return [
        {
            key: objective.tensor_sha256(value) if torch.is_tensor(value) else value
            for key, value in sorted(state.items())
        }
        for state in optimizer.state.values()
    ]


def grouped_panel(rows):
    """Same K4-availability partition for both arms; not new admission metrics."""
    result = {}
    for name, warm in (("warm", True), ("padded", False)):
        selected = [row for row in rows if (row["identity"]["start"] >= 3) == warm]
        nonzero = [row for row in selected if row["global_index"] != 160]
        result[name] = dict(
            window_count=len(selected),
            nonzero_count=len(nonzero),
            domains={
                domain: dict(
                    mean_original_objective=(
                        sum(
                            row["panel"]["objective"][domain + "_balanced"]
                            for row in selected
                        )
                        / len(selected)
                        if selected
                        else None
                    ),
                    nonzero_statistics={
                        key: (
                            sum(row["panel"]["domains"][domain][key] for row in nonzero)
                            / len(nonzero)
                            if nonzero
                            else None
                        )
                        for key in (
                            "bias_mse",
                            "rms_error_mse",
                            "absolute_rms_error",
                            "centered_residual_mse",
                        )
                    },
                )
                for domain in ("h1", "ar")
            },
        )
    return result


def execute(
    args, resource, history, chunk, helper, p020, accumulation, diagnostic, objective
):
    import inspect
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.dual_fno import load_dual_fno, validate_runtime_precision
    from fluid_control import p064_controlled_aero_abd as schedule
    from train_tandem_fno import build_model, predict

    started = time.monotonic()
    samples = []

    def guard(startup=False):
        memory = {
            line.split(":")[0]: int(line.split()[1]) / 2**20
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemFree:", "MemAvailable:"))
        }
        if (
            time.monotonic() - started > 3600
            or memory.get("MemAvailable", 0) < (50 if startup else 22)
        ):
            raise RuntimeError("resource/deadline floor")
        samples.append(dict(elapsed_seconds=time.monotonic() - started, **memory))

    cfg = OmegaConf.load(args.config)
    cfg.data.root = args.effective_data_identity["base"]["root"]
    cfg.data.additional_train_roots = [
        args.effective_data_identity["train8"]["root"],
        args.effective_data_identity["train16"]["root"],
    ]
    if (
        Path(inspect.getfile(schedule)).resolve() != args.schedule_adapter.resolve()
        or sha(args.schedule_adapter) != args.schedule_adapter_sha256
    ):
        raise ValueError("imported P064 schedule adapter source differs")
    trainer = objective.load_frozen_trainer(
        args.source_root / "scripts/train_fcp011_decoder_scope.py"
    )
    input_sha = trainer.validate_data_contract(cfg)
    if (
        sha(inspect.getfile(save_checkpoint)) != CHECKPOINT_SHA
        or sha(inspect.getfile(load_checkpoint)) != CHECKPOINT_SHA
    ):
        raise ValueError("official checkpoint source differs")
    guard(True)
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count() != 1:
        raise RuntimeError("single approved GPU required")
    torch.cuda.set_per_process_memory_fraction(0.06, dist.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    precision = validate_runtime_precision()
    parent_manifest = json.loads(args.parent_manifest.read_text())
    if precision != parent_manifest["precision_protocol"]:
        raise ValueError("parent precision differs")
    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)
    flow = None
    parent_order = json.loads(args.parent_order.read_text())
    adapter, identity = load_dual_fno(
        args.parent_manifest,
        cfg,
        "cpu",
        build_model=build_model,
        load_checkpoint=load_checkpoint,
        expected_manifest_sha256=MANIFEST_SHA,
    )
    if identity.payload["kind"] != "FC_P026_K1_HISTORY_FORCE_FNO":
        raise ValueError("exact P026 K1 parent required")
    flow, aero = adapter.flow_model, adapter.aerodynamic_model
    if any(
        sha(inspect.getfile(type(model))) != resource.OFFICIAL_SHA
        for model in (flow, aero)
    ):
        raise ValueError("official FNO source differs")
    if objective.tensor_state_sha256(flow) != FLOW_TENSOR_SHA:
        raise ValueError("P009 flow parent tensor differs")
    if objective.tensor_state_sha256(aero) != AERO_TENSOR_SHA:
        raise ValueError("P026 K1 aerodynamic parent tensor differs")
    flow.to(dist.device).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
    aero_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resource.trainable_scope(aero)
    resource.validate_arm_schema(aero, aero, 1, history.LIFT)
    aero.to(dist.device).train()
    initial = snapshot(aero)
    initial_sha = objective.tensor_state_sha256(aero)
    frozen = {
        n: objective.tensor_sha256(p)
        for n, p in aero.named_parameters()
        if not p.requires_grad
    }
    base = TandemRolloutDataset(
        cfg.data.root,
        "train",
        100,
        stride=20,
        num_workers=cfg.training.workers,
        force_indices=(0, 1, 2, 3),
    )
    train = None
    scheduled = None
    try:
        train, _ = compose_training_data(
            base,
            [Path(x) for x in cfg.data.additional_train_roots],
            rollout_steps=100,
            stride=2,
            workers=cfg.training.workers,
            force_indices=(0, 1, 2, 3),
        )
        actual_inventory = inventory(train)
        audit = DataLoader(
            train,
            batch_size=1,
            shuffle=True,
            prefetch_factor=0,
            use_streams=False,
            seed=20261003,
        )
        expected = list(iter(audit.sampler))
        if len(train) != 1368 or trainer.sequence_sha(expected) != ORDER_SHA:
            raise ValueError("original sampler order differs")
        b00 = None
        if args.arm in ("B", "C", "D"):
            b00 = TandemRolloutDataset(
                args.b00_data_root,
                "train",
                100,
                stride=1,
                num_workers=cfg.training.workers,
                force_indices=(0, 1, 2, 3),
            )
        b02 = None
        if args.arm == "D":
            b02 = TandemRolloutDataset(
                args.b02_data_root, "train", 100, stride=1,
                num_workers=cfg.training.workers, force_indices=(0, 1, 2, 3),
            )
        scheduled = schedule.ScheduledABCDataset(
            train,
            base,
            expected,
            args.arm,
            b00_dataset=b00,
            b02_dataset=b02,
            num_workers=cfg.training.workers,
        )
        loader = DataLoader(
            scheduled,
            batch_size=1,
            shuffle=False,
            collate_metadata=True,
            prefetch_factor=int(cfg.data.prefetch_factor),
            num_streams=int(cfg.data.num_streams),
            use_streams=True,
            seed=20261003,
        )
        optimizer = torch.optim.AdamW(
            [p for p in aero.parameters() if p.requires_grad],
            lr=1.5625e-7,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        check_optimizer(aero, optimizer, 0)
        items = trainer.diagnostic_windows(train)

        def run(batch, ident, backward, model=aero, k=args.history_k):
            guard()
            single = {key: value[0] for key, value in batch.items()}
            pre, actions, padding = preceding(scheduled, ident, single, k, history)
            batch = {key: value.to(dist.device) for key, value in batch.items()}
            pre = pre.to(dist.device)
            actions = actions.to(dist.device)

            def counted(*values):
                guard()
                return predict(*values)

            states = objective.frozen_flow_states(
                flow, batch["state"], batch["mask"], batch["omega"], counted
            )
            state_sha = objective.tensor_sha256(states)
            h1 = objective.true_state_inputs(batch["state"], batch["target_state"])
            result = chunk.chunk_force_objective(
                model,
                states,
                h1,
                batch["mask"],
                batch["omega"],
                batch["target_force"],
                pre,
                actions,
                counted,
                objective,
                backward=backward,
            )
            result.update(
                identity=ident, history=padding, flow_history_sha256=state_sha
            )
            return result

        def panel():
            before = objective.tensor_state_sha256(aero)
            grads = {
                n: None if p.grad is None else objective.tensor_sha256(p.grad)
                for n, p in aero.named_parameters()
            }
            py = random.getstate()
            optimizer_before = optimizer_signature(optimizer, objective)
            mode_before = {
                name: module.training for name, module in aero.named_modules()
            }
            npr = np.random.get_state()
            rows = []
            with torch.random.fork_rng(devices=[dist.device]), torch.no_grad():
                for item in items:
                    result = run(
                        {key: value[None] for key, value in item["sample"].items()},
                        item["identity"],
                        False,
                    )
                    preds = {
                        d: v[0].cpu()
                        for d, v in result.pop("normalized_predictions").items()
                    }
                    metrics = p020.metrics(
                        helper,
                        preds,
                        item["sample"]["target_force"],
                        base.force_std,
                        result,
                    )
                    rows.append(
                        dict(
                            global_index=item["global_index"],
                            identity=item["identity"],
                            history=result["history"],
                            flow_history_sha256=result["flow_history_sha256"],
                            panel=metrics,
                        )
                    )
            random.setstate(py)
            np.random.set_state(npr)
            if objective.tensor_state_sha256(aero) != before or grads != {
                n: None if p.grad is None else objective.tensor_sha256(p.grad)
                for n, p in aero.named_parameters()
            }:
                raise RuntimeError("diagnostic model/gradient mutation")
            if (
                optimizer_signature(optimizer, objective) != optimizer_before
                or {name: module.training for name, module in aero.named_modules()}
                != mode_before
            ):
                raise RuntimeError("diagnostic optimizer/mode mutation")
            return dict(
                rows=rows,
                aggregate=p020.aggregate(rows),
                history_subgroups=grouped_panel(rows),
            )

        replay = {"parent_tensor_sha256": AERO_TENSOR_SHA}
        panels = [dict(consumed_windows=0, **panel())]
        observed = []
        records = []
        iterator = iter(loader)
        for step in range(1, 33):
            before = snapshot(aero)

            def windows():
                for _ in range(8):
                    batch, meta = next(iterator)
                    ident = schedule.scheduled_training_identity(meta)
                    observed.append(scheduled.schedule[len(observed)])
                    yield batch, ident

            def train_window(item):
                result = run(*item, True)
                result.pop("normalized_predictions")
                print(
                    json.dumps(
                        dict(
                            event="training_window_complete",
                            history_k=args.history_k,
                            original_global_index=observed[-1].original_global_index,
                            source=observed[-1].source,
                            b00_start=(
                                observed[-1].b00_start
                                if observed[-1].source == "controlled_b00" else None
                            ),
                            b02_start=(
                                observed[-1].b00_start
                                if observed[-1].source == "controlled_b02" else None
                            ),
                            consumed=len(observed),
                        )
                    ),
                    flush=True,
                )
                return result

            def gradients(model):
                return split_norm(
                    {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
                    args.history_k,
                    history.LIFT,
                )

            record = accumulation.accumulation_step(
                aero, optimizer, windows(), train_window, gradients
            )
            check_optimizer(aero, optimizer, step)
            if any(
                p.grad is not None or objective.tensor_sha256(p) != frozen[n]
                for n, p in aero.named_parameters()
                if n in frozen
            ):
                raise RuntimeError("frozen bias changed")
            record.update(
                update=step,
                consumed_windows=step * 8,
                applied_clip_scale=min(
                    1.0, 1.0 / (record["preclip_mean_gradient_norm"] + 1e-6)
                ),
                parameter_update=displacement(
                    aero, before, args.history_k, history.LIFT
                ),
                cumulative_displacement=displacement(
                    aero, initial, args.history_k, history.LIFT
                ),
            )
            records.append(record)
            del before
            guard()
            print(
                json.dumps(
                    dict(
                        event="accumulation_update_complete",
                        history_k=args.history_k,
                        update=step,
                    )
                ),
                flush=True,
            )
        panels.append(dict(consumed_windows=256, **panel()))
        if observed != list(scheduled.schedule) or next(iterator, None) is not None:
            raise RuntimeError("observed sampler order differs")
        terminal_sha = objective.tensor_state_sha256(aero)
        ablation = None
        if args.history_k == 4:
            lift = dict(aero.named_parameters())[history.LIFT]
            saved = lift.detach().clone()
            try:
                with torch.no_grad():
                    lift[:, :9].zero_()
                    lift[:, 13:16].zero_()
                ablation = panel()
            finally:
                with torch.no_grad():
                    lift.copy_(saved)
            if objective.tensor_state_sha256(aero) != terminal_sha:
                raise RuntimeError("terminal ablation restoration failed")
        effective = protocol(args.arm, schedule, parent_order)
        protocol_digest = canonical_sha(effective)
        args.output.mkdir()
        (args.output / "training_protocol.json").write_text(
            json.dumps(effective, sort_keys=True, separators=(",", ":"))
        )
        if sha(args.output / "training_protocol.json") != protocol_digest:
            raise RuntimeError("effective protocol bytes differ")
        flow_dir = args.output / "flow"
        flow_dir.mkdir()
        for name in ("model_file", "state_file"):
            path = (
                args.parent_manifest.parent
                / parent_manifest["flow"]["checkpoint_relative_directory"]
                / parent_manifest["flow"][name]
            )
            shutil.copy2(path, flow_dir / path.name)
        info = dict(
            status=f"FC_P064_ARM_{args.arm}_CONTROLLED_AERO_CHECKPOINT",
            checkpoint_epoch=1,
            training_experiment="FC-P064",
            arm=args.arm,
            history_profile="p026_k1",
            history_k=1,
            model_in_channels=aero_cfg.model.in_channels,
            training_protocol_sha256=protocol_digest,
            training_protocol_file="training_protocol.json",
            history_state_module_sha256=HISTORY_SHA,
            history_inference_module_sha256=INFERENCE_SHA,
            training_windows=256,
            optimizer_steps=32,
            accumulation_windows=8,
            actual_learning_rate=1.5625e-7,
            parent_sampler_order_sha256=ORDER_SHA,
            schedule_sha256=effective["schedule_sha256"],
            parent_history_inventory=actual_inventory,
            selection_performed=False,
            validation_accessed=False,
            frozen_test_accessed=False,
            ppo_executed=False,
        )
        for role in ("flow", "aerodynamic"):
            info[role + "_parent_model_sha256"] = parent_manifest[role]["model_sha256"]
            info[role + "_parent_state_sha256"] = parent_manifest[role]["state_sha256"]
        save_checkpoint(
            args.output / "aerodynamic",
            models=aero,
            optimizer=optimizer,
            epoch=1,
            metadata=info,
        )
        aero.cpu()
        optimizer = None
        torch.cuda.empty_cache()
        fresh = build_model(aero_cfg)
        fresh_meta = {}
        if (
            load_checkpoint(
                args.output / "aerodynamic",
                models=fresh,
                metadata_dict=fresh_meta,
                device="cpu",
            )
            != 1
            or fresh_meta != info
            or objective.tensor_state_sha256(fresh) != terminal_sha
        ):
            raise RuntimeError("official aerodynamic fresh reload differs")
        del fresh
        fresh_flow = build_model(cfg)
        if (
            load_checkpoint(flow_dir, models=fresh_flow, metadata_dict={}, device="cpu")
            != 0
            or objective.tensor_state_sha256(fresh_flow) != FLOW_TENSOR_SHA
        ):
            raise RuntimeError("official frozen-flow fresh reload differs")
        del fresh_flow
        manifest = copy.deepcopy(parent_manifest)
        manifest.update(
            status=f"FC_P064_ARM_{args.arm}_DUAL_FNO_MANIFEST_VERIFIED",
            kind=f"FC_P064_ARM_{args.arm}_CONTROLLED_AERO_FORCE_FNO",
            training_experiment="FC-P064",
            arm=args.arm,
            history_input=history_input(1),
            parent_history_inventory=actual_inventory,
            training_protocol_sha256=protocol_digest,
            history_state_module_sha256=HISTORY_SHA,
            history_inference_module_sha256=INFERENCE_SHA,
            training_windows=256,
            optimizer_steps=32,
            accumulation_windows=8,
            actual_learning_rate=1.5625e-7,
            parent_manifest_sha256=MANIFEST_SHA,
        )
        manifest["p026_aerodynamic_initial_model_sha256"] = parent_manifest[
            "aerodynamic_initial_model_sha256"
        ]
        manifest["p026_aerodynamic_initial_state_sha256"] = parent_manifest[
            "aerodynamic_initial_state_sha256"
        ]
        manifest["aerodynamic_initial_model_sha256"] = parent_manifest[
            "aerodynamic"
        ]["model_sha256"]
        manifest["aerodynamic_initial_state_sha256"] = parent_manifest[
            "aerodynamic"
        ]["state_sha256"]
        # The legacy architecture remains the unchanged six-channel base/flow
        # schema. P026 loading must construct each role from its explicit schema.
        manifest["flow_architecture"] = copy.deepcopy(parent_manifest["architecture"])
        manifest["aerodynamic_architecture"] = copy.deepcopy(
            parent_manifest["architecture"]
        )
        manifest["aerodynamic_architecture"]["in_channels"] = aero_cfg.model.in_channels
        manifest["aerodynamic"].update(
            metadata_kind=info["status"],
            model_sha256=sha(args.output / "aerodynamic/FNO.0.1.mdlus"),
            state_sha256=sha(args.output / "aerodynamic/checkpoint.0.1.pt"),
        )
        for key in (
            "aerodynamic_parent_model_sha256",
            "aerodynamic_parent_state_sha256",
        ):
            manifest[key] = info[key]
        manifest["training_semantics"] = effective
        source = {
            key: sha(getattr(args, key))
            for key in (
                "history_module",
                "history_objective",
                "resource_helper",
                "accumulation_helper",
                "comparison_helper",
                "diagnostic_script",
                "history_inference_module",
                "config",
                "parent_manifest",
                "candidate_audit",
                "schedule_adapter",
                "parent_order",
                "b00_manifest",
                "b00_conversion_receipt",
            )
        }
        if args.arm == "D":
            source.update(
                b02_manifest=sha(args.b02_manifest),
                b02_conversion_receipt=sha(args.b02_conversion_receipt),
            )
        manifest["source_sha256"] = source
        manifest["effective_training_data"] = args.effective_data_identity
        (args.output / "dual_model_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        result = dict(
            status=f"FC_P064_ARM_{args.arm}_TRAINING_COMPLETE_NOT_ADMISSION",
            arm=args.arm,
            history_k=1,
            protocol_sha256=protocol_digest,
            parent_history_inventory=actual_inventory,
            records=records,
            fixed_train_panels=panels,
            terminal_history_zero_ablation=ablation,
            replay_max_abs=replay,
            parent_sampler_order_sha256=ORDER_SHA,
            schedule_sha256=effective["schedule_sha256"],
            optimizer_steps=32,
            training_windows=256,
            flow_tensor_sha256=FLOW_TENSOR_SHA,
            aerodynamic_initial_tensor_sha256=initial_sha,
            aerodynamic_terminal_tensor_sha256=terminal_sha,
            official_fresh_reload_verified=True,
            scientific_admission=False,
            source_sha256=source,
            trainer_sha256=sha(__file__),
            resources=samples,
            precision=precision,
            input_sha256=input_sha,
            effective_training_data=args.effective_data_identity,
        )
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
    finally:
        if flow is not None and (
            objective.tensor_state_sha256(flow) != FLOW_TENSOR_SHA
            or any(p.grad is not None for p in flow.parameters())
        ):
            raise RuntimeError("frozen P009 flow changed")
        (scheduled if scheduled is not None else train if train is not None else base).close()


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "source-root",
        "diagnostic-script",
        "accumulation-helper",
        "comparison-helper",
        "history-module",
        "history-objective",
        "history-inference-module",
        "resource-helper",
        "config",
        "parent-manifest",
        "parent-order",
        "schedule-adapter",
        "b00-data-root",
        "b00-manifest",
        "b00-conversion-receipt",
        "b00-source-hdf",
        "base-data-root",
        "train8-data-root",
        "train16-data-root",
        "candidate-audit",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "b02-data-root", "b02-manifest", "b02-conversion-receipt", "b02-source-hdf"
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--schedule-adapter-sha256", required=True)
    parser.add_argument("--b00-manifest-sha256", required=True)
    parser.add_argument("--b00-conversion-receipt-sha256", required=True)
    parser.add_argument("--b02-manifest-sha256")
    parser.add_argument("--b02-conversion-receipt-sha256")
    for name in ("base", "train8", "train16"):
        parser.add_argument(f"--{name}-manifest-sha256", required=True)
        parser.add_argument(f"--{name}-normalization-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        sha(args.parent_manifest) != MANIFEST_SHA
        or sha(args.history_inference_module) != INFERENCE_SHA
    ):
        raise ValueError("parent/inference source differs")
    args.history_k = 1
    if args.b00_manifest != args.b00_data_root / "manifest.json":
        raise ValueError("b00 manifest must belong to the exclusive dataset view")
    b02_args = (
        args.b02_data_root, args.b02_manifest, args.b02_conversion_receipt,
        args.b02_source_hdf, args.b02_manifest_sha256,
        args.b02_conversion_receipt_sha256,
    )
    if args.arm == "D" and any(value is None for value in b02_args):
        raise ValueError("arm D requires the complete b02 train-only binding")
    if args.arm != "D" and any(value is not None for value in b02_args):
        raise ValueError("arms A/B/C must not expose b02 inputs")
    if args.arm == "D" and args.b02_manifest != args.b02_data_root / "manifest.json":
        raise ValueError("b02 manifest must belong to the exclusive dataset view")
    parent_order = json.loads(args.parent_order.read_text())
    if canonical_sha(parent_order) != ORDER_SHA:
        raise ValueError("actual P026 sampler order differs")
    if not args.schedule_adapter.is_file():
        raise ValueError("schedule adapter missing")
    validate_b00_view(args)
    if args.arm == "D":
        validate_b02_view(args)
    args.effective_data_identity = validate_training_roots(args)
    resource_path = args.resource_helper
    import importlib.util

    if sha(resource_path) != RESOURCE_SHA:
        raise ValueError("resource helper differs")
    spec = importlib.util.spec_from_file_location("p026_resource", resource_path)
    resource = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resource)
    history = resource.load(args.history_module, HISTORY_SHA, "p026_history")
    chunk = resource.load(args.history_objective, OBJECTIVE_SHA, "p026_chunk")
    helper = resource.load(args.source_root / "scripts/probe_fcp019_gradient_alignment.py", P019_SHA, "p026_p019")
    p020 = resource.load(args.comparison_helper, P020_SHA, "p026_p020")
    accumulation = resource.load(args.accumulation_helper, P015_SHA, "p026_p015")
    diagnostic = resource.load(args.diagnostic_script, P014_SHA, "p064_p014")
    if sha(args.config) != diagnostic.CONFIG_SHA or sha(args.candidate_audit) != AUDIT_SHA:
        raise ValueError("configuration or train44 audit identity differs")
    objective = diagnostic.load_objective(args.source_root)
    if not args.execute:
        print("FC_P064_PREPARATION_ONLY_NO_GPU")
        return
    execute(
        args,
        resource,
        history,
        chunk,
        helper,
        p020,
        accumulation,
        diagnostic,
        objective,
    )


if __name__ == "__main__":
    main()
