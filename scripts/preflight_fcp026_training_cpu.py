"""Official CPU DataPipe inventory/sampler audit; no field reads or training."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.trainer) != args.trainer_sha256:
        raise ValueError("trainer source differs")
    spec = importlib.util.spec_from_file_location("p026_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    sys.path[:0] = [str(args.source_root / "src"), str(args.source_root / "scripts")]
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader, MultiDataset
    from physicsnemo.distributed import DistributedManager
    from fluid_control.tandem_datapipe import TandemRolloutDataset

    cfg = OmegaConf.load(args.config)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only audit required")
    DistributedManager.initialize()
    if (
        sha(args.source_root / "src/fluid_control/tandem_datapipe.py")
        != "c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae"
    ):
        raise ValueError("base loader source differs")
    if (
        sha(args.config)
        != "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
    ):
        raise ValueError("config differs")
    roots = [Path(cfg.data.root), *[Path(x) for x in cfg.data.additional_train_roots]]
    expected = [
        "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2",
        "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35",
        "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b",
    ]
    children = []
    try:
        for root, digest, stride in zip(roots, expected, [20, 2, 2], strict=True):
            if (
                sha(root / "manifest.json") != digest
                or sha(root / "normalization.json")
                != "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
            ):
                raise ValueError("data metadata differs")
            if any((root / s).exists() for s in ("validation", "test", "frozen_test")):
                raise ValueError("nontrain exposure")
            children.append(
                TandemRolloutDataset(
                    root,
                    "train",
                    100,
                    stride=stride,
                    num_workers=1,
                    force_indices=(0, 1, 2, 3),
                )
            )
        dataset = MultiDataset(*children, output_strict=True)
        inventory = trainer.inventory(dataset)
        # Do not iterate batches: only official sampler over metadata-built indices.
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            prefetch_factor=0,
            use_streams=False,
            seed=20261003,
        )
        order = list(iter(loader.sampler))
        order_sha = hashlib.sha256(
            json.dumps(order, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            order_sha != trainer.ORDER_SHA
            or len(order) != 1368
            or sorted(order) != list(range(1368))
        ):
            raise ValueError("original order differs")
        identities = []
        for family, child in enumerate(children):
            for file_index, start in child.index:
                identities.append(
                    dict(
                        dataset_index=family,
                        case=child.paths[file_index].stem,
                        start=int(start),
                        warm_history=start >= 3,
                        split="train",
                    )
                )
        output = dict(
            status="FC_P026_CPU_TRAIN_INVENTORY_ORDER_VERIFIED_NOT_TRAINING_APPROVAL",
            inventory=inventory,
            sampler_order_sha256=order_sha,
            ordered_identities=[identities[i] for i in order],
            trainer_sha256=args.trainer_sha256,
            source_sha256=sha(
                args.source_root / "src/fluid_control/tandem_datapipe.py"
            ),
            protocols={
                str(k): dict(
                    effective_protocol=trainer.protocol(k),
                    canonical_sha256=trainer.canonical_sha(trainer.protocol(k)),
                )
                for k in (1, 4)
            },
            cuda_available=False,
            field_batches_iterated=False,
            full_hdf_hash_performed=False,
            training_performed=False,
        )
        with args.output.open("x") as stream:
            json.dump(output, stream, indent=2)
        print(
            json.dumps(
                dict(
                    status=output["status"],
                    inventory=inventory,
                    sampler_order_sha256=order_sha,
                )
            )
        )
    finally:
        for child in children:
            child.close()


if __name__ == "__main__":
    main()
