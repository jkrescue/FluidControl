"""Official CPU checkpoint round trip; ENGINEERING FIXTURES, never candidates."""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

HISTORY_SHA = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"
RESOURCE_SHA = "b806ded8258c787807e67ccb42b5166dbd06e36fc40eba5025d9bda0769eedc6"
OFFICIAL_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
MANIFEST_SHA = "91cc2c9a295a1ace5eadcffd8242234b93b73d14959bb902e17b59655f4acf13"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
TENSORS = {
    "flow": "89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb",
    "aerodynamic": "6f58aea89ecdde46bfaafac0f181d2603bbc96e3bba1faa7d8a818dcf6f7984d",
}
STATUS = "FC_P026_HISTORY_ENGINEERING_FIXTURE_NOT_CANDIDATE"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(path, digest, name):
    if sha(path) != digest:
        raise ValueError("pinned source differs: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(role, k, parent_tensor, mapped_tensor):
    channels = 6 if k == 1 else 18
    return dict(
        status=STATUS,
        checkpoint_epoch=1,
        role=role,
        history_k=k,
        model_in_channels=channels,
        official_appended_coordinates=2,
        parent_tensor_sha256=parent_tensor,
        mapped_tensor_sha256=mapped_tensor,
        history_adapter_sha256=HISTORY_SHA,
        optimizer_created=False,
        training_performed=False,
        candidate=False,
        scientific_admission=False,
        action_semantics="stored_prescribed_action_samples_not_exact_nominal_time_commands",
    )


def validate_fixture(metadata_value, role, k, channels):
    if k not in (1, 4) or channels != (6 if k == 1 else 18):
        raise ValueError("legacy/history configuration mismatch")
    required = dict(
        status=STATUS,
        checkpoint_epoch=1,
        role=role,
        history_k=k,
        model_in_channels=channels,
        official_appended_coordinates=2,
        history_adapter_sha256=HISTORY_SHA,
        optimizer_created=False,
        training_performed=False,
        candidate=False,
        scientific_admission=False,
    )
    if any(metadata_value.get(key) != value for key, value in required.items()):
        raise ValueError("fixture profile/epoch/shape metadata differs")


def compare_states(actual, expected):
    import torch

    if actual.keys() != expected.keys():
        raise ValueError("fresh state keys differ")
    for name, value in actual.items():
        target = expected[name]
        if (
            value.shape != target.shape
            or value.dtype != target.dtype
            or not torch.equal(value, target)
        ):
            raise ValueError("fresh tensor bytes differ: " + name)


def execute(args):
    import gc
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.models.fno import FNO
    from physicsnemo.utils import save_checkpoint, load_checkpoint
    from physicsnemo.distributed import DistributedManager

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only execution required")
    if sha(inspect.getfile(FNO)) != OFFICIAL_SHA:
        raise ValueError("official FNO source differs")
    save_source, load_path = map(
        Path, (inspect.getfile(save_checkpoint), inspect.getfile(load_checkpoint))
    )
    if save_source != load_path or sha(save_source) != CHECKPOINT_SHA:
        raise ValueError("official save/load implementation differs")
    if (
        sha(args.config) != CONFIG_SHA
        or sha(args.parent / "dual_model_manifest.json") != MANIFEST_SHA
    ):
        raise ValueError("base configuration/P018 manifest differs")
    history = load_source(args.history_module, HISTORY_SHA, "p026_history")
    resource = load_source(args.resource_helper, RESOURCE_SHA, "p026_resource")
    objective = load_source(
        args.source_root / "scripts/train_fcp013_independent_force_fno.py",
        "f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7",
        "p026_original",
    )
    sys.path[:0] = [str(args.source_root / "src"), str(args.source_root / "scripts")]
    from train_tandem_fno import build_model

    torch.set_num_threads(2)
    DistributedManager.initialize()
    cfg = OmegaConf.load(args.config)
    manifest = json.loads((args.parent / "dual_model_manifest.json").read_text())
    # Parent artifact hashes are pinned through the exact immutable manifest.
    for role in ("flow", "aerodynamic"):
        record = manifest[role]
        for key in ("model", "state"):
            if (
                sha(args.parent / role / record[key + "_file"])
                != record[key + "_sha256"]
            ):
                raise ValueError("parent file SHA differs")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=False)
    args.output_created_by_this_run = True
    records = []
    for role in ("flow", "aerodynamic"):
        parent = build_model(cfg)
        parent_metadata = {}
        epoch = load_checkpoint(
            args.parent / role,
            models=parent,
            metadata_dict=parent_metadata,
            device="cpu",
        )
        resource.validate_metadata(role, epoch, parent_metadata, manifest)
        if objective.tensor_state_sha256(parent) != TENSORS[role]:
            raise ValueError("separate parent tensor identity differs")
        for k in ((1,) if role == "flow" else (1, 4)):
            fixture_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            fixture_cfg.model.in_channels = 6 if k == 1 else 18
            model = build_model(fixture_cfg)
            expected = history.history_warmstart_state(
                parent.state_dict(), model.state_dict(), k
            )
            model.load_state_dict(expected, strict=True)
            mapped_sha = objective.tensor_state_sha256(model)
            info = metadata(role, k, TENSORS[role], mapped_sha)
            validate_fixture(info, role, k, fixture_cfg.model.in_channels)
            directory = args.output / ("engineering_" + role + "_k" + str(k))
            save_checkpoint(directory, models=model, epoch=1, metadata=info)
            del model
            gc.collect()
            fresh = build_model(fixture_cfg)
            reloaded_metadata = {}
            loaded_epoch = load_checkpoint(
                directory, models=fresh, metadata_dict=reloaded_metadata, device="cpu"
            )
            if loaded_epoch != 1 or reloaded_metadata != info:
                raise ValueError("official fresh metadata/epoch differs")
            validate_fixture(reloaded_metadata, role, k, fixture_cfg.model.in_channels)
            compare_states(fresh.state_dict(), expected)
            if objective.tensor_state_sha256(fresh) != mapped_sha:
                raise ValueError("fresh official tensor hash differs")
            legacy_rejected = None
            if k == 4:
                try:
                    validate_fixture(reloaded_metadata, role, 1, 6)
                except ValueError:
                    legacy_rejected = True
                if legacy_rejected is not True:
                    raise RuntimeError("K4 fixture accepted as legacyK1")
            if objective.tensor_state_sha256(parent) != TENSORS[role]:
                raise RuntimeError("parent changed")
            records.append(
                dict(
                    directory=directory.name,
                    role=role,
                    k=k,
                    metadata=info,
                    official_fresh_reload=True,
                    all_state_tensors_exact=True,
                    inherited_mapping_exact=True,
                    legacy_profile_rejected=legacy_rejected,
                    files={p.name: sha(p) for p in directory.iterdir() if p.is_file()},
                )
            )
            del fresh, expected
            gc.collect()
        del parent
        gc.collect()
    return dict(
        status="FC_P026_OFFICIAL_CPU_FIXTURES_VERIFIED_NOT_CANDIDATE",
        fixtures=records,
        required_official_image_id=IMAGE,
        actual_image_requires_external_inspect=True,
        official_source_sha256=OFFICIAL_SHA,
        checkpoint_source_sha256=sha(save_source),
        history_adapter_sha256=HISTORY_SHA,
        resource_helper_sha256=RESOURCE_SHA,
        script_sha256=sha(__file__),
        parent_manifest_sha256=MANIFEST_SHA,
        base_config_sha256=CONFIG_SHA,
        cuda_available=False,
        optimizer_created=False,
        training_performed=False,
        candidate_saved=False,
        engineering_fixtures_saved=True,
        scientific_admission=False,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "source-root",
        "history-module",
        "resource-helper",
        "config",
        "parent",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("separate CPU execution approval required")
    try:
        result = execute(args)
        with (args.output / "engineering_receipt.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2))
    except Exception as error:
        if (
            getattr(args, "output_created_by_this_run", False)
            and not (args.output / "engineering_failure.json").exists()
        ):
            (args.output / "engineering_failure.json").write_text(
                json.dumps(
                    dict(
                        status="FC_P026_CPU_FIXTURE_FAILURE_NOT_CANDIDATE",
                        error=repr(error),
                    ),
                    indent=2,
                )
            )
        raise


if __name__ == "__main__":
    main()
