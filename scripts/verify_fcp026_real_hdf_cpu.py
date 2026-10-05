"""Six real train windows: official-reader CPU integration, not full-data audit."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from fluid_control.tandem_datapipe import TandemRolloutDataset, HDF5Reader
import p026_state_history as history

ADAPTER_SHA = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"
LOADER_SHA = "c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
FAMILIES = [
    (
        "base",
        20,
        20,
        "matched_start_acquisition_train_b00_zero",
        "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2",
    ),
    (
        "train8",
        2,
        4,
        "dynamic_train8_b00_prbs",
        "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35",
    ),
    (
        "train16",
        2,
        4,
        "direct_cfd_directppo2048_v1_env0_ep0009_b00",
        "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b",
    ),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_sample(left, right):
    if set(left.keys()) != set(right.keys()):
        raise ValueError("sample keys changed")
    for key in left.keys():
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        if not torch.isfinite(left[key]).all():
            raise ValueError("nonfinite original sample")


def main():
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only required")
    torch.set_num_threads(2)
    if (
        sha(history.__file__) != ADAPTER_SHA
        or sha(inspect.getfile(TandemRolloutDataset)) != LOADER_SHA
    ):
        raise ValueError("adapter/base-loader source differs")
    rows = []
    for family, stride, warm_start, case, manifest_sha in FAMILIES:
        root = Path("/workspace") / family
        if (
            sha(root / "manifest.json") != manifest_sha
            or sha(root / "normalization.json") != NORMALIZATION_SHA
        ):
            raise ValueError("manifest/normalization identity differs")
        if any(
            (root / split).exists() for split in ("validation", "test", "frozen_test")
        ):
            raise ValueError("smoke view must expose train only")
        base = TandemRolloutDataset(
            root, "train", 100, stride=stride, num_workers=1, force_indices=(0, 1, 2, 3)
        )
        try:
            matches = [i for i, path in enumerate(base.paths) if path.stem == case]
            if len(matches) != 1:
                raise ValueError("unique representative case missing")
            file_index = matches[0]
            for start in (0, warm_start):
                index = base.index.index((file_index, start))
                original, metadata = base[index]
                sample1, meta1, hist1 = history.HistoryWindowAdapter(base, 1)[index]
                sample4, meta4, hist4 = history.HistoryWindowAdapter(base, 4)[index]
                equal_sample(original, sample1)
                equal_sample(original, sample4)
                if metadata != meta1 or metadata != meta4:
                    raise ValueError("metadata changed")
                expected_indices = (
                    [0, 0, 0, 0] if start == 0 else list(range(start - 3, start + 1))
                )
                expected_padding = (
                    [True, True, True, False] if start == 0 else [False] * 4
                )
                if (
                    hist4["metadata"]["frame_indices"] != expected_indices
                    or hist4["metadata"]["padding_mask"] != expected_padding
                ):
                    raise ValueError("frame/padding identities differ")
                reader = base._reader(file_index)
                if not isinstance(reader, HDF5Reader):
                    raise ValueError("not official HDF5Reader")
                for offset, frame_index in enumerate(expected_indices):
                    frame, _ = reader[frame_index]
                    expected_state = (
                        (frame["state"].float() - base.state_mean) / base.state_std
                    ) * original["mask"]
                    expected_action = (
                        frame["omega"].float().reshape(1) / base.action_scale
                    )
                    torch.testing.assert_close(
                        hist4["states"][offset], expected_state, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        hist4["actions"][offset], expected_action, rtol=0, atol=0
                    )
                x1 = history.build_input(
                    hist1["states"],
                    original["mask"],
                    hist1["actions"],
                    original["omega"][1],
                )
                x4 = history.build_input(
                    hist4["states"],
                    original["mask"],
                    hist4["actions"],
                    original["omega"][1],
                )
                height, width = original["mask"].shape[-2:]
                legacy = torch.cat(
                    (
                        original["state"],
                        original["mask"],
                        original["omega"][0].reshape(1, 1, 1).expand(1, height, width),
                        original["omega"][1].reshape(1, 1, 1).expand(1, height, width),
                    )
                )
                torch.testing.assert_close(x1, legacy, rtol=0, atol=0)
                torch.testing.assert_close(x4[9:12], original["state"], rtol=0, atol=0)
                torch.testing.assert_close(x4[16:18], legacy[4:6], rtol=0, atol=0)
                if not torch.isfinite(x1).all() or not torch.isfinite(x4).all():
                    raise ValueError("nonfinite history input")
                rows.append(
                    dict(
                        family=family,
                        case=case,
                        start=start,
                        original_stride=stride,
                        dataset_index=index,
                        history_metadata=hist4["metadata"],
                        k1_shape=list(x1.shape),
                        k4_shape=list(x4.shape),
                        exact_original_samples_targets_metadata=True,
                        exact_official_reader_history=True,
                        exact_k1_legacy_input=True,
                    )
                )
        finally:
            base.close()
    print(
        json.dumps(
            dict(
                status="FC_P026_SIX_TRAIN_WINDOWS_CPU_INTEGRATION_NOT_ADMISSION",
                rows=rows,
                cuda_available=False,
                source_sha256=dict(
                    adapter=sha(history.__file__),
                    loader=sha(inspect.getfile(TandemRolloutDataset)),
                    official_hdf_reader=sha(inspect.getfile(HDF5Reader)),
                    script=sha(__file__),
                ),
                action_semantics=history.ACTION_SEMANTICS,
                all44_coverage_audited=False,
                full_field_hash_performed=False,
                data_modified=False,
                model_created=False,
                heldout_accessed=False,
                scientific_admission=False,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
