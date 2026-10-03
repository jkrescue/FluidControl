from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_quickscreen_configs_only_override_epochs_and_output() -> None:
    one = yaml.safe_load(
        (ROOT / "conf/tandem_fno_full40_quickscreen_onestep.yaml").read_text(
            encoding="utf-8"
        )
    )
    h20 = yaml.safe_load(
        (ROOT / "conf/tandem_fno_full40_quickscreen_h20.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert one["defaults"][0] == "tandem_fno_full40_onestep"
    assert one["training"] == {"epochs": 10}
    assert h20["defaults"][0] == "tandem_fno_full40_h20"
    assert h20["training"] == {"epochs": 5}
    forbidden = {"model", "data", "seed", "batch_size", "gpu_memory_fraction"}
    assert forbidden.isdisjoint(one)
    assert forbidden.isdisjoint(h20)


def test_quickscreen_runner_is_frozen_blind_guarded_and_nonformal() -> None:
    text = (ROOT / "scripts/run_full40_dev30_quickscreen_spark.sh").read_text(
        encoding="utf-8"
    )
    assert 'src=$root,dst=/workspace' not in text
    for mount in ("scripts", "src", "conf"):
        assert f"src=$root/{mount},dst=/workspace/{mount},readonly" in text
    assert "src=$data_host,dst=/workspace/devdata,readonly" in text
    assert "matched_start_full40_v1/frozen" not in text
    assert "dst=/workspace/frozen" not in text
    assert "--min-free-gib 20" in text
    assert "--gpus device=0" in text
    assert "refusing existing quick-screen output" in text
    assert "FULL40_DEV30_QUICKSCREEN_PARENT_SHA256" in text
    assert "sha256sum" in text
    assert "list(range(1, 11))" in text
    assert "resolved config differs" in text
    assert "STAGE_CANDIDATE_ONLY" in text
    assert "tandem_fno_full40_quickscreen_onestep" in text
    assert "tandem_fno_full40_quickscreen_h20" in text
    assert "run_provenance.json" in text
    assert "dev30_manifest_sha256" in text
    assert "train_only_normalization_sha256" in text
    assert '"frozen_test_mounted_or_accessed": False' in text
    assert '"formal_gate_authorized": False' in text
    assert '"ppo_authorized": False' in text


def test_formal_configs_are_not_shortened() -> None:
    one = yaml.safe_load(
        (ROOT / "conf/tandem_fno_full40_onestep.yaml").read_text(encoding="utf-8")
    )
    h20 = yaml.safe_load(
        (ROOT / "conf/tandem_fno_full40_h20.yaml").read_text(encoding="utf-8")
    )
    assert one["training"]["epochs"] == 30
    assert h20["training"]["epochs"] == 10
