import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_predeclaration_is_train_only_and_keeps_formal_gates():
    payload=json.loads((ROOT/"artifacts/tandem_cylinders/dynamic_train8_fno_finetune_predeclared_20261003.json").read_text())
    assert payload["data"]["base_train_stride"] == 20
    assert payload["data"]["additional_train_stride"] == 2
    assert payload["data"]["frozen_test_access"] is False
    assert payload["training_variants"]["h20_parent"] == {"epochs":6,"rollout_steps":20}
    assert payload["training_variants"]["h50_parent"] == {"epochs":4,"rollout_steps":50}
    assert payload["training_variants"]["h100_parent"] == {
        "epochs": 2, "rollout_steps": 100, "batch_size": 2
    }
    assert payload["training_variants"]["validation_rollout_steps"] == 100
    assert payload["training_variants"]["teacher_forcing_ratio"] == 0.0
    assert payload["parent_selection"]["mutable_best_directory_forbidden"] is True
    assert payload["data"]["base"].endswith("full40_dev30_v1")
    assert payload["data"]["base_manifest_sha256"].startswith("5213c7bb")
    assert payload["training_variants"]["allocator_fraction"] == {
        "h20": 0.25, "h50": 0.45, "h100": 0.45
    }


def test_config_and_runner_are_immutable_parent_guarded():
    config=(ROOT/"conf/tandem_fno_dynamic_train8_h20.yaml").read_text()
    config50=(ROOT/"conf/tandem_fno_dynamic_train8_h50.yaml").read_text()
    config100=(ROOT/"conf/tandem_fno_dynamic_train8_h100.yaml").read_text()
    runner=(ROOT/"scripts/run_dynamic_train8_fno_finetune_spark.sh").read_text()
    assert "additional_train_stride: 2" in config
    assert "train_stride: 20" in config
    assert "validation_rollout_steps: 100" in config
    assert "teacher_forcing_start: 0.0" in config
    assert "rollout_steps: 50" in config50
    assert "epochs: 4" in config50
    assert "rollout_steps: 100" in config100
    assert "batch_size: 2" in config100
    assert "cp --reflink=auto" in runner
    assert "immutable parent copy SHA differs" in runner
    assert "parent_copy,dst=/workspace/parent,readonly" in runner
    assert 'immutable.get("model_sha256")' in runner
    assert 'immutable.get("state_sha256")' in runner
    assert "DYNAMIC_TRAIN8_FNO_FINETUNE_DRY_RUN_READY" in runner
    assert "EXECUTE_REVIEWED_DYNAMIC_TRAIN8_FNO_FINETUNE" in runner
    assert "--min-free-gib 20" in runner
    assert '--memory 90g' in runner
    assert 'h20) config="tandem_fno_dynamic_train8_h20"; fraction=.25' in runner
    assert 'h50) config="tandem_fno_dynamic_train8_h50"; fraction=.45' in runner
    assert 'h100) config="tandem_fno_dynamic_train8_h100"; fraction=.45' in runner
    assert runner.count("--allocator-fraction .15") == 2
    assert "--allocator-fraction .12" not in runner
    assert '$(basename "$parent")" != best' in runner
    assert '--env "USER=$(id -un)"' in runner
    assert 'tandem_cylinders_matched_start_full40_dev30_v1' in runner
    assert 'audit_dev30_validation_diagnostic.py' in runner
    assert 'audit_full40_dynamic6_fno.py' in runner
    assert '--segment-stride 1 --evaluation-batch-size 8' in runner
    assert '--data /workspace/devdata --normalization-data /workspace/devdata' in runner
    assert '--data /workspace/dynamic --normalization-data /workspace/devdata' in runner
    assert "DYNAMIC6_REAL_HDF_PREFLIGHT_PASS" in runner
    assert "dynamic6 requires six regular validation HDF files" in runner
    assert "handle['state'].shape!=(201,3,128,256)" in runner
    assert 'artifacts/tandem_fno_dynamic_train8_${horizon}_${run_id}' in runner
    assert 'training.max_train_batches=1 training.max_validation_batches=1' in runner
    assert 'DYNAMIC_TRAIN8_FNO_PROBE_SEED:-20261003' in runner
    assert 'training.seed="$probe_seed"' in runner
    assert 'tandem_cylinders_matched_start_full40_v1"' not in runner
    assert "frozen" not in " ".join(line for line in runner.splitlines() if "mount" in line)


def test_hydra_composition_keeps_variant_memory_and_batch_contracts():
    hydra = pytest.importorskip("hydra")
    compose = hydra.compose
    initialize_config_dir = hydra.initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        h20 = compose(config_name="tandem_fno_dynamic_train8_h20")
        h50 = compose(config_name="tandem_fno_dynamic_train8_h50")
        h100 = compose(config_name="tandem_fno_dynamic_train8_h100")
    assert float(h20.training.gpu_memory_fraction) == 0.25
    assert int(h20.data.prefetch_factor) == 0
    assert int(h20.training.rollout_steps) == 20
    assert int(h20.training.batch_size) == 4
    assert float(h50.training.gpu_memory_fraction) == 0.45
    assert int(h50.data.prefetch_factor) == 0
    assert int(h50.training.rollout_steps) == 50
    assert int(h50.training.batch_size) == 4
    assert float(h100.training.gpu_memory_fraction) == 0.45
    assert int(h100.data.prefetch_factor) == 0
    assert int(h100.training.rollout_steps) == 100
    assert int(h100.training.validation_rollout_steps) == 100
    assert int(h100.training.batch_size) == 2
    assert int(h100.training.epochs) == 2
    assert float(h100.training.teacher_forcing_start) == 0.0
    assert float(h100.training.teacher_forcing_end) == 0.0
