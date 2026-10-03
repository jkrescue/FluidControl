import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_predeclaration_is_train_only_and_keeps_formal_gates():
    payload=json.loads((ROOT/"artifacts/tandem_cylinders/dynamic_train8_fno_finetune_predeclared_20261003.json").read_text())
    assert payload["data"]["base_train_stride"] == 20
    assert payload["data"]["additional_train_stride"] == 2
    assert payload["data"]["frozen_test_access"] is False
    assert payload["training_variants"]["h20_parent"] == {"epochs":6,"rollout_steps":20}
    assert payload["training_variants"]["h50_parent"] == {"epochs":4,"rollout_steps":50}
    assert payload["training_variants"]["validation_rollout_steps"] == 100
    assert payload["training_variants"]["teacher_forcing_ratio"] == 0.0
    assert payload["parent_selection"]["mutable_best_directory_forbidden"] is True


def test_config_and_runner_are_immutable_parent_guarded():
    config=(ROOT/"conf/tandem_fno_dynamic_train8_h20.yaml").read_text()
    config50=(ROOT/"conf/tandem_fno_dynamic_train8_h50.yaml").read_text()
    runner=(ROOT/"scripts/run_dynamic_train8_fno_finetune_spark.sh").read_text()
    assert "additional_train_stride: 2" in config
    assert "train_stride: 20" in config
    assert "validation_rollout_steps: 100" in config
    assert "teacher_forcing_start: 0.0" in config
    assert "rollout_steps: 50" in config50
    assert "epochs: 4" in config50
    assert "cp --reflink=auto" in runner
    assert "immutable parent copy SHA differs" in runner
    assert "parent_copy,dst=/workspace/parent,readonly" in runner
    assert "DYNAMIC_TRAIN8_FNO_FINETUNE_DRY_RUN_READY" in runner
    assert "EXECUTE_REVIEWED_DYNAMIC_TRAIN8_FNO_FINETUNE" in runner
    assert "--min-free-gib 20" in runner
    assert 'h50) config="tandem_fno_dynamic_train8_h50"' in runner
    assert "frozen" not in " ".join(line for line in runner.splitlines() if "mount" in line)
