"""Static fail-closed checks for the two-branch post-evaluation queue."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_control_train16_posteval_queue_spark.sh"


def test_queue_runs_main_and_marks_balanced_external_takeover() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    main = 'evaluate_candidate main "$main" control_train16_h100'
    assert main in source
    assert "BALANCED_POSTEVAL_EXTERNAL_WORKER_TAKEOVER" in source
    assert "evaluate_candidate balanced" not in source
    assert "--horizons 1 10 50 100" in source
    assert "audit_full40_validation_gate.py" in source
    assert "audit_full40_dynamic6_fno.py" in source
    assert "diagnose_fno_force_window.py" in source
    assert "audit_dynamic_fno_development_gates.py" in source
    assert "--candidate-kind dev30_free_ar_development" in source


def test_queue_binds_snapshot_checkpoint_and_never_launches_ppo() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'source="$candidate/source_snapshot"' in source
    assert 'checkpoint="$candidate/best"' in source
    assert "audit_dynamic_fno_candidate_lineage.py" in source
    assert "external_worker_takeover" in source
    assert '"ppo_auto_launched":False' in source
    assert "train_full40_hydrogym_ppo" not in source
    assert "--allocator-fraction .15" in source
    assert "frozen_test" not in "\n".join(
        line for line in source.splitlines() if "--mount" in line
    )
