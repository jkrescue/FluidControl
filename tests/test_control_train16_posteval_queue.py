"""Static fail-closed checks for the two-branch post-evaluation queue."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_control_train16_posteval_queue_spark.sh"


def test_queue_is_main_first_and_runs_all_reviewed_suites() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    main = 'evaluate_candidate main "$main" control_train16_h100'
    balanced = (
        'evaluate_candidate balanced "$balanced" '
        "control_train16_h100_lift_balanced"
    )
    assert source.index(main) < source.index(balanced)
    assert "--horizons 1 10 50 100" in source
    assert "audit_full40_validation_gate.py" in source
    assert "audit_full40_dynamic6_fno.py" in source
    assert "diagnose_fno_force_window.py" in source
    assert "audit_dynamic_fno_development_gates.py" in source


def test_queue_binds_snapshot_checkpoint_and_never_launches_ppo() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'source="$candidate/source_snapshot"' in source
    assert 'checkpoint="$candidate/best"' in source
    assert "audit_dynamic_fno_candidate_lineage.py" in source
    assert "worker_transfer_complete.json" in source
    assert '"ppo_auto_launched":False' in source
    assert "train_full40_hydrogym_ppo" not in source
    assert "--allocator-fraction .15" in source
    assert "frozen_test" not in "\n".join(
        line for line in source.splitlines() if "--mount" in line
    )
