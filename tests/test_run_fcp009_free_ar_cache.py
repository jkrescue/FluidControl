from __future__ import annotations

import subprocess
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts/run_fcp009_free_ar_cache_spark.sh"


def test_runner_binds_cache_only_contract() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "02c0c99f7b8b48657269595993b1f3dba2a58bf6" in text
    assert "a2c83846714b3d5575840dcd89c0c8e955fc40f217dad011f3e74d277807bfdf" in text
    assert "22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff" in text
    assert 'git archive "$source_commit" -- src scripts conf' in text
    assert "--batch-size 4 --workers 1" in text
    assert "--min-free-gib 20 --allocator-fraction .15" in text
    assert "timeout --signal=TERM --kill-after=30s 25m" in text
    assert "candidate_saved" in text and "formal_evaluation_authorized" in text
    assert "/validation" not in text and "/frozen_test" not in text


def test_runner_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
