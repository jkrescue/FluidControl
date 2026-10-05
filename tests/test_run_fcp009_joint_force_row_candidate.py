from __future__ import annotations

import subprocess
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts/run_fcp009_joint_force_row_candidate_spark.sh"


def test_runner_binds_bounded_cache_candidate_contract() -> None:
    text = RUNNER.read_text()
    assert "2c6be38021425ec8a7e18f6f967135155e2d6198" in text
    assert "dbe75763f8c480feefc54cb657f53848c24055e88ecc70a9a0aaaad544d81ecf" in text
    assert "931fcd2ddd6901ddfbb3ecdbfe9774d7b1e5fe6d17479c87c44ccaf51ff2b0bc" in text
    assert "timeout --signal=TERM --kill-after=30s 5m" in text
    assert "--min-free-gib 20 --allocator-fraction .15" in text
    assert "--batch-size" not in text
    assert "/validation" not in text and "/frozen_test" not in text
    assert "native_h1_sanity" in text and "formal_evaluation_authorized" in text


def test_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
