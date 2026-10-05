from __future__ import annotations

import subprocess
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts/run_fcp008_posteval_spark.sh"


def test_runner_is_thin_and_binds_epoch_zero_identity_and_protocol() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "checkpoint_relative_directory" in text
    assert 'checkpoint_relative" == candidate_build/candidate' in text
    assert "FCP008_FORMAL_APPROVAL_SHA256" in text
    assert "formal_evaluation_approval.json" in text
    assert "posteval_chain_receipt_sha256" in text
    assert 'numerical_commit="7216214b545fbbd50b2fb5ed866f231039b06b18"' in text
    assert "--entrypoint python" in text and "python - <<" not in text
    assert "--segment-stride 25 --evaluation-batch-size 4" in text
    assert "--segment-stride 1 --evaluation-batch-size 8" in text
    assert "--checkpoint-epoch 0" in text
    assert text.count("--allow-calibrated-epoch-zero") == 5
    assert "--expected-calibrated-state-sha256" in text
    assert "frozen_test" not in " ".join(
        line for line in text.splitlines() if "False" not in line and "false" not in line
    )


def test_runner_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
