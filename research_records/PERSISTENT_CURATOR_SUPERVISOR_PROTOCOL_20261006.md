# External CPU supervisor — pending execution review

This supplements the existing REPLAY_PREPARATION.md; it does not authorize replay.
Old sampler, original20exports/NPZs and the reviewed replay/module bytes are unchanged.
`supervise_curator_replay.py` pins/imports only the stdlib helpers from reviewed
canonical probe_k1_uma_inference.py SHA56c174...5e74. No torch/CUDA/model import
occurs in that helper import. It reuses physical memory checks and exact child
process-group TERM/KILL cleanup; no GPU operation is called.

Separate approval schema (Root supplies actual authorization, no draft is approved):

    status: PERSISTENT_CURATOR_REPLAY_APPROVED
    execution_authorized: true
    supervisor_sha256: exact new script SHA
    replay_script: absolute reviewed replay_persistent_curator.py path
    sampling_module: absolute reviewed persistent_curator_frame.py path
    output: /workspace/fluid_control/artifacts/persistent_curator_equivalence_20261006
    supervision_output: /workspace/fluid_control/artifacts/persistent_curator_supervision_20261006

Both output directories must be new. Only generated replay/supervision artifacts
are written. Script uses fixed replay/module SHAs and checks approval byte SHA,
own source SHA, exact output scope and same pinned Curator interpreter.

Required actual user unit: MemoryMax=4G, MemorySwapMax=0, CPUQuota=100%,
RuntimeMaxSec=330, KillMode=control-group, CUDA_VISIBLE_DEVICES empty.
The process verifies actual cgroup memory/swap/CPU quota before starting child.
Startup MemAvailable>=50GiB, runtime>=22GiB sampled0.5s by external parent;
300s inner deadline. Signal/read/child/result failure invokes owned process-group
cleanup. systemd330s control-group backstop covers supervisor death. Record actual
unit and source hashes. No unit or replay has been launched by preparation.

Suggested later approved invocation, inside that reviewed unit only:

    .venv-curator-py312/bin/python supervise_curator_replay.py --approval <actual-json> --approval-sha256 <actual-sha> --execute

Current tiny synthetic regression:7PASS0.72s (three sampler equivalence-mechanics
tests plus four supervisor contract tests). This is not real Curator frame replay.
