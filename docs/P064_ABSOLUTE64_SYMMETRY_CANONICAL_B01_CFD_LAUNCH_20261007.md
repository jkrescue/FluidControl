# Absolute64 canonical b01 CFD launch and terminal identity

This is the immutable execution record for FC-E111, not a reusable approval.

- Approval: `docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json`, SHA `d6303082b0f97087961a9171505a2b4289623eb5f634f2502ac4f7d65639143b`.
- Driver: `artifacts/p064_absolute64_canonical_b01_cfd_source_20261007_immutable/run_p064_absolute64_symmetry_canonical_b01_cfd.py`, SHA `037f3c959d6fa2f313c4a964d00d263a4587d37b78b42cfa7300b10f286101c9`.
- Unit: `fluid-control-p064-absolute64-symmetry-canonical-b01-cfd-20261007.service`.
- Invocation: `6fa0baeb90034371b3b49f3d8d51192a`; terminal `MainPID=0`, `Result=success`, `ExecMainStatus=0`.
- Output: `artifacts/p064_absolute64_symmetry_canonical_b01_cfd_20261007`.
- Result SHA: `5fa8f47bb3a4939b6b8377f5ccd8c178dec560220929e36e48b8875e90a7504e`.
- Progress SHA: `559c782a10c036fa9c120c84f99cdd35fcb48dcbfed2b8676c00ff8a88df96b3`.

The exact `ExecStart` was:

```text
/workspace/fluid_control/.venv-curator-py312/bin/python /workspace/fluid_control/artifacts/p064_absolute64_canonical_b01_cfd_source_20261007_immutable/run_p064_absolute64_symmetry_canonical_b01_cfd.py --spec /workspace/fluid_control/docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json --spec-sha256 d6303082b0f97087961a9171505a2b4289623eb5f634f2502ac4f7d65639143b --execute
```

The transient unit bound `MemoryMax=8G`, `MemorySwapMax=0`, `CPUQuota=400%`,
`RuntimeMaxSec=3750`, `TimeoutStopSec=120`, `KillMode=control-group` and
`OOMPolicy=stop`. The driver retained two owned solver containers at8GiB,
no-swap and2 CPUs each, the50GiB startup /22GiB runtime Available guards,
and the20GiB physical reserve.

Independent terminal evidence is in
`docs/P064_ABSOLUTE64_B01_CFD_TERMINAL_REVIEW_20261007.md`, SHA
`17d9328babbbbbe23f1c4bd830e7347626e1cdd055087bbec7ffac0dbecb4bbe`.
The independent raw audit is invocation `b4b0efda17754ce092d7bf7c01d393a1`;
receipt `artifacts/p064_absolute64_b01_raw_audit_20261007/stdout.jsonl`, SHA
`d114a326919c1f2abf4b1a35ab51f3de09e33b43eac5de821d776f8fd126dad3`.
A separate producer-side read-only checker was mistakenly also executed under
invocation `6a6b40f61ec44760a0c3c795750d2a5b`; it did not rerun the policy, model or CFD.

All six original physical windows pass, but the absolute64 prediction-selection
FAIL remains and B is not silently replaced. This record grants no retry, PPO,
new CFD execution, threshold change or broader generalization claim.
