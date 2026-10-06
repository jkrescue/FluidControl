# Lead approval: fixed training-cache coverage analysis

2026-10-06. Preparation protocol: `/tmp/train-cache-coverage/PROTOCOL.md`.
Parent evidence: P027 result SHA256
`7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366`.

Question: which existing action-history categories and physical durations are
represented by the cached 44 training trajectories, and how do truth-conditioned
force errors differ from free rollout and persistence within that same cache?

Approve one JSON-only analysis after independent CPU review accepts the exact
source SHA256 `1fbd532caae13240f62a45922e499f02cf6402ad1c860828cd1a100350288317`.
Use all 44 cases, origin51 and offsets1–5. No new model/HDF access, inference,
training, CFD, threshold tuning or held-out data. Report all group members;
exact stored action signs are descriptive and may include rounding artifacts.
Do not infer state OOD from elapsed time or compare old TF32 numbers as a
matched improvement against the new no-TF32 replay.

Execution budget: one CPU, 1 GiB memory, no swap, 60-second runtime; existing
physical MemAvailable must exceed 20 GiB before launch. Unique unit
`fluid-control-train-cache-coverage-20261006.service`; exclusive output
`artifacts/train_cache_coverage_20261006.json`. Do not overwrite or auto-retry.

Root owns execution. Independent reviewer: closed_loop_readiness_review.
Decision: use the output to identify missing diagnostic coverage and choose
one justified next intervention; no model acceptance follows from this analysis.
