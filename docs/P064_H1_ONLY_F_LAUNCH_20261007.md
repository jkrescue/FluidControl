# Candidate F actual launch — not a scientific result

Lead conditionally authorized one run after Sota independently accepted frozen
pending SHA20761112de38c736e8a5a2df623e04074e8f0c5ef754147911a91edb7526a489.
The condition was met before execution. [Approval](P064_H1_ONLY_F_TRAINING_APPROVAL_20261007.json)
SHA d157e311d70261511144c9fc22a27a468730b4ea2b5ea37705c773b43aea8ef2.

- Actual unit: `fluid-control-p064-h1-only-f-20261007.service`.
- Invocation: `31f1692d9f984b91a17e21a133426e99`; initial PID1781483.
- Started 2026-10-06 21:33:05 UTC; first actual training window at21:33:30.
- Exclusive output: `artifacts/p064_h1_only_candidate_f_20261007`.
- Startup Available123954233344 bytes; GPU process list empty before launch.
- Rehashed original434 sources plus3 new immutable files; actual frozen argv
  no-execute preflight and pinned official runtime imports passed.
- Type=exec, RemainAfterExit=yes, MemoryMax12GiB, swap0, CPUQuota800%,
  TasksMax2048, RuntimeMaxSec3660, TimeoutStopSec20, KillModecontrol-group,
  OOMPolicystop; original worker3600s and physical Available50/22GiB guards.

Executed immutable directory: `artifacts/p064_h1_only_source_20261007_immutable`.
Runner SHA10c6c690ff2b7155dea9134d220cf9568606260a4d58e8e695b16bdbe2c796a6;
objective SHA1cb9ea57f873a167352a962037a09b777cbf3fab0a4d83078c1fba67dab7f716;
isolated consumer SHAf326cffe5d89d5d336dcc10c9a90fed13e0d61470947d2bc54d79cf2fe761131.
Canonical consumer is named `src/fluid_control/dual_fno_h1_only.py` to avoid
changing the existing default loader; the bytes are identical. Training still
loads K1 with original83ac loader; the new consumer is for F after completion.

Five CPU tests passed, independently repeated by Sota. Canonical test changes
only its consumer-path fallback. The scientific change is pureH1 backward;
mixed20 forward, frozen-flow AR states and original diagnostic total remain.
H1 gradient contribution is2x B's H1 component, AR contribution0; clipping may
change. No E auxiliary, weight scan, budget extension or intermediate selection.
Recovery owns actual monitoring and independent terminal review. This document
records launch only, not completion, prediction improvement or admission.
