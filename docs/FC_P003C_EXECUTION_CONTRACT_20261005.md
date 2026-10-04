# FC-P003C execution contract

Status: implementation and CPU dry-run only. This document does not authorize
GPU training, post-evaluation, PPO, or CFD.

The unique candidate root is
`artifacts/tandem_fno_true_state_paired_step_lambda10_20261005`. It starts from
the immutable Main-e2 model/state and preserves the FC-P003B official FNO,
train20+train8+train16 regular data and order, dynamic8 two-pass pair order,
16 interleaved positions, seed 20261003, batch 1, H100, two epochs, lambda 10,
force weights `[1,1,4,1]/7`, optimizer and clipping. The sole scientific change
is `paired_objective_kind: true_state_step_force` with time chunk 10.

`scripts/run_fcp003c_training_spark.sh --dry-run` composes the child Hydra
configuration with the pinned project image containing official PhysicsNeMo
2.2.2 libraries, then independently checks the resolved architecture, data and
training fields. `--execute` is fail-closed unless a separately SHA-pinned Lead
approval exists and both layers of the mixed-loss probe agree: the completion
receipt is `...TECHNICAL_PROBE_COMPLETE`, its bound result is
`...TECHNICAL_PROBE_PASS`, the parent did not change, exactly one scratch
optimizer step occurred, and no candidate/validation/frozen data were written
or read. A source snapshot and parent pair are copied read-only before launch.

`scripts/run_fcp003c_posteval_spark.sh` defaults to dry-run. Its fixed sequence
is validation10 H1/H10/H50/H100 stride25 batch4, endpoint audit against the
full40 contract, dynamic6 H1/H10/H50/H100 stride1 batch8, force-window6, and the
unchanged development gate. Validation inference mounts dev30 at
`/workspace/devdata`; the endpoint CPU audit runs in a separate container and
mounts full40 validation, manifest and normalization at that same
`/workspace/devdata` runtime name. Matching the report's recorded data path is
required by the reviewed endpoint auditor and prevents the historical path
mismatch. Each stage has a checkpoint-bound
SHA receipt and supports strict resume without repeating a completed, validated
GPU stage. The final receipt hashes the complete bundle, records frozen/PPO as
false, and retains scientific FAIL as a valid completed evaluation outcome.

The candidate lineage auditor recomputes the actual dataset and normalization
hashes, resolved configuration, source snapshot, two epoch histories, exact
dynamic pair multiplicities and positions, optimizer-step counts, per-channel
loss evidence, best-selection epoch, and best/checkpoint archive payload
equivalence. Passing this software contract is not a surrogate scientific PASS.
