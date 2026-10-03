# H20 rear-drag FNO and v4 real-CFD data milestone

Date: 2026-10-03 UTC. Canonical project: `fluid_control/physicsnemo_control`
on primary DGX Spark `.97`. This note records completed evidence, not a
closed-loop control result.

## Official PhysicsNeMo FNO combination: completed but not Gate B

The predeclared H20 rollout plus rear-cylinder-drag-weighted loss ablation
completed all ten epochs on compute-only `.78`, with model and history
checksummed back to the primary Spark. The existing official PhysicsNeMo FNO
architecture, v3 26/4/5 real-CFD split and same train-only normalization
were retained. The finalizer compared validation before opening the frozen
five-case test. The checkpoint's validation 100-step total-drag NRMSE was
13.414%, improving on its H20 parent 15.522%; it therefore qualified for
one frozen Gate-B evaluation.

| Evaluation | 100-step total-drag NRMSE | Criterion |
| --- | ---: | ---: |
| Validation, four cases | 13.414% | improvement over 15.522% parent |
| Frozen CFD test, five cases | 15.540% | <=10% |
| Independent shedding phase | 10.812% | <=10% |

The full Gate-B report is
`artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/gate_b_audit.json`.
It reports `GATE_B_NEEDS_MULTISTEP_RETRAINING`. Dynamics remained finite and
the action perturbation checks passed; the two failed checks are the frozen
and independent-phase 100-step drag NRMSE. This run improves over the parent
on the frozen five-case metric (17.769% to 15.540%), but **does not** meet
the control-horizon accuracy requirement. The checkpoint
`best/FNO.0.9.mdlus` has SHA-256
`0e31d5e40bf8b805f64ebaf553af35aee62e6030e2aa042da8bcc6a5551fb226`.
Worker minimum observed unified-memory availability during training was
100.369 GiB; the primary finalizer's GPU guard completed normally.

The predeclared four-action validation CFD ranking test is the next
decision-fidelity check. Do not treat the improved validation error as proof
that the model chooses a beneficial or lift-safe action.

## v4 signed-action acquisition: curated and split-audited

Two new *real* OpenFOAM signed-pulse trajectories from the `t=82` training
restart passed source/force/solver QC and official PhysicsNeMo Curator
processing. The `control_gap_v4` data profile contains **28 training, four
validation, five frozen-test** trajectories, each with 801 frames. Existing
v3 validation and frozen-test HDF5 files are hard links, hence byte-identical;
only the training split grows from 26 to 28. Training-only normalization was
recomputed from the 28 training trajectories. Manifest SHA-256:
`903dbf80e0c4b831698ece27775f3fdb65a204ad122cc7e28fef8320bcd5de24`.
The final builder marker is `CONTROL_GAP_V4_CURATED_OK`; split audit status
is `SPLIT_INTEGRITY_OK`.

The acquisition is *targeted* at the signed-action confusion, but it is not
yet evidence of better prediction or control, and not an active-learning
sample-efficiency result without an equal-budget random-data comparator.
Train and evaluate v4 separately from the immutable v3 model and data.
