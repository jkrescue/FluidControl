# Dynamic-FNO formal Gate/PPO handoff audit (2026-10-04)

Status: **closed by default**.  This is a read-only wiring audit; it does not
promote a checkpoint, execute a Gate, open frozen data, or authorize PPO.

## What exists

The current H100 candidate family uses the official PhysicsNeMo FNO with
`train20 + train8`, the unchanged train20 normalization, zero teacher forcing,
and H100 train/validation rollouts.  The Spark copy at
`artifacts/tandem_fno_dynamic_train8_h100_worker_h100_e5_2ep_r2_20261004`
contains two checkpoint generations.  Epoch 2 currently hashes to
`ef95ff96582983680800710679258a6705eb2294fab6a43dfa38163a606ed0c8`;
this identifies a file, not a promoted model or Gate result.

There is **no safe formal command yet** for this lineage.  Running
`scripts/run_full40_validation_only_spark.sh --execute` directly is invalid:
the wrapper admits only `artifacts/tandem_fno_full40_*`, hard-codes
`conf/tandem_fno_full40_h20.yaml`, and calls the old train20-only plan.  Merely
renaming/copying the dynamic checkpoint into an accepted directory would lose
its train8 ancestry and must be rejected.

## Required candidate publication and formal validation

Before execution, publish one immutable candidate view containing exactly one
`FNO.*.mdlus`, selected by the already declared rule, plus a new receipt that
binds all of the following:

- checkpoint and optimizer-state SHA, epoch, complete training history and
  selection rule;
- exact resolved H100 config and its Hydra parents;
- source commit/tree and training/evaluation implementation SHA;
- parent H50 checkpoint/state SHA chain;
- dev30 manifest `5213c7bb...ddd2`, train8 manifest `a0bd0e3b...3f35`, and
  shared normalization `f1b4607e...2bc1`;
- train20/train8 window counts and strides, official image ID, sync receipts,
  and `frozen_test_transferred_or_opened=false`.

Add a **new candidate-specific** validation wrapper or provenance adapter; do
not edit an old receipt.  Its dry-run and, only after review, execute forms
should be equivalent to:

```bash
DYNAMIC_FNO_CHECKPOINT=<immutable-best-dir> \
DYNAMIC_FNO_LINEAGE=<new-lineage-receipt.json> \
VALIDATION_OUTPUT=<new-empty-artifact-dir> \
scripts/run_dynamic_h100_full40_validation_only_spark.sh --dry-run

FULL40_VALIDATION_APPROVAL_TOKEN=<reviewed-token> \
DYNAMIC_FNO_CHECKPOINT=<immutable-best-dir> \
DYNAMIC_FNO_LINEAGE=<new-lineage-receipt.json> \
VALIDATION_OUTPUT=<new-empty-artifact-dir> \
scripts/run_dynamic_h100_full40_validation_only_spark.sh --execute
```

The wrapper must use `conf/tandem_fno_dynamic_train8_h100.yaml`, the pinned
PhysicsNeMo image, full40 **validation10 only**, H1/H10/H50/H100 with the formal
stride/action protocol, and the unchanged full40 normalization.  It must bind
the evaluator/config/checkpoint/data SHAs and write a new report, segments and
Gate receipt.  `audit_full40_validation_gate.py` can retain its unchanged
metric thresholds, but its result needs the supplemental dynamic-training
lineage receipt; the old dev30-to-full40 promotion receipt proves evaluation
data identity only and does not prove train8 training ancestry.

## PPO remains blocked after an endpoint pass

`run_full40_canonical_ppo_spark.sh` currently hard-codes the old H20 config and
checkpoint default.  Its preflight also recomputes three independent pieces of
evidence for the exact checkpoint:

1. `gate.json`: H100 endpoint force/action-difference Gate;
2. `canonical_window_gate.json`: 6.15-D/U total-drag, rear-Cl fluctuation and
   mean-bias fidelity on b01/b05;
3. `dynamic_action_gate.json`: dynamic-action fidelity for `|omega|<=0.75`,
   `|delta omega|<=0.1`, horizon at least 100, on b01/b05.

Only the first producer exists today.  The current dynamic6 force-window run is
explicitly diagnostic and cannot be renamed into either missing Gate.  New
predeclared producers must create those checkpoint-bound receipts without
changing thresholds after seeing results.  PPO also needs a candidate-aware
preflight that checks the new lineage receipt, uses the H100 dynamic config to
rebuild the network, and records its SHA in readiness/audit output.  Full40
zero cases and the full40 normalization remain correct runtime inputs; train8
need not be mounted for inference, but its manifest and normalization identity
must remain cryptographically bound.

The first permissible command is therefore the new wrapper's `--dry-run`.
PPO `--execute` remains forbidden unless all three fresh receipts pass for the
same checkpoint SHA.  A failed H100 validation leaves the current closed state
unchanged; passing surrogate Gates would authorize only surrogate PPO, still
followed by the predeclared frozen-policy real-OpenFOAM comparison.
