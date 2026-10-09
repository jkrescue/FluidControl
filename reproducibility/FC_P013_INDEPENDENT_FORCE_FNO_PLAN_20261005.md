# FC-P013 — independent official FNO for aerodynamic prediction

Lead approval: implementation and CPU tests only. A single no-update GPU resource
probe and subsequent training each require separate source-bound approval.

## Evidence and hypothesis

P011 failed both complete development evaluations. P012's twelve train windows
did not meet either predeclared strong gradient-scale or conflict condition.
Therefore this is not a loss-weight scan or a claim that gradient conflict caused
the failure. P008–P011 mostly altered output/decoder representations; this next
test asks whether a full, independent official FNO can learn aerodynamic forces
more accurately while preserving the existing flow predictor exactly.

The existing single-FNO baseline and all results remain intact. No new geometry,
Reynolds number, dataset, controller, or scientific acceptance threshold is added.

## System and fixed inputs

- Flow FNO: frozen P009 parent, model SHA
  `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`,
  state SHA `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`.
- Aerodynamic FNO: independent instance initialized from the exact same pair,
  constructed and loaded through the existing official PhysicsNeMo FNO APIs.
  Train its feature representation for four-force prediction; ignore its field
  outputs. Do not invent a new neural architecture or claim the adapter is an
  official PhysicsNeMo component.
- Both networks receive the same current state, mask, current action and next
  action. Advance the state with the frozen flow model's existing **residual**
  update and mask. Forces come only from the aerodynamic model's existing masked
  spatial average. Never feed the aerodynamic model's discarded field into rollout.
- Preserve the six-channel input, seven-channel official model, normalization,
  physical scaling, action indexing and all existing baseline mathematics.

## Proposed bounded training contract

Exact prior 1368 train H100 windows and order (720 base20 / 408 train8 / 240
train16); batch one, seed 20261003, no validation/frozen access or checkpoint
selection. Parent resolved configuration SHA
`07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`
is inherited for architecture/data. Record new training semantics explicitly;
do not mislabel this as the unchanged P011 objective.

For each window use equally weighted true-state H1 and frozen-flow autoregressive
inputs. For each domain use the same P011 balanced four-force objective:
0.5 times equal-channel normalized MSE plus 0.5 times rear-Cl normalized MSE.
Total objective is 0.5 H1 + 0.5 AR of that force loss; there is **no field loss**.
AdamW learning rate 1e-5, weight decay 1e-4, clipping norm 1, one optimizer update
per window, terminal checkpoint only. Record actual gradients and unused output
rows honestly. Chunk force forward/backward with mathematically exact full-window
weighting to bound memory; CPU tests must verify aggregation and target alignment.

This comparison tests the composite independent-force representation system.
It cannot separately attribute any gain to capacity, decoupling, or H1/AR mixing.
No causal claim about gradient conflict is permitted.

## Verification before training

1. CPU tests: separate model instances; frozen flow unchanged; residual updates;
   same-input pairing; correct H1 target alignment; masked force pooling; loss
   chunk weighting; wrong/missing/swapped checkpoint rejection; old single-model
   path unchanged.
2. One separately approved H100 real-train resource probe: both forward branches
   and aerodynamic backward, no optimizer/update/save, before/after tensor hashes,
   measured memory/time, all required gradients finite. Five-minute cap.
3. Actual timings determine training budget, provisionally at most four hours.
   Continuous unified MemAvailable >=20 GiB; isolated official image; allocator
   <=0.45. No concurrent job that violates the shared physical memory floor.

## Evaluation and persistence

Save/load both official checkpoint pairs and bind their identities in a new
dual-model manifest. The flow pair must remain byte-identical to P009. Fresh
reload is mandatory. Original single-model evaluation remains available.

A thin, explicitly project-owned adapter combines frozen-flow field outputs and
aerodynamic force outputs. Apply it only at prediction/loading points in the
existing evaluators; preserve validation10, dynamic6, force-window and development
metric formulas, data, horizons, output schemas and numerical thresholds. Bind
both checkpoint pairs in evaluation receipts. Verify the actual combined model,
not a hypothetical replacement of force values in old JSON results.

Keep the existing fixed six train-window H1/AR diagnostics. Report all windows;
do not select a checkpoint or training duration using development results.
If train fitting remains poor, investigate representation/optimization evidence.
If train improves but formal fails, report the generalization deficit. Neither
case authorizes PPO. Only unchanged complete admission can lead to a newly trained
compatible policy and real-CFD feedback evaluation under the original criteria.

## Ownership

Compute: new trainer, resource-probe implementation and their tests.
Surrogate agent: separate dual-checkpoint loader/adapter and minimal evaluator
integration plus regression tests; no edits to Compute's trainer.
Control/Evaluation agent: independent review, ledger and scientific interpretation.
Lead: approval, integration, dashboard, next experiment decision.
