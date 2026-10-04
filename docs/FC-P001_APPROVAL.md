# FC-P001 — paired-stat candidate post-evaluation approval

Approved by Lead, 2026-10-04 15:18 UTC. This continues existing experiments FC-E006/FC-E007, not a new training run or changed research objective.

## Hypothesis and comparison

The existing lambda10 matched-pair statistic objective improves control-relevant force response relative to the existing lambda0 control, without unacceptable field/force degradation. A smaller training paired loss alone does not confirm this hypothesis.

Both candidates were trained from Main-e2, with the same regular/paired sampling and batch1 settings. Evaluate their existing selected epoch2 checkpoints without further optimization. Record exact model/state, resolved-config, launch and completion receipts, image and evaluation-code hashes before inference. Training source is b6aada9; do not assume current Git HEAD was used for training.

## Owned work

- Surrogate agent: new paired-candidate lineage adapter and immutable evaluation runner; Main lambda0 execution.
- Physics/Data agent: verified Worker lambda10 deployment, persistent execution and SHA-checked transfer to Spark.
- Control/Evaluation agent: independent protocol/result review, live task authority, experiment ledger updates.
- Lead: review evidence, determine acceptance/rejection/uncertainty and next experiment. This approval does not authorize surrogate PPO before its unchanged admission checks pass.

## Fixed protocol

Use official pinned PhysicsNeMo 2.2.2 FNO loading/inference and the existing evaluator. No new architecture, no retraining, no CFD data regeneration, no frozen HDF access.

1. Validation10: dev30 validation split, existing train-only normalization; horizons1/10/50/100, stride25, batch4, observed actions. Expected segment counts320/320/310/290. Preserve separate development diagnostic and formal endpoint audit.
2. Dynamic6: existing six-case validation panel; same normalization; horizons1/10/50/100, stride1, batch8. Expected H100 count606, all-horizon count3858. Preserve strict start0 action-difference analysis separately from all-window pooled metrics.
3. Force-window: existing start0 H100 diagnostic, same six cases and requested6.15D/U window (62 sampled endpoints span6.1D/U). Compute per-case mean totalCd, rear meanCl and centered rearCl RMS errors, relative-action errors and existing development gate. Do not replace per-case requirements with an average.
4. Store fields/forces/action response separately. Existing observational spatial/pressure diagnostics are not admission criteria.

Original numeric physical/admission thresholds remain unchanged. PASS of one item does not override FAIL of another. Report lambda0/lambda10 with exactly matching protocols; no best-case selection and no fabricated missing values.

## Preconditions, budget and recovery

Before GPU execution, validate both candidates' batch1/paired manifest/lambda/source/parent contracts; the old train16 batch2 lineage is inapplicable. Validate all CLI options against actual code, container paths against resolved configurations, paired checkpoint generation payloads, full data counts and immutable scripts. Run scoped tests and record their output.

One evaluation process per node, 8 CPU limit, existing container isolation, allocator fraction0.15, guard enforcing at least20 GiB physical unified MemAvailable. Initial budget: one complete evaluation suite per candidate; investigate if an individual suite exceeds60 minutes without evidence of progress. Never restart solely due to an observation timeout. New inference repeats require a documented invalid result, not a postprocessing-only error.

Reuse complete verified inference artifacts for CPU audit recovery. Preserve failures and signatures. Launch immutable shell copies; do not modify a running script. Worker results must return to the canonical Spark candidate directories with transfer receipts.

## Decision

Publish all pass/fail results and compare magnitudes per common protocol. If paired loss does not improve the intended action/force statistics, reject or qualify the hypothesis and use the failure map to select the next controlled intervention. If candidates satisfy admission, proceed with the existing FNO-specific PPO/real-CFD validation plan; no physical-benefit claim until paired OpenFOAM verification. Update PROJECT_STATE, EXPERIMENTS, results.csv and DECISIONS from actual evidence.
