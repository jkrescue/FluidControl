# P025 isolated-input statistical-loss comparison — CPU preparation only

Question: can the existing fixed symmetric mean/RMS training terms change the
learned96-input direction to improve both H1 and AR errors, instead of merely
amplifying P023's direction? P024 revealed direction-dependent tradeoffs, not an
unavoidable limitation. P020's similar terms updated old weights, so it does not
answer this frozen-parent question. No new architecture or invented library API.

Root approves CPU implementation/tests; GPU execution requires final independent
review and immutable approval. This is one bounded test, not a coefficient sweep.

## Fixed matched protocol

P018 canonical expanded zero-block initialization, same six exact-causal windows,
normalization, official image/source and frozen field histories as P023. All old
FNO parameters fixed; only96new input coefficients, freshAdamWlr1e-5, betas.9/.999,
eps1e-8,weight_decay1e-4; average all6 gradients thenclip1;16updates.
Original objectiveJ0 unchanged in evaluation. Only training objective changes to

Jstat = J0 + (5/16)*(H1_mean_error_squared + H1_RMS_error_squared
                  + AR_mean_error_squared + AR_RMS_error_squared).

The four statistics use normalized rearCl predictions and unchanged targets,
tailindices38:100, centered RMS with correction0, exactly P020's mathematical
definition. All six windows train; five nonzero windows form the same summary.
Compute statistic losses through differentiable float64 casts of predictions,
matching P019's statistic values/cotangents, then autograd through FULL causal
H100 recurrence and checkpoint10. No detach, no independent-chunk VJP shortcut;
P020's old chunkwise gradient implementation is not applicable to this recurrence.
Zero predicted centered RMS is an undefined-gradient failure, not silently eps-
smoothed. Preserve original physical and surrogate accuracy thresholds.

## Matched control and required checks

Use pinned P023HIGH original-loss control (resultadfdd9a8) without rerunning its
completed16updates. Before training, repeat the initial panel twice and require
exact identities/rawnormalized predictions/metrics/aggregate reproduction of its
initial state. Otherwise stop; do not redefine comparability or adjust tolerance.
No intermediate model selection. Terminal panel twice; terminal zero-input
ablation twice, with model/gradient/optimizer/allRNG nonmutation checks.
Total96windowbackwards,16updates,36evaluation windows plus600flowcachecalls;
9600training+9600recomputation+3600panelpaired modelcalls.

CPU tests must match each normalized statistic and prediction cotangent against
the reviewed P019 expression, plus checkpoint/full-H100 end-to-end96gradients
with nonzero causal block. Verify statistical terms applied once, bothdomains
receive gradients, all old weights frozen and optimizer owns only96 coefficients.
Retain exact-zero-input, freshoptimizer, finite/memory and restoration protections.

## Decision and resource scope

Compare terminal with shared initial and stored P023HIGH using the same P022/P023
simultaneous H1/AR objective, mean, RMS and centered-waveform rules plus observed
repeat checks. Report every window, especially816; preserve original-loss and
augmented-loss values separately. No candidate/heldout/PPO/save or bestepoch.
If unsupported, end this isolated-block/statistical-loss branch and reassess the
representation/optimization strategy; do not sweep its coefficient or repeat it.
If supported, prepare complete train-only exact-time coverage and a separately
approved candidate experiment; no automatic scientific admission or policy run.

One official isolated GPU container, allocator.06/container12GiB. Startupfree30/
available50GiB, continuous both20GiB. Inner1200seconds/external1300/outer1340;
stop finite/identity/reproduction/resource violations and preserve failure.
