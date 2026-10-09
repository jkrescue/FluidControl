# FC-P022: current-force input versus zero-input controlled comparison

Root authorizes CPU implementation/testing preparation only. GPU execution must
be separately approved against reviewed immutable source hashes. Existing CFD
and FNO baselines remain intact; this is not a replacement project or new model.

## Hypothesis and fixed comparison

Exact current force information can improve control-relevant H1 and free-running
H100 prediction under the same optimization budget, relative to zero force inputs.
Use official pinned PhysicsNeMo FNO and reviewed P021 input/recurrence adapter.
Parent is P018 terminal; flow model stays frozen. Force model input6->10 is
warm-started with zero new columns, preserved physical inputs and moved coordinate
columns. Keep original two frozen lifting biases. No additional architecture,
statistic loss, normalization changes, future force input or CFD target edits.

Arms A_zero and B_causal start independently from identical expanded tensors.
Six train windows in order160,816,923,975,1077,1233; average all six gradients
before clip_norm1 and each optimizer step. Fresh AdamW per arm: lr1.5625e-7,
betas(.9,.999), eps1e-8, weight_decay1e-4. Sixteen updates per arm;192 window
backwards total. No best-iterate selection or learning-rate sweep.

Original normalized J0 is half H1 plus half AR, each domain half equal-four-force
MSE plus half rear-Cl MSE. H1 current inputs are exact same-time measurements;
AR uses measured initial force, then only own predictions. Paired batch2 and
checkpoint blocks10 retain full100-step differentiation, with no detach/reset.
Both arms use the same high/TF32 precision and code path.

## Measurement and decision

Record endpoints0 and16 twice each: all six windows separately, four-force
MAE/MSE, six-window original objectives and five-nonzero-window tail38:100
rear-Cl mean bias squared, RMS-amplitude error squared/absolute, centered
waveform residual MSE and actual prediction/truth RMS. Keep zero window separate
in statistic summaries and record strict-causal persistence without changing targets.

Local support requires B to improve both original domain objectives versus its
initial state and A; improve both domains' mean-bias and amplitude squared errors
versus initial and A; and not worsen centered waveform MSE versus initial.
Report individual regressions even if aggregates improve. Repeat spread is only
observed numerical variation, not a rigorous confidence interval. Unresolved
comparisons are inconclusive. Failure rejects this fixed finite protocol, not
all possible force conditioning. No automatic extra epochs after failure.

Save diagnostic results only, no deployable candidate. No validation/frozen data,
formal admission or PPO. A changed surrogate would still require compatible new
policy training and paired real-CFD validation; local fitting cannot prove that.

## Resources and stopping rules

P021 r2 resource result975fc40bbb88d5d0ab3d239ee0ce994bc0635aabcad9b7d74568bf73f1a8ce7a
measured4.60--4.86s per window backward, maxreserved3.779GiB, hostfree>=28.596GiB.
192-backward projection907.9s excludes endpoints, setup and optimizer. Actual
learned nonzero force-feedback stability and optimizer resources remain unproven.

Cache six frozen histories on CPU; stage only one window at a time on GPU.
Sequential arms, no concurrent GPU job. Preserve allocator0.06, container12GiB,
post-hash/pre-model startup MemFree>=30GiB and MemAvailable>=50GiB, both running
floors>=20GiB. Record optimizer moments/finite gradients, before/after update
memory and per-update progress. Internal whole-run deadline1800s; bounded outer
shutdown grace only. On nonfinite, memory violation, deadline or identity mismatch
stop and retain evidence; do not relax resource limits or automatically retry.
Scoped verified-file cache advice may follow reviewed r2 launcher; no global purge.

## Data coverage and subsequent work

The exact-time sidecar covers only606 endpoints in six fixed windows. It does not
prove full44-trajectory/1368-window training readiness. If locally supported,
independently extend train-only current-force provenance before full training,
then review causal formal-evaluation adapters while preserving original gates.
Only after accepted prediction and compatible controller results may the project
claim surrogate-assisted closed-loop success. Mean physical lift review remains
separate and is scheduled17:32UTC; this experiment changes no physical criterion.
