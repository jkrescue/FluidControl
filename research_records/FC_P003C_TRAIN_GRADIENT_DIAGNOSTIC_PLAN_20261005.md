# FC-P003C train-only gradient diagnostic plan

Status: implementation and CPU tests only. No GPU diagnostic, optimizer,
candidate checkpoint, validation/frozen read, or hyperparameter decision is
authorized by this document.

## Purpose and fixed inputs

The diagnostic explains the scale and direction of the force-learning signals;
it does not replace FC-P003C's complete post-evaluation and cannot admit PPO.
It uses the immutable Main-e2 parent (`8466bd47...4240`, state
`1e5d4c05...3cbd0`), resolved FC-P003C configuration, train-only dev30/train8/
train16 roots, dynamic8 pair manifest `b756c6d7...2d28c`, normalization
`f1b4607e...92bc1`, and real official-sampler receipt `da078a1c...3242`.
No validation or frozen path is mounted.

The 16 positions are exactly
`[0,91,182,273,364,455,546,637,729,820,911,1002,1093,1184,1275,1367]`.
The script recreates the official regular sampler, verifies its full index SHA,
recreates the two dynamic8 permutations, and then reads only those 16 regular
H100 windows and 16 paired items through fixed samplers. This avoids scanning
irrelevant HDF windows while preserving the approved identities and order.
This is a fixed-parent, epoch-1 insertion-sequence diagnostic. It does not
recreate the 32 changing optimizer states encountered over two training
epochs, and it must not be described as doing so.

## Reused project interfaces

- `regular_rollout_objective`: authoritative field and regular-force losses;
- `paired_statistic_loss` plus the existing paired rollout: old lambda10 term;
- `predict_true_state_force_chunk`: new causal true-state term, ten chunks;
- official PhysicsNeMo FNO, DataLoader and checkpoint loader.

The diagnostic uses `torch.autograd.grad`; it never creates an optimizer,
updates `.grad`, clips/steps parameters, or saves a model. Gradients are moved
to CPU and reduced sequentially so graphs are released between components.

For every actual insertion position it records field, `0.2*regular_force`, full
regular, `10*old_pair`, and `10*true_state_pair` norms; pair/regular ratios;
cosines and sign-conflict fractions; mixed-gradient norm and the implied fixed
clip-scale. Both the regular force term and the new term report four channel
losses and gradient norms; the regular decomposition is required to reproduce
the authoritative combined force loss and gradient. The old term reports its
three physically defined statistic groups (mean total Cd,
mean rear Cl, rear-Cl RMS), because those nine horizon/statistic cells do not
constitute four independent force-channel losses. Mean-gradient diagnostics are
also reported but explicitly do not replace the 16 per-position observations.
Parameter-and-buffer hashes before and after the run must be identical.

## Bounded feasibility and proposed execution budget

The completed one-position mixed probe took 7.25 s and peaked at 14.225 GiB
allocated / 17.533 GiB reserved. This diagnostic adds the old paired rollout
and multiple derivative extractions but does not update or save a model. A
conservative Worker budget is one GPU, 8 CPU, allocator fraction at most 0.35,
90 GiB container memory, at least 20 GiB unified `MemAvailable`, 20 minutes,
and one exclusive output. Expected useful runtime is roughly 5--15 minutes.
Any nonfinite gradient, sampler mismatch, parent hash change, memory-floor
crossing, or unexpected output stops the run and preserves evidence; no
automatic retry is proposed.

The output is diagnostic only. It must not be used to tune lambda or channel
weights from validation performance, claim a cause from mean gradients alone,
or describe a force-only improvement as full-field surrogate accuracy.
