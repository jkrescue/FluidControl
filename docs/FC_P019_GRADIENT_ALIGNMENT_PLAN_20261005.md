# FC-P019 fixed-panel gradient alignment — staged proposal, GPU not authorized

## Hypothesis and interpretation

At the fixed P018 terminal, descending the original mixed H1/AR pointwise force
objective may locally increase tail62 mean-error squared or RMS-error squared.
This is a directional diagnostic, not an optimizer experiment or admission gate.
Positive directional derivative along the negative original gradient supports
local conflict for that statistic; negative means local alignment. Repeat-sensitive
or zero-norm cases remain unresolved. No numerical cutoff, majority-count gate,
loss weight, architecture change, model selection or training is introduced.

P010 fitted a restricted affine readout with a chosen additional RMS loss weight
and did not reach its convergence criterion. P016 changed model parameters for
32 steps. This probe instead measures gradients of the existing nonlinear
28-parameter-tensor force FNO, with no optimizer or parameter displacement.
Raw-gradient directions must not be described as actual AdamW directions.

## Fixed identities and data

Use only P018 terminal aerodynamic model SHA
`8a89f4774923afa698328e8e65ae337e7e0efefdae0bc9452d6b7a78758fb70d`, state
`d78d43d43738dd63b9556819e22f6b57c16993affaaff94dc3a29b6250994d6c`, manifest
`91cc2c9a295a1ace5eadcffd8242234b93b73d14959bb902e17b59655f4acf13`.
Frozen P009 flow model/state remain `dc41fc91…f42e31` / `4998e534…fb771e`.
Pinned P018 audit is `03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`.
Full model, state, tensor, base-configuration and protocol hashes are checked by
the script; the immutable source manifest is inherited from pinned P014 helper
`849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d`.

Exactly P014/P016 six train windows, global indices 160,816,923,975,1077,1233.
Index160 is the zero window and is reported separately. No validation/frozen
mount or read. Launcher approval must recompute all 44 train HDF hashes against
the pinned audit, bind the resulting map and exact train-only mounts; the script
checks train manifests/normalization and this audit but does not repeat bulk HDF
hashing. This prerequisite is not yet execution evidence.

## Numerical contract

Preserve base config SHA `07e55fd1…085d9`, original default TF32/high precision,
flow eval/frozen, force train mode (no stochastic/running-statistic layers), and
the exact ten chunk forwards of mixed H1/AR batch20. Targets1–100 are supervised
uniformly by the immutable P013 objective; tail62 is targets39–100, array38:100.

For normalized rear-Cl predictions p, truths y, physical training std s:

- mean statistic B = [s mean(p-y)]² on tail62;
- RMS statistic R = [s std(p,ddof=0) - s std(y,ddof=0)]² on tail62.

These are physical coefficient-squared errors matching the original force-window
quantities; centered residual RMSE is not substituted for RMS amplitude error.
No epsilon is added to RMS. Exactly zero predicted RMS has undefined derivative:
record unavailable, do not invent a value or claim alignment.

Compute original gradient and B/R gradients separately in H1 and AR. Capture
normalized outputs, calculate analytic output cotangents in float64, replay the
same mixed20 chunks and contract cotangents with outputs for streaming backward.
No full H100 graph, no gradient through frozen flow, no optimizer, clipping or
update. Replay-output differences and cotangent casting are explicitly recorded.
Real/complex CPU double accumulators use Re(sum(conj(g_stat)*g_original)).
Report dot, norms, cosine and derivative along -g_original and its unit vector.

Each of all five gradients is computed twice, including independent original
captures for each repeat. Report paired discrepancies and both measured signs;
their spread is observational, not a rigorous floating-point error bound or a
scientific PASS tolerance. Per-window results and index160 separate; report
six-window original-objective mean, and five-nonzero macro gradient comparisons.
Also report each statistic's five-window gradient against the six-window original
gradient, explicitly identifying the differing aggregation scope.

## Budget, invariants and output

Six frozen-flow sequences cached only in CPU RAM if needed; stream one window
and chunk at a time. Maximum 60 window backward passes (6×5×2), 15-minute wall
limit, allocator fraction0.15, MemAvailable and MemFree both >=20GiB. Launcher
must impose an external hard deadline/resource guard and exact owned-container
cleanup; no overlap with another GPU job. Print a heartbeat after each gradient.

Model parameters/buffers and the two frozen lifting biases must retain exact
tensor SHA; force gradients are cleared and RNG restored. Expected flow-eval and
force-train modes are preserved (these are newly loaded local models, not an
externally supplied model whose default training mode is restored). Only result JSON,
logs and resource observations are written, never checkpoints or candidate.
Completion status is diagnostic-only and scientific_admission remains false.
CPU tests must verify analytic cotangents against full autograd, tail slicing,
streaming VJP/averaging, complex real inner products, zero RMS handling, repeat
reporting and immutable model scope. Root review and separate GPU approval are
required before any execution.
