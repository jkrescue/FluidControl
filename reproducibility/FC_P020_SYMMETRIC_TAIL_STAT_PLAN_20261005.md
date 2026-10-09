# FC-P020 symmetric tail-statistic finite-update comparison

Staged preparation and CPU tests only. GPU execution requires separate Lead
approval. This is a train-only mechanism check, not a saved candidate or admission.

## Question and sole intervention

P019's five nonzero windows each have negative own-window AR RMS directional
derivative, but their aggregate has positive derivative. Cross-window terms
therefore reverse the aggregate sign; missing per-window descent is not the
explanation. Raw-gradient directions are not AdamW updates. Test whether adding
symmetric H1/AR temporal statistics improves actual finite updates without the
H1 tradeoff observed in P010's restricted affine, AR-statistic-only fitting.

Both arms start at exact P018 terminal with a fresh AdamW and the same seed.
Arm A uses immutable original J0. Arm B uses

    J1 = J0 + (5/16) * (B_H1 + R_H1 + B_AR + R_AR)

On normalized rear-Cl tail62 (array indices38:100),
B = mean(prediction - target)^2 and
R = (std(prediction, ddof=0) - std(target, ddof=0))^2.
The fixed5/16 coefficient inherits the original per-domain rear-Cl weight.
These normalized statistics have the same units as normalized pointwise MSE,
and B+R <= tail pointwise MSE. This supplies a prespecified scale, not proof of
an optimal coefficient or equal gradient contributions. No coefficient search,
loss-adaptive weights, architecture changes or removal of pointwise supervision.
No epsilon is added to RMS; zero predicted RMS makes the derivative undefined
and stops the probe without claiming a completed comparison.

## Fixed data, model, optimization and numerical scope

Exact P018 candidate/model/state/audit hashes and original P013/P014 source
manifest are inherited through SHA-pinned P019 helper
03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64.
Only existing train indices160,816,923,975,1077,1233, with160 the separate zero
window. Launcher must independently verify all44 train HDF bytes against pinned
P018 audit03153fa5…2323b9 and train-only mounts before launch. No heldout read.

Each arm: exactly16 updates, all six raw window gradients averaged before one
global clip1, fresh AdamW lr1.5625e-7, betas(.9,.999), eps1e-8, wd1e-4. Same seed
20261003, flow frozen/eval, force train mode,28 trainable force parameter tensors
and exact2 frozen lifting biases. No scheduler, extra epochs, interim selection,
or optimizer-state continuation. Arm A then B, verified P018 tensors restored
between arms; identical RNG initialization in each arm. Only diagnostics at0/16,
each evaluated twice without model/gradient mutation. Preserve both sets of
per-window/aggregate metrics and their signed differences. Check A/B initial
agreement against their combined observed repeat spreads, in addition to exact
P018 tensor restoration.

Preserve official mixed20 chunk10 and default TF32/high protocol. Frozen-flow
states may be cached on CPU. B first captures forces without gradients, computes
float64 normalized output cotangents, then replays the unchanged pointwise
objective. An output hook adds weighted statistic cotangents once to its original
backward; no additional chunk factor and no H100 retained graph. CPU toy tests
must compare this combined gradient against fullgraph autograd using the actual
pinned objective, including H1/AR placement, tail indices and averaging/clip.
Record output-replay/cotangent-cast discrepancies as observations, not pass gates.

## Fixed terminal interpretation

Preserve per-window/zero and five-nonzero summaries, separately for H1/AR:
original objective, mean error squared, RMS amplitude error squared and absolute
RMS error, centered residual MSE, and four-force pointwise errors. Physical
statistics multiply normalized errors by the fixed training force std; actual
training uses normalized quantities. Centered residual MSE is not amplitude error.

Support the proposed modification only if B terminal's four five-nonzero squared
statistics are strictly lower than both its initial values and A terminal, both
six-window original H1/AR objectives are lower than initial, and both five-window
centered residual MSE values do not increase from initial. These are local
interpretation rules, not new scientific admission thresholds. Report exact
deltas; each required comparison reports the sum of its two endpoint absolute
repeat differences. An improvement no larger than that observed spread, or an
A/B initial discrepancy beyond it, is explicitly numerically inconclusive, not
support for the intervention. Exactly unchanged centered error with zero repeat
spread retains the original non-increase rule. This observational screen is not
a rigorous error bound or a new scientific tolerance. If only AR improves at H1's expense, reject this
modification. If both arms improve similarly, do not attribute benefit to new
statistics. No post-hoc coefficient sweep. Only support plus separate approval
can motivate full-train fixed-terminal testing under original formal gates.

## Resource and output limits

Sequential arms, maximum192 window backwards (2×16×6), plus B no-grad captures
and fixed0/16 diagnostics. One current GPU gradient, no P019 CPU double gradient
collections. Allocator0.15; MemAvailable and MemFree each>=20GiB;30-minute whole
probe budget, external hard guard/owned-container cleanup required. No parallel
GPU workload. Emit progress per window/update; abort nonfinite gradients,
parameters or optimizer state, frozen-bias changes or flow mutation.

Save only JSON evidence/logs. No checkpoints/candidate. Restore exact P018
parameters/buffers and clear gradients in finally, preserve initial RNG. Report
the changed in-memory terminal tensor hashes separately from final restored
hashes. Completion is NOT scientific admission, policy training or CFD success.
