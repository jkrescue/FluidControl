# FC-P003C: true-state per-step paired force supervision

Lead record: 2026-10-05 Asia/Shanghai.
Status: engineering implementation and CPU tests approved. No full training,
PPO, CFD execution, or new GPU probe is approved by this document alone.

## Evidence and hypothesis

FC-P003B completed formal evaluation and SHA-verified return to Spark.
Its checkpoint is `ed0da140da2b81b99da200847b57fbbd97a885dbdc030691bac3437ce8ac2e08`;
post-evaluation receipt SHA is
`98f3d336b79a7816f17c204bfd521c1bdbcc83eed1b5ddf5df59c04995314258`.
The development gate remains FAIL. Four rotating-branch rear-lift fluctuation
RMS errors are 0.160242, 0.206943, 0.172678, 0.153707: only about 2.94% lower
on average than FC-P003, versus unchanged branch limits near 0.0294.

The same-endpoint diagnostic contains 606 matched endpoints. With recorded
CFD states supplied at every prediction, rotating-branch H1 rear-Cl MAE is
0.191916, 0.156482, 0.157575, 0.191730; zero-action errors are 0.009827 and
0.006964. These are instantaneous-force errors, not fluctuation RMS errors.
Thus recursive accumulation alone cannot explain the failure. This does not
prove a unique cause or establish that the proposed loss will solve it.

Hypothesis: replacing paired window-statistic supervision with direct
per-endpoint action-minus-zero force supervision improves the learned
action-dependent force response and, subsequently, the existing window test.

## Controlled comparison

Use FC-P003B as the comparison, but initialize from the SAME Main-e2 parent,
not from the FC-P003B trained checkpoint. Preserve architecture, official FNO
and data components, train-only normalization, regular datasets and order,
seed, learning rate, two epochs, 16 interleaved paired updates, dynamic8 pairs
used twice per epoch, lambda=10, and force weights [1,1,4,1]/7.

Only the paired objective changes: old nine window statistics are replaced
by normalized force-difference squared error at each endpoint t+1, using each
branch's real CFD state at t and recorded actions at t and t+1. This helper is
project code, not a new official NVIDIA API. No validation/frozen samples may
enter training. No new model architecture or dataset generation is approved.

Keeping lambda=10 does NOT equalize gradient strength between different loss
definitions. The completed one-pair technical probe shows rear-Cd dominating
the proposed loss, while the principal failed metric concerns rear-Cl.
Record this risk and per-channel contributions; do not change channel weights
or lambda after viewing validation outcomes within this experiment.

## Implementation and tests

Keep the existing default statistical-objective path unchanged. The new
single-GPU path accumulates regular-loss gradients, releases that graph,
then accumulates ten true-state chunks, each scaled by chunk_length/100 and
lambda. Check finite gradients, clip only once after all contributions, and
perform exactly one optimizer step per regular batch. Reject unsupported
distributed execution explicitly. Do not modify previous immutable runners.

Test summed-vs-monolithic gradients and one-step updates, endpoint indexing,
chunk weighting, nonfinite failure without optimizer step, clipping placement,
unchanged default behavior, loader/RNG order and pair identities/positions.
Record objective kind, scalar/per-channel losses, and update counts.

After CPU review, propose one bounded real mixed-loss resource/equivalence
probe in an isolated container. Bind source/config/data/image hashes before
execution; retain at least 20 GiB physical unified MemAvailable continuously.
The earlier paired-only probe does not prove mixed-training memory safety.
Probe or training execution requires a separate Lead approval after review.

## Evaluation and decisions

Full training, if subsequently approved, uses exactly the existing complete
validation10, dynamic6, force-window and development protocols; frozen data
remain sealed. Retain same-endpoint H1/H100 and per-branch force diagnostics.
Only all existing admission requirements passing permits consideration of a
new compatible PPO policy. No threshold is relaxed by this experiment.

If rotating H1 errors and window errors remain essentially unchanged, reject
this supervision hypothesis for the tested budget. If H1 improves but windows
do not, investigate rollout/action-history transfer rather than claiming
control readiness. Improvement of a component is not project completion.
The final goal remains genuine CFD feedback with >=2% total drag reduction,
rear-Cl fluctuation ratio <=1.05 and normalized mean-lift bias <=0.10 under the
existing paired 80 D/U protocol (discard first 20, assess final 60).

Surrogate owns implementation; Compute owns isolated launch/probe preparation;
Evaluation independently reviews correctness and results; Lead approves each
execution milestone and scientific next action. Preserve all prior artifacts.
