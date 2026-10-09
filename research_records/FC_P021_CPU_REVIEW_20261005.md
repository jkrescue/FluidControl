# FC-P021 CPU engineering review

Root, implementation and independent reviewer each ran 24 CPU tests successfully.
Reviewed module SHA bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962;
test SHA a3dfb4b88d70eabb72d2df560444f828a6041b812e7cba9d06d1785139f9b5eb.
This permits integrating the separate project-owned engineering module, not
replacing a production model or launching training.

The module preserves full H100 force recurrence through ten-step checkpoint
blocks, uses only initial measured force then predicted AR force, preserves
the original objective, and explicitly maps official FNO coordinate columns.
Tests include nonzero complex gradients, an analytic cross-block derivative,
future-truth mutation invariance, initial-force influence and rejection of known
batch-mixing/stochastic/stateful layers. Only the pinned official model's empty
float32 device markers are accepted as buffers.

Root independently ran the tiny official FNO fixture in the fixed CPU-only
container, with 5 spectral layers and 2 decoder layers. Warm-start output
difference was zero. With nonzero added input weights only in that synthetic
fixture, fullgraph/checkpoint H100 output difference and all28 trainable gradient
differences were zero; ten spectral imaginary-component gradient slots were
nonzero. Two frozen biases stayed unchanged. See
FC_P021_ROOT_CPU_EXECUTION_20261005.json for the observed command and result.
This is not CFD training, GPU evidence, full-size TF32 equivalence or an accepted
surrogate. No optimizer or model checkpoint was created.

The strict-current-time persistence analysis was independently recomputed exactly.
Its final receipt SHA is a91fb62d30eba9634e9f7c32ff5117e8e20f8e4802b90fcbd0c7bed356ebaf5f.
Five-nonzero H1 rearCl MAE0.1071951411, centered residual MSE0.01325332265 and
RMS-amplitude absolute error0.00402197537 leave the prior interpretation unchanged:
copying current force is worse than FNO for waveform prediction. AR constant-initial
persistence remains unchanged because all initial values were exact.

Preserved evidence: artifacts/fcp021_cpu_engineering_20261005/ contains the reviewed
module/tests, tiny fixture and receipts. Earlier smoke receipts retain their older
module identity and are not used to certify the final revision.

Next: prepare and independently review a one-real-training-window, no-optimizer
GPU resource probe. GPU execution requires separate Root approval; physical
MemAvailable and MemFree must each remain at least20GiB. No scientific admission,
full-training promotion or PPO is authorized by this CPU review.
