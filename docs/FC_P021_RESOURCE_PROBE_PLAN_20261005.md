# FC-P021 one-window GPU resource probe — preparation only

Root approves preparing the harness and CPU tests described here, NOT executing
GPU work. A separate execution approval must bind final reviewed hashes.

Question: can the reviewed causal-force module execute its actual full-size
official FNO H100 recurrent forward/backward with adequate memory and bounded
latency? This is an engineering measurement, not an optimizer or accuracy test.

## Fixed workload and identities

- Exact P018 terminal parent, frozen flow, original configuration and training
  normalization; only official force FNO physical input count6->10 with reviewed
  zero-column/relocated-coordinate warm start.
- Exactly existing train global index816: dynamic_train8_b00_prbs, start90,
  H100. First nonzero window in the previously fixed panel, not selected by outcome.
- Exact same-time current forces from reviewed six-window timestamp audit
  SHA72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2.
  The loader must verify the274 raw/config sources using an approved-files-only
  mirrored read-only view; no whole-CFD-case or heldout mount.
- Pin module bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962,
  original objective and all inherited P018/config/model/data identities.
- Same official image, default reviewed high/TF32 precision, paired sequential
  batch2 and full H100 chain rule, checkpoint blocks10. No detach or truncated
  recurrence; no new statistic loss.

Generate the frozen-flow history once. One no-grad legacy six-input force panel
may quantify expanded initialization differences under the SAME paired batch2
schedule at all100 times. Keep original weights unchanged; no perturbations to
real-model added columns. The synthetic nonzero-column gradient fixture is NOT
part of this real-data probe. Frozen flow/legacy force may be moved to CPU after
their histories/reference panel are cached; record this memory strategy.

Run exactly one forward/backward for zero-input A and one for causal-input B,
sequentially at identical expanded initial tensors. Clear gradients between arms;
never construct an optimizer, update weights, save a candidate or select a model.
Validate finite all28 gradients, frozen biases/no flow gradients and unchanged
model tensors after cleanup. Record new-force-column gradient norms separately;
A's data gradient there should be zero. This diagnoses implementation, not the
sufficiency of a future learning rate or training budget.

## Resources and outputs

GPU0 on Main Spark only, no concurrent GPU job. PyTorch allocator fraction0.06
(below the design's0.15 maximum), container memory12GiB, initial post-hash physical
MemFree>=30GiB and MemAvailable>=50GiB. Both20GiB floors remain mandatory with
external2-second guard and internal checks. Whole-probe inner deadline900seconds;
outer owned-container timeout/cleanup may allow only shutdown grace. No global
cache eviction and no automatic cap increase or retry on OOM/deadline.

Record source/data/parent/tensor hashes, exact executed identity, forward/recompute
call counts, each-arm elapsed time, CUDA allocated/reserved peaks, external host
minima, input/precision profile, normalized H1/AR objective and physical outputs.
Preserve initialization output differences as measured values; a changed kernel
shape need not be bitwise identical. Do not invent a scientific tolerance to
approve a discrepancy. Root/independent review decide engineering validity.

Report Adam moment storage projection as twice the trainable parameter bytes and
an additional parameter-sized allowance for update temporaries. These are
projections, not measured optimizer/full-training feasibility. No dummy optimizer
steps or fake GPU loads. Extrapolated 192-window runtime for a later16-update pair
is likewise an estimate; it cannot authorize a training run or deadline change.

Write only exclusive diagnostic JSON/logs and resource observations. Restore/
verify source identities and clear gradients. On failure preserve all evidence;
diagnose it before proposing a revised plan. On engineering success, next action
is independent review and a separately approved finite scientific comparison,
not PPO or acceptance. Full closed-loop goal and original metrics are unchanged.
