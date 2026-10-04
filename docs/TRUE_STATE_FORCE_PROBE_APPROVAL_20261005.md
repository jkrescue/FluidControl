# True-state paired-force technical preflight

Lead record: 2026-10-05 Asia/Shanghai.
Status: PLAN APPROVED; EXECUTION PENDING FINAL CODE REVIEW AND LEAD GO.

## Purpose and limits

FC-P003 did not materially improve the rotating-branch lift response. The
next proposed supervision primitive predicts force at t+1 from the recorded
CFD state at t and recorded actions at t/t+1. Before any integration, verify
its causal indexing, gradient accumulation and resource cost on real training
data. This is not a new scientific training experiment: no optimizer object,
optimizer step, policy, candidate checkpoint, CFD run or validation/frozen
access is permitted. FC-P003B continues unchanged on Worker.

The result can establish only the feasibility of the paired loss term. It
cannot establish full mixed-trainer memory, surrogate improvement, PPO
admission or physical control benefit. A formal follow-up remains conditional
on FC-P003B's full evaluation and a separate approved experiment definition.

## Bound inputs

- Main Spark project only; Worker is not used by this probe.
- Fixed pair `b00:multisine`, start 0, batch 1, first 100 endpoint steps.
- Dynamic pair manifest:
  `artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json`,
  SHA `b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c`.
- Action HDF SHA
  `400b3b5f8ea381da59c7806aef19f7cd2b50da303f717ad93cebf505c5949e9f`.
- Same-start zero HDF SHA
  `243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01`.
- Train8 manifest SHA
  `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35`.
- Train-only normalization SHA
  `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`.
- Main-e2 model SHA
  `8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240`;
  state SHA `1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0`.
- PhysicsNeMo image ID
  `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Project helper commit `26abbc3`, SHA
  `d382c7c886509fb12b85e7f79080dcf1b85d9c2315c15717e97310459dbfee30`.

Use the existing official-reader composition, FNO and predict function. Record
all executed source/config hashes, model mode and package versions; the helper
is explicitly project code, not a new NVIDIA API.

## Numerical contract

Force order is front_cd/front_cl/rear_cd/rear_cl; normalized weights are
[1,1,4,1]/7. Both branches use the same training normalization. At zero-based
step k, use initial state for k=0, otherwise target_state[k-1]; the force target
is target_force[k], physically endpoint k+1. Do not use predicted states.

First compare T=20 unchunked with two chunks of 10 on identical fixed weights.
Each chunk mean loss is multiplied by chunk_length/T before backward. Clear
gradients between reference, chunked prefix and full H100 runs. After the
prefix check, run H100 in ten time chunks of 10, releasing each graph after
backward. Do not vectorize all 200 branch states or retain all time graphs.

Equivalence tolerances are fixed before GPU execution: scalar loss
rtol=2e-5, atol=1e-7; every parameter-gradient tensor rtol=3e-4, atol=3e-6.
These are FP32 numerical-equivalence checks, not scientific accuracy gates.
Do not change them after observing a failure; first diagnose any discrepancy.
Verify loss and all parameter gradients, not just a scalar gradient norm. Full-run gradients
must be finite and nonzero. Confirm weights and model buffers are unchanged.
Report unweighted normalized loss and per-channel normalized MSE; retaining
lambda=10 in a future experiment would not make this loss numerically
equivalent to the previous window-statistic objective.

## Resources, artifacts and decision

One isolated network-none container, fixed image, read-only inputs and source;
only the dedicated output directory is writable. Physical unified
MemAvailable must remain >=20 GiB under the existing continuous guard. Proposed
allocator fraction is 0.20; the final launch must bind the reviewed limit.

Output: `artifacts/true_state_paired_force_backward_probe_20261005/`.
Proposed user unit:
`fluid-control-true-state-paired-force-backward-probe-20261005.service`.
Budget: one reviewed T20 reference, T20 chunked comparison, and H100 gradient
evaluation; no hyperparameter sweep and no optimizer updates. Preserve failures
and use a new output path for any separately approved retry.

Record runtime, CUDA peak allocated/reserved, physical MemAvailable start/min/end,
loss/gradient equivalence, finite/nonzero gradients, input/source/image hashes,
and explicit no-training/no-checkpoint fields. A resource or numerical failure
requires diagnosis, not a relaxed threshold or a silently repeated run.

Compute owns implementation/execution, Evaluation independently reviews, Lead
issues final GO after exact source/launcher/test evidence is available.
