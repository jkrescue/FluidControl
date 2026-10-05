# FC-P014 — fixed-train objective and lift-residual decomposition

Status: implementation and CPU tests authorized by Lead; GPU execution is NOT
authorized by this document. Written after the P013 fixed-six result was observed,
before this new diagnostic. This is not retrospective preregistration of P013.

## Question and evidence

P013 fixed-six result SHA
`8e0255c955c1b26fdff240a0854fc0a92d3bd247cc38ed6c268fc1d397cec873`
shows H1 rear-Cl MAE worse on six windows, AR worse on five, H1 mean-lift error
worse on six, but H1 tail centered-RMS error improves on four. Fields are identical.
Question: does increased signed mean residual explain the deterioration, or does
centered waveform disagreement also increase? Under the actual P013 objective,
do H1 and AR improve in opposite directions or deteriorate together?

P009/P011 failed formal lift windows; P012 did not support its predeclared strong
gradient-scale/conflict conditions. P014 will not claim a unique optimization
cause, introduce a loss-weight sweep, change architecture, or replace formal gates.

## Fixed identities and execution scope

- Scene remains fixed tandem Re100, L/D5, rear rotation only.
- Flow and baseline force predictor: P009 official model/state SHA
  `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` /
  `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`.
- Terminal force predictor: P013 r2 model/state SHA
  `2eb4dde99c5f1a8ba8d8f2821c3871e906a6f13dd3d01f1ed1e8eba5379f1a03` /
  `b0d859e6576c679bc7c09812bff6d55db6341253061236a45e1c52b9e17993ca`.
- Dual manifest SHA `23d2917ef038196f066934f51d0be2770c23b26099eb7ab97439ce58488e13bb`;
  training-result SHA `239f6567d662157e8f0bec8b1277cc219f7580ac3dde3238657dea8d6b4821d0`.
- Resolved config SHA `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`;
  normalization SHA `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`.
- Reuse source snapshot `artifacts/fcp013_training_source_1634c05_immutable`.
  P013 objective SHA `f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7`;
  P011 windows/data SHA `9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5`.
  The script verifies these and its direct numerical dependency hashes.
- Official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
  Default TF32/high, float32 forwards, batch one window, seed 20261003.
  Never change precision to obtain numerical agreement.

Frozen flow remains in eval mode. Each force predictor temporarily uses train
mode, matching P013 aerodynamic.train(), then is restored. torch.no_grad() is
not eval mode. Reject BatchNorm/Dropout and verify unchanged model tensors.
Earlier fixed-six diagnostics used eval mode and a different forward schedule;
record the difference and do not claim bitwise replay.

No optimizer, backward, clipping, updates, model saves, checkpoint selection,
held-out data, PPO, or CFD. Only the diagnostic JSON may be written to a new output.
The same two immutable checkpoints are compared; no continuation is proposed here.

## Exact train panel

Execution prerequisite: the approved launcher verifies the pinned P013 candidate
audit and recomputes all 44 train HDF SHA values from its train_hdf_sha256 mapping,
recording the verified mapping and receipt. The diagnostic checks data manifests
and normalization but does not repeat this full-file read. No validation/frozen
HDF file may be mounted. Approval and resource guard must prevent overlap with
the running formal GPU suite; --execute is not execution approval.

Reuse P011 `diagnostic_windows()` and assert these ordered identities:

| Index | Family | Case | Start |
|---:|---|---|---:|
|160|base20|matched_start_acquisition_train_b00_zero|320|
|816|train8|dynamic_train8_b00_prbs|90|
|923|train8|dynamic_train8_b02_prbs|100|
|975|train8|dynamic_train8_b04_prbs|0|
|1077|train8|dynamic_train8_b06_prbs|0|
|1233|train16|direct_cfd_directppo2048_v1_env0_ep0009_b00|0|

All are train H100 windows from existing DataPipe composition. Preserve action
indexing, force-channel order, physical scaling, and mask pooling. Five nonzero
history windows are reported separately from zero; no population inference from
six correlated train windows and no new acceptance count.

## Computation and minimum outputs

For each window compute frozen P009 flow states once with P013
`frozen_flow_states()`, and H1 inputs with `true_state_inputs()`. Reuse these exact
inputs for P009 force and P013 force. For each predictor call the immutable
`chunk_force_objective(..., chunk_size=10, backward=False)` under no-grad. Its ten
forwards each contain **20 examples: H1 endpoints 0:10 followed by AR 0:10**, then
the next ten endpoints. Do not substitute the separate batch10 H1 and batch1 AR
schedule of the earlier physical diagnostic. A recording wrapper around the same
`predict()` call only captures returned force tensors and returns them unchanged.

Retain original float32 chunk-weighted H1/AR balanced losses, four channel MSEs,
and total `0.5*H1+0.5*AR`. Balanced loss is
`0.5*mean(four normalized channel MSEs)+0.5*rear-Cl normalized MSE`.
No field loss. This exact objective is evaluated for both parent and terminal;
the earlier shuffled-window training history is not a convergence curve.

For each domain retain 100 normalized rear-Cl residuals and physical residuals
`e=(prediction_normalized-target_normalized)*rear_Cl_std`, computed in float32
then reduce on CPU in float64. This may differ in float32 from separately
denormalizing predictions and targets; no bitwise physical equivalence is claimed.
For all H100 points and tail62 indices `[38:100]`, report signed residual mean b,
MSE, RMSE, bias MSE b², centered residual MSE, centered residual RMSE, and numerical
decomposition residual `MSE-b²-centered_MSE`. No decomposition tolerance is an
admission criterion. Centered residual RMSE is waveform disagreement, not the
difference between two centered RMS amplitudes used by the formal lift gate.

Also report an **analytic**, not autograd-measured, normalized rear-Cl output-bias
derivative per H100 domain: `1.25*mean(normalized_residual)`; combined derivative
is the equal-domain mean. This follows final affine bias and masked mean pooling
on a nonempty binary mask. No backward is performed; this is a local coordinate
diagnostic, not full-model stationarity or optimizer convergence.

Result contains 12 model/window rows, source/manifest/result/config hashes,
precision, window identities, per-row parent-to-terminal objective and component
deltas, and before/after tensor hashes. All scalar/vector entries must be finite.
No aggregate score chooses a model or overrides a regression. Record batch schedule
explicitly; do not expect bitwise equivalence with the earlier different schedule.

## Predeclared interpretation, not admission

- MSE increases, bias MSE increases, centered MSE does not: observed deterioration
  on that panel is attributable to increased mean residual in this decomposition.
- Both components increase: mixed offset/waveform deterioration.
- Centered MSE increases without increased bias MSE: waveform deterioration.
- Opposite signs of H1/AR objective changes: observed domain tradeoff. Both worsen:
  no compensating tradeoff on this panel. Zeros remain zeros, not rounded signs.
- Opposing analytic bias derivatives: local coordinate tension between domains;
  matching signs: common local bias-descent direction. Neither proves why training
  reached the terminal point, global convergence, or that an extra update helps.
- Report all numerical values, signed changes and zero/nonzero panels; do not add
  magnitude, majority, convergence, or scientific admission thresholds.

Next action is independent review of the finite, identity-verified decomposition.
Any optimization intervention needs a separately approved single hypothesis and
budget. Original complete formal evaluation remains authoritative and unchanged.

## Proposed execution budget and approval boundary

Before execution Lead binds final script/test/plan hashes in a separate approval,
requires P013 completion and fixed-six receipts, and verifies existing train-file
identities. Use an immutable launch copy, train-only mounts matching P013, no
validation/frozen mounts, image above, allocator <=0.15, GPU0, MemAvailable >=20 GiB
and MemFree >=20 GiB (checked before GPU initialization and each window),
40-GiB container cap, 1200-second wall cap, no concurrent conflicting GPU work.
Budget: six frozen-flow H100 rollouts plus 120 mixed-batch20 force forwards;
two official FNO instances, no autograd graph. No retry changes numerical protocol.
Identity, finite-value, tensor-mutation, resource or timeout errors preserve the
failed run and require diagnosis. GPU execution remains unapproved at implementation.

Future container command (not authorization; use the established train-only mounts):

```bash
python /workspace/diagnostic/diagnose_fcp014_train_objective.py \
  --source-root /workspace/project --config /workspace/config.yaml \
  --dual-manifest /workspace/dual/dual_model_manifest.json \
  --training-result /workspace/dual/result.json \
  --output /workspace/output/result.json --execute
```

Default invocation without `--execute` verifies only source/checkpoint/result/config
file identities and does not import the GPU runtime, read HDFs, or write an output.
