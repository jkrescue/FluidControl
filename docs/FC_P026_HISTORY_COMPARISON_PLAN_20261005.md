# FC-P026 — matched K1/K4 history-conditioned force-FNO comparison

Status: Lead-reviewed design; CPU engineering preparation approved. It does not authorize a GPU resource probe,
training, formal evaluation, PPO, frozen-data access, or real-CFD execution.

## Question and interpretation boundary

P025 is the final unsupported experiment in the isolated 96-coefficient,
statistical-loss branch. FC-P026 asks a different representation question:
does a short causal history of the cropped flow observation improve the independent
aerodynamic FNO, compared with an otherwise matched current-frame model?

The working flow predictor, real CFD data, normalization, action semantics,
downstream metrics, and acceptance thresholds remain fixed. This is not another
loss, learning-rate, scale, or architecture sweep. Both arms use the official
PhysicsNeMo two-dimensional FNO; project code only assembles and audits its input
history.

If both arms fail under the fixed finite budget below, the two candidates are
rejected. That result would **not** prove that all short histories are useless,
that the cropped state is Markov, or that a different history length, optimizer,
or larger dataset could never help. Those remain untested and must not be inferred
from this experiment.

## Authoritative P018 baseline identities

The proposal is grounded in the actual retained P018 artifacts, not a reconstructed
optimizer description.

- P018 trainer SHA: `56eed43b2f0664bfe42038602955a39eac18168045091da50e82568d3ee44a76`.
- P015 accumulation helper SHA: `2f5de1a946b040c42c21f8164da41a5afc642fd6174e7bb35372bb7ba98eb996`.
- Resolved base configuration SHA: `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`.
- P018 protocol SHA: `310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d`.
- Sampler-order SHA: `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`.
- P018 result/completion SHA: `5ca668810110676dc500f501fb602a3e2fbc1cd9a266ad6ad745484be403f926` /
  `bce5fb4688e6fc784c309a4e63979a06603270c8b1e5b336572a1092306205bf`.
- Frozen flow model/state SHA: `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` /
  `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`.
- P018 aerodynamic model/state SHA: `8a89f4774923afa698328e8e65ae337e7e0efefdae0bc9452d6b7a78758fb70d` /
  `d78d43d43738dd63b9556819e22f6b57c16993affaaff94dc3a29b6250994d6c`.
- P018 terminal aerodynamic tensor-state SHA: `6f58aea89ecdde46bfaafac0f181d2603bbc96e3bba1faa7d8a818dcf6f7984d`.
- Frozen-flow tensor-state SHA: `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`.
- Official image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Base/train8/train16 manifest SHA: `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2` /
  `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35` /
  `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b`.
- Train-only normalization SHA: `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`.

The retained P018 optimizer contract is fresh AdamW with learning rate
`1.5625e-7`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, eight
raw-window gradients averaged before one norm-1 clip/update, exactly 171 updates
over 1,368 windows, seed `20261003`, and default-TF32/high precision. The force
objective remains exactly P018: equal H1 and frozen-flow free-AR domain mass; in
each domain, `0.5 * equal-four-channel normalized MSE + 0.5 * rear-Cl normalized
MSE`. There is no field loss.

## Arms and the single intended difference

Both arms use the retained P018 frozen flow pair and begin from the retained P018
aerodynamic checkpoint bytes.

This requires an explicit correction to the legacy P018 initialization path.
`train_fcp018_reduced_rate.py` loaded **both** model instances from its single
`args.parent`, which was the P009 epoch-0 pair. FC-P026 must not reuse that loading
logic blindly. It must accept two separately bound parents: load the flow instance
from the exact P009 epoch-0 flow model/state pair above, and load the aerodynamic
instance from the exact P018 epoch-1 aerodynamic model/state pair above. The
official loader return epoch, checkpoint metadata kind, file basename, directory
confinement, and both file SHA values must be checked before tensor mapping. The
28 trainable parameter names and two frozen biases are then verified against the
loaded P018 aerodynamic model. A P009-loaded aerodynamic model, an epoch-swapped
pair, a shared generic parent argument, or a metadata-kind mismatch must fail
before optimizer construction.

### K1 control

The aerodynamic FNO receives the existing six channels:

`[q_t(u,v,p), mask, omega_t, omega_{t+1}]`.

It is a matched continuation control, not merely the historical P018 result. It
gets a fresh optimizer and the same additional 1,368-window/171-update exposure as
K4.

### K4 history arm

The aerodynamic FNO receives exactly 18 channels:

`[q_{t-3}, q_{t-2}, q_{t-1}, q_t, mask,
  omega_{t-3}, omega_{t-2}, omega_{t-1}, omega_t, omega_{t+1}]`.

Each `q` contributes normalized `(u,v,p)` and each omega is normalized by the
same fixed action scale. Past/current actions are the stored applied actions from
the HDF trajectory; `omega_{t+1}` is the same stored next command already used by
K1. No future flow or force is supplied. In free-AR rollout, the frozen flow FNO
alone advances the state, the four-state buffer shifts in that prediction, and the
stored action sequence supplies the action buffer. The aerodynamic model's three
field outputs remain discarded and never feed the state recurrence.

The official FNO architecture remains out-channels 7, latent-channels 48, five
FNO layers, modes `[32,32]`, decoder layers 2, decoder size 128, padding 8, and
coordinate features enabled. Only `in_channels` changes from 6 to 18. No RNN,
attention block, new force head, or invented PhysicsNeMo API is introduced.

The K4 lifting weight is expanded deterministically: inherited current-state,
mask, current-action, and next-action columns are copied into their declared new
positions; the extra **288 history lifting coefficients** are initialized to
exact zero. Every other tensor is copied from P018. A pre-training replay must
measure the actual K4-versus-P018 output difference under pinned default-TF32/high;
zero columns do not justify assuming bitwise equality after changing GEMM shape.

Both arms expose the same 28 trainable parameter names and keep the same two
official lifting biases frozen. K4 necessarily has the additional 288 trainable
elements in its expanded lifting tensor. Consequently this is a history
representation intervention with the minimal required input-projection capacity,
not an equal-parameter architecture comparison.

## History construction and all-window exposure

Both arms consume all 1,368 original H100 train windows in the exact retained
order: 720 base, 408 train8, and 240 train16. Batch size is one. No early window
is dropped to make K4 convenient.

The K4 inventory must be computed from the real HDF metadata before GPU approval
and must equal exactly:

- 1,300 warm windows with all three preceding trajectory frames available;
- 68 cold/padded windows touching the trajectory's left boundary.

For a requested history frame index below zero, K4 uses **trajectory frame 0**.
It never repeats an arbitrary window-start frame. Thus a window beginning at
trajectory frame 1 retains real frame 0 and pads only the earlier missing slots;
a window beginning at frame 2 retains real frames 0 and 1 and pads only the one
missing slot. The identical rule applies to stored action history. Every warm and
cold start remains in training and is separately counted in diagnostics.

For true-state H1 evaluation at relative step `r`, the four state inputs are the
corresponding real trajectory frames ending at the current physical frame, with
only trajectory-left-boundary padding. For free-AR, the buffer is seeded with the
same real/padded pre-window history, then shifted with frozen-flow predictions.
This rule must be shared by training, resource probing, validation10, dynamic6,
force-window evaluation, and eventual HydroGym reset support.

## Matched optimization and checkpoint policy

Each arm independently receives:

- one complete 1,368-window pass and 171 optimizer updates;
- the exact P018 optimizer and clipping contract above;
- the same H1/AR force objective, chunking, targets, normalization, precision,
  seed, sampler order, and train-only mounts;
- fixed diagnostic panels at consumed windows `0/456/912/1368`, used only for
  reporting;
- one terminal checkpoint only, with no early stopping, best-epoch selection,
  resume, validation feedback, or second pass.

The inherited `1.5625e-7` learning rate is intentionally retained for causal
comparability, but it is a serious sensitivity risk for the newly zero-initialized
288 coefficients. It must not be swept or raised within FC-P026. Every update must
record, separately for the new history block and inherited aerodynamic parameters:
gradient norm before clipping, applied clip scale, parameter-update norm, cumulative
displacement from initialization, and finite AdamW state. Terminal reporting must
include the learned history-block norm and a single predeclared sensitivity
ablation that resets only those 288 coefficients to their exact initial zeros,
re-evaluates the fixed train diagnostic panel, and restores the terminal model.
That ablation is explanatory only and cannot select a checkpoint or trigger a
learning-rate change.

K1 and K4 use distinct experiment kinds, output roots, terminal model/state pairs,
and manifests. Each must bind the parent pairs, architecture/history schema,
1300/68 inventory, sampler order, optimizer state with step 171, train manifests,
normalization, precision, and complete source SHA map. Official fresh reload must
reproduce all terminal tensors. The P018 artifacts remain read-only.

## Conditional CPU and GPU engineering gates

Training approval is conditional on all of the following engineering evidence.
The no-update resource probe can precede checkpoint and formal-evaluator integration
items 3 (save/reload portion) and 5: those are mandatory before full training, not
dependencies of measuring the shared objective's full-size memory footprint.
Parent loading, tensor mapping, unchanged trainable scope, actual data alignment
and objective equivalence remain mandatory before the no-update probe.

1. Real-HDF enumeration reproduces all 1,368 identities, the original order SHA,
   family counts, and the exact 1,300 warm / 68 padded partition.
2. Unit tests cover frame-0 left padding at starts 0, 1, and 2; real past-frame
   retention; stored-action indexing; H1 target alignment; AR buffer shifting;
   and rejection of future-state/force access.
3. Official PhysicsNeMo CPU tests instantiate K1 and K4, map the P018 tensors,
   verify the exact 28-name trainable scope/two frozen biases, verify all unchanged
   tensor bytes, and save/fresh-load both identities.
4. The shared prediction adapter proves that identical K1 behavior retains the
   old raw-output/residual/pooling semantics, while K4 changes only aerodynamic
   input assembly; frozen-flow trajectories must be identical across arms.
5. Formal evaluator tests exercise K4 history at every H1, free-AR, and reset
   caller. Passing a K4 checkpoint through a legacy six-channel evaluator must
   fail closed.

After the applicable checks above, a separately approved no-update GPU resource probe uses one
fixed real H100 train window per arm. It executes the complete H1 and free-AR
forward/backward path, creates no optimizer or checkpoint, and records elapsed
time, CUDA allocated/reserved memory, host MemAvailable/MemFree, model hashes,
and all finite gradients. The new K4 history block must have a finite, nonzero
gradient. Both physical memory floors remain 20 GiB.

The probe determines an arm-specific wall-time cap. Scientific fairness is equal
data and optimizer exposure, not an artificially equal number of seconds. P018's
four-hour per-arm ceiling is the provisional upper bound; if K4 cannot complete
within the reviewed resource envelope, the protocol returns to Lead rather than
dropping windows, changing batch size, or changing the learning rate.

## Unchanged complete development evaluation

Training completion is not admission. After terminal integrity and independent
audits, both arms must run the original complete validation10, dynamic6,
force-window, and development-admission suite with the original datasets,
horizons, starts, aggregations, and thresholds.

The numerical metric code stays unchanged, but K4 evaluation is **not** a blind
reuse of the legacy six-channel caller. Every H1, free-AR, and reset path must use
the reviewed K4 history adapter and the exact training-time frame-0 padding/action
semantics. Receipts must bind the history schema and both checkpoint identities.
Field predictions must remain those of the frozen flow model and should be checked
for equality across arms; aerodynamic predictions are evaluated from the actual
saved K1/K4 model, not substituted into old JSON.

Interpretation is fixed:

- If K4 fails the original complete admission, it does not enter PPO even if it
  improves over K1.
- If both fail, reject these two finite-budget candidates; do not claim a universal
  refutation of short-history conditioning.
- If K4 passes and K1 fails, the result supports this exact K4 intervention, but a
  compatible PPO and real-CFD feedback still require separate approval.
- If both pass, the comparison does not establish a need for history; prefer the
  simpler K1 unless later evidence justifies K4.
- Warm and padded/cold starts must be reported separately. Improvement confined
  to warm starts cannot be presented as a reset-safe control surrogate.

No threshold changes, extra epochs, K sweep, learning-rate sweep, new CFD,
validation-based selection, frozen-test access, PPO, or real-CFD run belongs to
FC-P026.

## Practical risks

1. The tiny inherited learning rate may leave the zero-initialized history block
   effectively unused. Recorded gradients, update norms, cumulative displacement,
   and the terminal zero-block ablation distinguish "no learned use" from evidence
   against the underlying history hypothesis; they do not authorize a retry.
2. K4 has 288 additional lifting coefficients, so a gain cannot be attributed to
   information alone with mathematical purity. The expansion is the minimum
   project adapter required by the official FNO input contract and is disclosed.
3. Overlapping windows represent only 44 CFD trajectories and limited source
   phases/action families. Equal 1,368-window exposure does not create 1,368
   independent physical samples.
4. Reset padding is deployment-relevant. Formal and future HydroGym code must
   reproduce trajectory-frame-0 padding exactly; silently using a window-start
   repeat or unavailable prehistory changes the experiment.
5. Running arms on different nodes may change timing but must not change image,
   precision, code, data bytes, or numerical protocol. Any hardware-specific
   deviation is recorded, not normalized away by altering the budget.
