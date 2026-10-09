# B fixed-small-panel fitability diagnostic — preparation only

No GPU execution, candidate promotion, development evaluation or CFD is authorized by this draft.

## Evidence and distinct question

E105 (`docs/P064_K1BG_B00_TRAIN_FIT_TERMINAL_REVIEW_20261007.md`) measured K1/B/G on 40 true-state points with 120 aerodynamic forwards and zero optimizer updates. It did not test convergence. P016 did train all 28 aerodynamic tensors on six fixed H100 windows: P009 parent, mixed H1/AR objective, AdamW 1e-5, exactly 32 updates. H1 increased 0.175% while AR decreased 3.803%; it was not a fit-to-error stopping experiment. P017 found local first-Adam-step overshoot, not network incapacity. Earlier FC-P003C 128/64 calibrations used a different joint field/force model and competing regular/paired streams. Absolute64 repeated B's full schedule twice; none establishes sufficient fitting of the current model on a fixed small true-state panel.

Question: can the current frozen-B aerodynamic FNO fit this small, fixed, already-used training panel under a bounded deterministic optimizer, without the AR-state objective? This is not another development candidate or a test of action causality.

## Fixed data and model

- Reuse E105 b00 train-only origins 0,100,...,700, five true-state transitions each: exactly 40 points, no held-out or development access. Bind the E105 result `bb256396fe174f7d1f8ef3cd68a06bc1653c0e8ff039604714479f44391da26c`, actual HDF/manifest/norm and parent/source SHAs before execution.
- Input q_s and recorded omega_s/omega_(s+1), target force_(s+1), original normalization/mask/action planes. No current-force input. Full target/input identities must match the previous diagnostic.
- Start from B, not fresh K1 or the new auxiliary candidate. Unchanged official FNO: six input channels, SEVEN output channels; discard the three field outputs and mask-pool the four force outputs exactly as before. Train the same 28 tensors, freeze the two named lifting biases. Flow model is not forwarded; no architectural edit.
- Original balanced normalized four-force absolute H1 objective, weights (0.125,0.125,0.125,0.625). No AR term, auxiliary penalty, weight decay or gradient clipping. This is an optimization diagnostic, not an equal-optimizer comparison against B.
- Deterministic fixed ordering and microbatch10; four microbatches accumulate the exact mean over all40. Same fixed precision for every objective, gradient and line-search trial; no stochastic augmentation/dropout or changing batch partition. Proposed highest/noTF32 must be explicitly recorded as the E105 diagnostic profile, not bitwise original-B training precision.

## Bounded optimizer and stopping

Use installed PyTorch LBFGS, strong_wolfe, lr=1 (initial line-search scale), history_size=5. No custom optimizer or LR sweep. Proposed implementation uses max_iter=1 per outer call to enable an explicit accepted-point audit; retain the SAME optimizer/history across calls. Global hard bounds: 200 outer calls, 300 full-panel gradient closures, 1200 seconds, whichever occurs first. Count actual optimizer-state n_iter increments and parameter displacement separately; an outer call is not automatically an accepted update. Every closure is four aerodynamic forwards/backwards; at most 1200 such forward/backward microbatches. Each returned step requires a separate no_grad full40 evaluation (up to 200 panels/800 microbatch forwards), plus initial/final panels. Do not report the loss returned by LBFGS.step as the updated-point loss.

Fit target: all four normalized channel RMSEs <=0.01 on the same40 points. This is an arbitrary preregistered train interpolation target, not a physical threshold, formal predictor gate or generalization claim. Record every accepted-point objective, channel normalized RMSE, physical MAE/RMSE/bias, gradient infinity/L2 norms, displacement, step size, closure count and elapsed time. Also report totalCd and rearCl physical errors. A small-step/gradient optimizer stop without meeting the fit target is local stagnation, not success.

Before each outer call, preserve last accepted parameters and optimizer state. A closure-budget/deadline exception inside strong_wolfe must restore that accepted point, never leave a trial point labelled final. Nonfinite loss/gradient or resource failure similarly records the failure and restores the prior accepted state where safe; no automatic retry. Budget exhaustion is `BUDGET_STOP_NOT_FITTED`, not proof of inadequate capacity or missing input. No checkpoint/candidate save; retain scalar curves and initial/final per-point predictions, input hashes and model digests.

## Runtime readiness still required

Read-only inspected installed torch 2.14.1+cu130 `torch/optim/lbfgs.py`: `_gather_flat_grad`, `_numel`, `_add_grad` explicitly support complex parameters using real views. Actual official model parameter dtypes/counts still need a preparation-only check; no inference/optimization has run. One device and one parameter group are required. History storage holds both s and y vectors; estimate from actual parameter bytes, not a nominal 12GiB budget. Include history5 (10 parameter-sized vectors), gradient/direction/work buffers, last-accepted model+optimizer snapshot, activation peak and runtime overhead. Propose 24 or32GiB only after this estimate, noSwap, CPU4, bounded allocator, startupAvailable50GiB/runtime22GiB/reserve20GiB. Final supervisor and exact CPU tests require separate review before GPU authorization.

## Interpretation

Meeting the target establishes only local memorization/fitability of this panel by this architecture/representation and optimizer. It does not establish dynamics, action-response causality, held-out accuracy, control benefit, or model admission. A falling curve at the cap means unresolved optimization budget; a plateau means this procedure stalled, not that all procedures or the architecture must fail. B controller and original prediction/physical gates remain unchanged. No new dev/PPO/CFD follows automatically.
