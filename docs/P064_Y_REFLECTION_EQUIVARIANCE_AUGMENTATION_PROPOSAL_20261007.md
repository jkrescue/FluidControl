# P064 physical y-reflection paired-loss proposal (preparation only)

Date: 2026-10-07  
Status: **proposal only — no implementation, training, inference, PPO, or CFD is authorized**

## Decision in one sentence

Test one representation/data hypothesis: retrain the existing official PhysicsNeMo aerodynamic FNO from the same K1 parent and the original B schedule, but give every original training window equal loss weight with its physically y-reflected counterpart. Everything else stays fixed. This is a falsifiable test of whether the missing reflection constraint contributes to the rotating-action force error; it is not a claim that the current model or the canonical control adapter is already reflection equivariant.

## Why this is the next bounded experiment

The saved evidence does not support another update-count, loss-weight, horizon, or controlled-data-ratio sweep:

- The formal B review remains `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`. Its failure is concentrated in rotating branches' force statistics: the two zero-action branches pass, while only 2/6 branches jointly pass the force-window criteria. This is a force-statistics failure, not a newly invented H100 field-L2 threshold (`P064_B_FORMAL_TERMINAL_REVIEW_20261006.md`).
- Teacher-forced H1 improves over autoregression on all four rotating branches, yet material H1 force residuals remain (for example, rear-Cl RMS error magnitude 0.0614688 on b01+ and 0.0370161 on b05-). Thus replacing predicted states with truth does not remove the force-readout error (`P064_TEACHER_FORCED_H1_TERMINAL_REVIEW_20261006.md`).
- On fixed b00 training samples, K1/B/G true-state rear-Cl MAE is 0.200856927/0.181738779/0.179079254 and total-Cd MAE is 0.051195516/0.050531504/0.046448530. Training helps, but leaves substantial in-sample force residual; the issue cannot be described only as development generalization (`P064_K1BG_B00_TRAIN_FIT_TERMINAL_REVIEW_20261007.md`).
- Extending B from 32 to 64 updates improves the fixed-development pooled H1 Cl/Cd MAE from 0.1389983/0.0380654 to 0.1268953/0.0362466, but regresses the fixed-six H1 and continuous-AR objectives. H1-only, AR5-reset, H25, and controlled-dose variants also failed their predeclared retention/selection rules. More of the same training is therefore not the proposed intervention (`P064_ABSOLUTE64_DEVELOPMENT_REVIEW_20261007.md`).
- The current training DataPipe and B trainer contain no physical reflection/flip augmentation. Historical `sign_flip` use is a counterfactual evaluation operation, and P020's symmetric-tail term constrains force statistics; neither creates a reflected state/action/target training pair.

This evidence makes reflection pairing worth one test, but does not prove it is the cause of the residual. Frozen-flow predictions are not known to be reflection equivariant, and successful canonical policy coordinates do not establish surrogate equivariance.

## The one intervention

For each of the original 256 B windows, construct exactly one physical y-reflected window and use

`L_window = 0.5 * L_original + 0.5 * L_reflected`.

The original objective inside each term remains `0.5 * L_H1 + 0.5 * L_AR`. Keep the same K1 initialization, original B shuffle/order and seed, 32 optimizer updates, accumulation over eight **original** windows, learning rate, Adam settings, clipping, precision, train-only normalization, official FNO architecture, and final-only checkpoint rule. Train the same 28 aerodynamic-FNO parameter tensors; keep its two frozen biases and the separate flow FNO frozen.

This schedule has 256 original windows but 512 transformed-window equivalents and approximately twice B's training-model forward/backward work. In the current chunked implementation this means 51,200 frozen-flow calls and 5,120 aerodynamic calls on mixed H1/AR batches of 20, excluding the unchanged diagnostic panels. Implement the pair serially—backward `0.5 L_original`, release that graph, then backward `0.5 L_reflected`—inside the existing eight-window accumulator, before its one divide-by-eight/clip/step. Scale each branch's newly produced backward contribution (the current objective performs ten internal backward calls); do not repeatedly scale the already accumulated `.grad`. Hooks, if used, must cover exactly the 28 trainable tensors and be removed in `finally`, including on an exception. Do not silently reduce windows, alter accumulation, or put both graphs in memory simultaneously. If this exact serial contract does not fit the reviewed resource envelope, report the engineering block rather than changing the experiment.

For the reflected AR term, start from the reflected physical q0 and roll the same frozen flow model independently with reflected actions. Do **not** obtain the reflected trajectory by flipping the original predicted trajectory: that would assume the unproven flow equivariance and would be a different intervention. FNO's internally generated coordinate features remain the destination-grid coordinates; they are not treated as state channels and are not sign-flipped.

## Exact physical transform

Reflection is about the centerline `y = 7.5` on the existing `0 <= y <= 15` domain. On the sampled `[channel, y, x]` grid:

- reverse the y index;
- state `[u, v, p]` signs are `[+, -, +]`;
- both `omega_now` and `omega_next`, and every action-history entry, change sign;
- forces `[front Cd, front Cl, rear Cd, rear Cl]` signs are `[+, -, +, -]`;
- target states and any state history receive the same spatial/channel transform;
- the mask is y-reflected, and masked normalized cells are reset to exactly zero.

The transform must be performed in physical units and then passed through the unchanged train-only normalization. It is invalid to negate normalized `v` or `Cl` directly because the stored means are nonzero. Equivalently, an odd normalized scalar `z` transforms as `-z - 2*mean/std`, followed by masking, but the implementation should prefer denormalize → physical reflection → original normalize. Do not recompute normalization statistics.

## Required CPU contracts before any training approval

The implementation is not ready for GPU review until all of these pass on actual project data/configuration:

1. Geometry and boundary symmetry: both cylinder centers are on `y=7.5`; the combined `upperLower` patch has `U: slip` and `p: zeroGradient`; inlet is `(1,0,0)` and the outlet/other paired boundaries are compatible with y reflection. Record exact source paths and hashes.
2. Grid correspondence: sampled y coordinates pair exactly under `y -> 15-y`; x is unchanged. The train mask is exactly equal to its y reversal. Check every training source, not only one representative HDF file.
3. Involution: the pure physical-array y flip/sign map must restore indices, mask, actions, and physical arrays bitwise after two applications. The float32 denormalize/renormalize path is checked separately with a predeclared dtype-scaled tolerance (`atol = 8 * eps_float32 * max(1, max_abs_physical_value)`, `rtol = 8 * eps_float32`), and its observed maximum absolute/relative errors are recorded. Masked normalized cells must remain bitwise zero. A grid/index or mask mismatch is still an exact failure; floating-point roundoff must not be misreported as physical asymmetry or used to loosen those exact checks.
4. DataPipe equivalence: an actual official-reader fixture proves that reflecting physical data and then using the existing project `TandemRolloutDataset` normalization/input builder yields the declared inputs/targets. `TandemRolloutDataset` is a project adapter using the official PhysicsNeMo reader/DataPipe, not itself an upstream PhysicsNeMo class.
5. Temporal causality: H1 uses reflected current truth and reflected next action; AR carries only reflected prior predictions/actions. No future target enters an input. Preceding state/action history, if present, is transformed consistently.
6. Loss/gradient accounting: an actual small-model integration test using the existing accumulator proves the two half-losses equal their explicit paired reference, a pre-existing nonzero gradient is not rescaled, hooks are removed after failure, the accumulator still divides by eight original windows, clipping/Adam step occur once, and old B behavior is byte-for-byte/numerically unchanged when augmentation is disabled. Each reflected branch constructs its own 100-step frozen-flow history from reflected q0/actions; the original branch's flow history cannot be reflected or reused. The fixed diagnostic panel remains the original unaugmented `run(..., backward=False)` path and reports the original H1/AR components, not a reflected average.

These tests can falsify the proposal before spending GPU time. A boundary, grid, or mask asymmetry is a hard stop, not something to hide with tolerance or a modified loss.

## Fixed evaluation and decision rule

Evaluate only the final update-32 checkpoint; do not choose an intermediate checkpoint. Reuse the already opened protocols and B comparator without changing inputs, precision, metrics, or thresholds:

- fixed development: b01/b03, 16 origins, H1–H5, highest/no-TF32;
- fixed-six retention: the existing six cases, teacher-forced H1 and continuous AR100, original high/TF32;
- report all phases, leads, persistence comparisons, field metrics, and all four force components.

The candidate supports the hypothesis only if pooled H1 rear-Cl MAE **and** total-Cd MAE are both strictly below B, while the fixed-six H1 and AR objectives are both no higher than B. This is the existing bounded engineering choice rule, not a new physical acceptance threshold. A small secondary mechanism table may report `model(Rx)` versus `R model(x)` on the same fixed inputs, but it is descriptive and cannot override the primary rule.

If any condition fails, retain B, record the reflection hypothesis as unsupported at this budget, and do not extend training, tune the pair weight, select a checkpoint, or launch PPO/CFD. Passing this development rule would still require a separately approved real-CFD control test; it would not retroactively pass the full prediction program.

## Budget and limits

- Scientific budget: 32 updates, 256 original windows, 512 transformed-window equivalents, eight original windows per accumulated update, final-only save.
- Expected compute: approximately twice B's aerodynamic training work; use 40 minutes as the planning estimate and 60 minutes as the hard execution limit.
- Proposed engineering envelope, subject to source review: GPU0, 16 GiB CUDA allocator cap, 24 GiB host `MemoryMax`, no swap, CPU quota 800%, startup `MemAvailable >= 50 GiB`, runtime guard `>=22 GiB`, and 20 GiB reserve; inner/outer/stop limits 3600/3660/20 seconds.
- No new CFD data, no development data in training, no normalization recomputation, no architecture/width/mode change, no flow training, no loss-weight or seed scan, and no automatic PPO/CFD.

This proposal is deliberately one experiment. Its useful negative result would be that exact physical reflection pairing does not jointly improve force H1 and retained AR performance under the original B budget, ruling out this missing constraint as the next practical remedy.
