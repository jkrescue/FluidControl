# Candidate G: training-only five-step frozen-flow resets

Preparation only; no GPU training authorization. This is a project training
input-distribution change, not a new official PhysicsNeMo feature or architecture.

## Evidence and falsifiable question

E100 pairs the same B/F true inputs and shows H5 true-state forces are not
uniformly better than saved autoregressive forces. It therefore does not establish
that flow error alone causes force error. F improves pooled H1 but degrades H5
and original fixed-six continuous-AR retention. Test whether retaining mixed
supervision with short autoregressive exposures improves the existing H5 use
without sacrificing the original H1 / long-AR retention. No claim of exact PPO
state-distribution matching: these are recorded training actions, not fresh PPO
rollouts. Resetting changes error cancellation as well as state error magnitude.

P008/P009 calibrated P003C affine force rows on H1 / continuous-H100 features;
they did not execute this K1 fresh, 28-tensor aerodynamic-branch experiment.
The H25 experiment trained the flow branch with the force branch frozen, unlike G.

## Fixed single candidate

Start at the exact original B K1 checkpoint, not B or F continuation. Preserve
B's 256-window order (192 original and 64 b00), seed 20261003, 32 AdamW updates,
8-window accumulation, learning rate 1.5625e-7, clip 1, official FNO/DataPipe,
normalization, runtime precision and batch layout. Train the same 28 aerodynamic
tensors; both designated biases and the separate flow FNO remain frozen. No E
auxiliary loss. No data creation, heldout opening, horizon sweep or extension.

Only training AR-current-state generation changes. Within each 100-target window,
at s=0,5,...95 take the normalized observed q_s; then use unchanged residual-flow
updates and omega_s -> omega_(s+1). Save pre-update states for force targets
1..100. Keep all 100 frozen-flow calls, including the 20 block-end updates
discarded by the next reset. This preserves B's training call budget: 25,600 flow
calls, 2,560 mixed aerodynamic calls (10 H1 + 10 AR samples per call), 25,600
H1 and 25,600 AR supervised points. These are overlapping training exposures,
not independent CFD samples. No cache build is added: B generates states online.
More precisely, 19 block ends are discarded at the next reset and the last q100
is unused, as it already was in B: total 20 discarded updates versus B's one,
not 20 additional calls. The frozen flow branch cannot improve field forecasts.

## Minimal code boundary

Base: reviewed immutable B runner 8066f4a1; run() at lines 544–577.
Training backward=True calls new p064_ar5_reset_states.training_states in place
of train_fcp013_independent_force_fno.frozen_flow_states (line 106).
Original p026_history_objective.chunk_force_objective stays byte-identical:
ten mixed20 calls, .5 H1 + .5 AR loss, original gradient averaging/clip/Adam.
Diagnostics backward=False continue the ORIGINAL continuous100 state generator.
Per-window audit saves reset indices, 100-call count, generated-state SHA and
existing input identity. Protocol distinguishes short-reset training AR from
continuous100 diagnostic AR. A distinct candidate kind/consumer must be wired
before any execution; do not disguise G as B.

## Evaluation and decision fixed before execution

Reuse B's existing comparison; do not retrain B. Run the same opened-development
b01/b03 panel (16 starts x H1–H5) once after a verified terminal candidate.
Only consider progression if BOTH pooled H1 rear-Cl MAE and total-Cd MAE are
strictly below B, and BOTH original fixed-six H1 and continuous100 AR objectives
are no worse than B. Report every phase/lead, field and force metric, including
H5 deterioration, even if the narrow rule passes. This is a candidate-selection
rule, not scientific admission or evidence of physical control benefit.

## Budget and safeguards

Same B resource envelope: 12 GiB/no swap, allocator .06, CPU800%, startup
Available >=50 GiB, runtime >=22 GiB with 20 GiB reserve, 3600 s inner / 3660 s
outer. F's actual last resource sample was about 999.6 s; ~17–20 minutes is a
reference, not a promise. No increased model-call count, no persistent state
cache, same-shaped in-memory tensors. Official imports/source/data/parent hashes,
actual loader/save compatibility, CPU integration and independent review must
precede a separate execution approval. Existing successful controller unchanged.
