# P064 b04 long-excitation coverage proposal (preparation only)

Date: 2026-10-07. This is a single, executable research proposal. It does not authorize CFD generation, conversion, training, inference, PPO, or control.

## Hypothesis and why this is the remaining useful data test

The testable hypothesis is that the short-horizon aerodynamic-force error is limited by coverage of **late, feedback-like states under changing bounded actions**, rather than by the amount of repeated b00/b02 feedback data or by a small loss/learning-rate change. The proposed intervention is one longer persistently excited trajectory at an already train-only phase, followed by the unchanged P064 training and development protocols.

This is deliberately not described as missing action coverage. The retained B run already sampled eight `dynamic_train8_b04_prbs` and eight `dynamic_train8_b04_multisine` windows among its 192 original-data windows. Existing negative evidence also includes C50 data-dose failure, D b00/b02 coverage failure, E four-state response-auxiliary failure, F H1-only H5/AR regression, G AR5-reset retention failure, H25 degradation, lag +/-1 failure, and persistence of the selected response-sign errors at highest/no-TF32 precision. The narrower unresolved question is whether the current 20-second b04 excitation exposes too little late-time state diversity.

This proposal cannot establish a pure causal effect of phase, behavior policy, or trajectory length. Relative to B, the eventual training schedule would jointly replace part of the controlled source profile and change phase, action program, and visited states.

## Fixed train-only source and observed coverage

Use phase `b04`, restart time `120.0`, from `matched_start_acquisition_train_b04_zero`. It is in the immutable train split (`splits/train.json` SHA256 `1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89`), not the opened b01/b03 development split (`splits/validation.json` SHA256 `2bf348b5b7a2d1d902078470ec02065cdc5b794de0d8c16be977e6df54e7d54f`). The source restart configuration SHA256 is `9c4706a7457ab0a1c6126100fd901da7818c973ded6417f4ed8e170785f07c20`.

The existing dynamic-train manifest SHA256 is `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35`; the original predeclaration SHA256 is `4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a`. Read-only HDF inspection produced:

| existing b04 train trajectory | frames / interval | omega range | omega RMS | unique omega (1e-6) | nonzero-delta fraction | max abs delta | sign reversals | longest exact hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m075` | 801 / 120-200 | [-.75, 0] | .74742 | 9 | .010 | .100006 | 0 | 793 |
| `p075` | 801 / 120-200 | [0, .75] | .74742 | 9 | .010 | .100006 | 0 | 793 |
| `zero` | 801 / 120-200 | [0, 0] | 0 | 1 | 0 | 0 | 0 | 801 |
| `multisine` | 201 / 120-140 | [-.26558, .47016] | .21846 | 200 | 1.000 | .060859 | 6 | 1 |
| `prbs` | 201 / 120-140 | [-.75, .75] | .41477 | 57 | .490 | .100006 | 7 | 38 |

The two existing b04 excitation HDFs are SHA256 `d7bbfbf1d18f40cb613f2564fabf3eafb95afbf3ee7635e29f9c20ae27f6e48b` (multisine) and `012eea462332b06298cc2128a47e1705ec9d6ec4effc08569bd73cd51970df0d` (PRBS). Thus another 20-second PRBS repeat would be weak and is not proposed.

## One fixed intervention

Generate one 801-frame b04 trajectory from `120.0` through `200.0`. Repeat the already predeclared 20-second `dynamic_train8_b04_prbs` action table four times, with the shared zero endpoints left intact. This keeps the observed action distribution, `|omega| <= .75`, `|delta omega| <= .1` per 0.1 interval, solver, mesh, boundary conditions, and phase restart fixed; the intended new information is the same excitation at later evolved states. Predeclare the complete 801-point table and all source hashes before solver execution.

Use the existing official path: OpenFOAM raw fields/forces -> PhysicsNeMo Curator source/filter/sink -> HDF5 -> official PhysicsNeMo HDF5Reader/DataPipe. Reuse the original train-only normalization bytes (`f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`); do not refit. Do not read or train on b01/b03 development data.

If and only if the generated trajectory and train-only view pass independent source, endpoint, action, force, official-reader, and first/last H100-window checks, train one final-only candidate:

- fresh K1 parent and the same official dual-FNO architecture;
- frozen flow FNO and the same two frozen aerodynamic biases; the remaining 28 aerodynamic-FNO parameter tensors train;
- unchanged `.5*H1 + .5*continuous-AR100`, precision, AdamW, clipping, accumulation, seed, learning rate, and 32 updates x 8 windows;
- preserve all 192 original B schedule entries exactly;
- controlled slots are fixed at 32 b00 windows plus 32 new long-b04 windows, each with mechanical starts `i*700//31`, `i=0..31`, at the same two within-update slots used by the reviewed D schedule.

This is a controlled-source-profile replacement at fixed total 25% controlled dose, not a continuation, checkpoint selection, or longer optimization budget. The retained B result SHA256 is `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`.

## Fixed evaluation and decision

Reuse the already opened, unchanged b01/b03 development panel: 16 origins, H1-H5, highest/no-TF32, identical actions/truth/B comparator. Report every phase, lead, persistence comparison, field metric, and the existing fixed-six continuous-AR panel.

The same bounded choice rule applies: pooled H1 rear-Cl MAE and total-Cd MAE must both be strictly below B, and fixed-six H1 and AR objectives must both be no higher than B. Otherwise retain B; do not extend training, select an intermediate checkpoint, launch PPO/CFD, or tune the excitation after seeing development results.

## Cost and stop rule

Expected wall time from prior actual runs is about 18 minutes for paired-quality CFD generation, 6-10 minutes for conversion/view verification, 19 minutes for the fixed training, and under one minute for fixed development inference: approximately 43-50 minutes before independent reviews. One failed engineering attempt may be diagnosed but not blindly retried. Given that bounded PRBS/multisine coverage already exists, this experiment should be approved only if the remaining surrogate-accuracy objective is worth that cost; it is not required for the already demonstrated B-policy real-CFD closed loop.
