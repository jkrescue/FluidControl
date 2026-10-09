# Control-force loss and gradient-scale audit

Date: 2026-10-05. Scope: read-only analysis and a conditional CPU/GPU diagnostic design. This note does not change a trainer, approve a training run, alter a gate, or use validation/frozen data to tune a loss coefficient.

## Facts established from the current implementation

The regular H100 objective in `scripts/train_tandem_fno_paired_stats.py` is

```text
L_regular = L_field + 0.2 L_force
```

with rollout discount 1.0. `L_force` is computed in train-normalized force space and uses normalized channel weights `[1, 1, 4, 1] / 7` in the order front Cd, front Cl, rear Cd, rear Cl. The train-only force standard deviations are respectively `0.0101904263`, `0.314410444`, `0.190201929`, and `1.27905803`. Thus a fixed physical coefficient error does not create equal normalized error across channels. In particular, normalization alone makes an equal physical rear-Cl error much smaller than an equal physical front-Cd error. Rear Cl also receives only `1/7` of the force-channel weight. This is a reason to measure its actual gradient contribution, not proof that it is the unique failure cause.

The existing paired-statistic objective is different. It compares nine physical statistics: mean total Cd, mean rear Cl, and rear-Cl fluctuation RMS at H20/H50/H100, then scales those errors with train-only force standard deviations. The interleaved lambda10 run optimizes `L_regular + 10 L_paired_stat`. Its final train-only paired loss near `0.0035` is not directly comparable to either `L_regular` or the proposed per-step objective because their dimensions, reductions, and normalization differ.

The approved technical probe evaluated one train-only dynamic pair, `b00:multisine`, with the proposed true-state per-step action-minus-zero force objective. At H100 it measured:

- normalized loss `0.0617745727` before any lambda multiplier;
- full-parameter gradient L2 norm `0.291342935`, all finite, with 47,210,800 nonzero values out of 47,222,711;
- per-channel normalized MSE `0.00391298 / 0.00062864 / 0.10467867 / 0.00916568` for front Cd / front Cl / rear Cd / rear Cl;
- no optimizer, no parameter update, no model save, and no validation or frozen access.

The rear-Cd term dominates this one-pair scalar diagnostic. Multiplying the new loss by 10 would give a scalar contribution near `0.618` for this pair, but this does **not** establish equivalence to lambda10 on the old statistic loss and does not establish that rear-Cl gradients will dominate, agree with, or survive the regular objective. The probe result is a numerical/technical observation, not a scientific admission result.

Evidence: `artifacts/true_state_paired_force_backward_probe_v2_20261005/results/result.json` (SHA-256 `773af0496e4a4eba50ac4f50728805ac25db8c0e6e86e6ac1e4fb1b970edf9c8`) and its completion receipt (SHA-256 `eb23c6619d589798b2aa7f629601c52b1c86d50269626a011e087a2b721512d4`). The inspected interleaved resolved config has SHA-256 `68af1e8dbaec7c63f2470bc5a6650975682ec05717833f77f9649c0ed6e47a49`.

## What remains unknown

No current training history records component-wise gradients for the regular field term, weighted regular force term, old paired-statistic term, or new true-state step-force term. It therefore remains unknown whether either paired term is large enough to affect rear-Cl predictions, is overwhelmed by the regular objective, or points in a conflicting parameter direction. Scalar losses cannot answer those questions. FC-P003B's independent post-evaluation must also complete before choosing a new intervention; a failure could reflect action/phase transfer, an H1 force mapping error, rollout error, or a combination.

Using the same numerical coefficient `10` for old and new losses would preserve a coefficient, not an effective gradient strength. A future experiment must be described as changing the supervision form, not as an equal-strength replacement unless a separate measurement supports that claim. No coefficient should be selected from validation performance.

## Minimal train-only mixed-gradient diagnostic, conditional on Lead approval

Use the immutable Main-e2 parent and the already approved training seed/order. Read only the 16 actual interleaved regular H100 batches and the dynamic8 pairs twice, matching the intended training insertion contract. Do not construct an optimizer, step parameters, save a candidate, or open validation/frozen data. Accumulate mean gradients separately, clearing gradients between components:

1. regular field loss, `g_field`;
2. regular force loss after the actual `0.2` multiplier, `g_regular_force`;
3. the full regular objective, `g_regular`;
4. the old statistic objective after multiplier 10, `g_old_pair`;
5. the true-state per-step objective after multiplier 10, `g_step_pair`, with exact `chunk_steps / 100` backward scaling.

The diagnostic should record:

- each normalized scalar loss and its multiplier-weighted contribution;
- at every one of the 16 actual insertion positions, global gradient L2/L-infinity norms, finite/nonzero counts, `pair / regular` and `pair / regular_force` norm ratios, cosine similarities, sign-conflict fraction, and the unchanged clip threshold's implied scale; report min/median/max as well as the individual values;
- the separately accumulated mean gradients and their cosine/norm diagnostics, explicitly labelled as aggregate diagnostics rather than a replacement for the per-position update evidence;
- norm of the explicitly summed mixed gradient and numerical agreement with the separately accumulated components;
- four per-channel loss and gradient norms, especially rear Cl; final output-projection row norms may be reported only after the model parameter layout is explicitly verified;
- peak accelerator memory and minimum unified `MemAvailable` under the existing 20 GiB guard.

Averaging parameter gradients can hide cancellation between positions and cannot reproduce the actual sequence of clipping and optimizer steps. It therefore cannot establish by itself that a paired signal is "overwhelmed." A one-regular-batch plus one-pair run is acceptable only as a resource/smoke check and must not support a scientific gradient-balance claim. The proposed 16-position diagnostic completely covers the paired-update positions in the current training contract; it is not claimed to be mathematically minimal. Results must be interpreted together with the eventual FC-P003B failure map; they must not by themselves authorize a lambda change, training run, PPO, or a relaxed surrogate gate.
