# Proposed H1–H5 frozen confirmation — protocol only

Status: PREPARATION_ONLY; no frozen HDF field read, model inference, new CFD, or seal opening authorized. Freeze the controller candidate and this protocol before a separate Lead authorization opens final-test payloads. Do not tune or select from its results. Existing K1 H100 development FAIL remains unchanged.

## Existing split evidence and scope

`data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json` declares 20 train / 10 validation / 10 frozen-test trajectories, 801 frames each, grid256×128. Train phases are b00/b02/b04/b06; repeatedly used validation10 phases b01/b05 and dynamic6 b01/b05 are not fresh confirmation data.

Frozen split `splits/frozen_test.json` SHA `1d13ee46cf697962169c98ab35d7bb8e310b18f9e266c63d084c2e57e495728a` lists all ten cases: prefix `matched_start_acquisition_frozen_test_`, phase `b03` and `b07`, each suffix `m0375`, `m075`, `p0375`, `p075`, `zero`. Each HDF hash is already recorded in that manifest. The sibling `frozen_test_seal.json` says `FROZEN_TEST_SEALED_NO_MODEL_ACCESS`; metadata does not by itself prove that nobody historically accessed payloads. Before opening, audit existing execution receipts for accidental exposure. The development-only `...full40_dev30_v1` intentionally has no frozen payloads; do not mistake that for missing original data or silently repurpose validation.

Existing curated frozen trajectories are the first-choice data: no new CFD is required for this fixed-action, unseen-phase short-time test according to the completed full40 manifest. Availability/hash verification is deferred until authorized opening. These trajectories do not cover arbitrary policy-generated action histories; do not claim such coverage. New policy-conditioned CFD data would only be needed for that separate question, not this confirmation.

## Frozen model and exact evaluation

Use the existing official PhysicsNeMo K1 dual checkpoint, manifest SHA `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`. Bind the actual candidate files from its reviewed manifest and CPU-reload receipt; no checkpoint selection or retraining. Train-only normalization SHA `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` is unchanged. Use the already measured highest/no-TF32 post-load inference override, explicitly distinct from original H100 precision provenance; record effective flags.

Reuse `scripts/evaluate_tandem_fno.py`, `fluid_control.dual_fno.load_dual_fno`, existing K1 history inference, and official HDF5Reader/DataPipe. CLI currently accepts `test`, not `frozen_test`: execution preparation must explicitly resolve the sealed split through a reviewed read-only configuration/path mapping; never rename or mutate the sealed dataset. No new NVIDIA API.

All ten cases, all five endpoint horizons H=1,2,3,4,5, identical starts `{0,25,...,775}` (32 per case). At each start initialize from the actual current frame, then free autoregression to H. Use stored physical omega_j and omega_(j+1), normalized exactly by the existing FP32 formula; predicted q_j plus these actions yields force target F_(j+1). No future truth in free-AR. H1 is the true-current-state case; do not pool it with later AR as if identically conditioned. Keep per-case/per-start/per-lead rows and persistence baselines; no favorable-window selection.

Fields: same valid-fluid mask, inverse train normalization, u/v and per-frame ROI-mean-removed pressure. Use existing `field_error_sums` / `relative_field_metrics`: pool physical SSE and physical target-energy sums before square root; report u/v/p and joint velocity L2. Zero denominator yields null, never epsilon substitution. Report physical per-channel RMSE alongside ratios. Pressure is gauge/ROI-centered solver pressure, not absolute pressure. Report all four force MAEs separately and total-Cd MAE; preserve macro-case summaries separately from pooled statistics.

Action response: at common start0 only, compare each four nonzero-action branch against same-phase zero for each H1–H5 endpoint. Report delta totalCd and all four delta-force errors, signs and full ordering, with exact same stored action conditioning. Later starts have divergent physical states and must not be called causal action-minus-zero interventions. Absolute trajectory-error metrics still use every fixed start.

## Cost and decision interpretation

The unchanged evaluator independently reruns each horizon: 10×32×(1+2+3+4+5)=4800 flow transitions and 4800 aerodynamic evaluations, no backward/optimizer/CFD. This is substantially smaller than full H100 formal evaluation, but wall time is not asserted without measurement. Proposed separate approval cap: 12GiB/no-swap, existing bounded GPU allocator, physical MemAvailable50GiB startup/22GiB runtime, 1200-second ceiling; stop on resource/numerical failure without retry or changing the set. No CUDA-cache-free admission threshold on UMA.

No invented accuracy pass threshold. Report all predeclared errors and uncertainty across the ten cases; assess against the user's declared short-horizon accuracy requirement before execution if a categorical claim is required. This confirms only fixed-action unseen-phase H1–H5 prediction, not H100 acceptance or physical control robustness. A second matched physical initial state is a separate minimal control confirmation if the current projected800 succeeds.
