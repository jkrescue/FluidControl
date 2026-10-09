# Stage-1 PPO objective and PhysicsNeMo surrogate assessment — 2026-10-02

## Decision

Stage 1 should target **total drag reduction of the two-cylinder system**, with downstream lift used only as an anti-degeneracy regularizer and with explicit actuation/rate penalties. It should not be presented as vortex-induced-vibration control because the current cylinders are fixed and no structural dynamics are solved.

Do not continue PPO training with the present reward. The present surrogate predicts only rear-cylinder `Cd/Cl`, the 32-step episode spans only about half of one uncontrolled shedding period, and the 50-step force rollout error is too large for a credible long-window drag objective.

## Relation to Hui Tang / PolyU work

The closest PolyU directions have two different objectives:

1. Ren, Rabault and Tang (Physics of Fluids, 2021) pursued drag reduction while preventing a degenerate lift-biased solution. Their reward was the negative actuation-period average of drag plus a weighted absolute lift term, `r = -<Cd> - w<|Cl|>`. They explicitly reported that a small lift weight obtained slightly more drag reduction but generated biased forcing and non-zero mean lift; increasing the lift weight mitigated that failure mode. The paper also trained over many shedding periods rather than a fraction of one period. DOI: <https://doi.org/10.1063/5.0037371>.
2. Zhao, Zhou, Ren, Tang and Wang (Ocean Engineering, 2024) studied a downstream cylinder in an upstream wake and used self-rotation primarily to suppress downstream lift fluctuation, reporting 98% reduction at the most difficult tandem distance. That is the closest geometric/control mechanism to this project, but its scientific target is lift mitigation rather than system drag. DOI: <https://doi.org/10.1016/j.oceaneng.2024.118138>.

Because the present project does not yet model structural motion and the requested first-stage goal is aerodynamic/hydrodynamic efficiency, total drag is the clearer primary quantity. A lift term remains necessary to stop PPO from exploiting a large mean transverse-force bias, consistent with the PolyU observation.

## Recommended reward

Use running averages over at least one shedding period:

```text
J_t = mean_W(Cd_front + Cd_rear) / Cd_total_0
    + 0.20 * mean_W(abs(Cl_rear)) / mean_abs_Cl_rear_0
    + 0.02 * (omega / 5)^2
    + 0.01 * (delta_omega / 0.5)^2

reward_t = -delta_t * J_t
```

Baseline values measured from the matched uncontrolled OpenFOAM case over `t=80..160` are:

| Quantity | Value |
|---|---:|
| front mean Cd | 1.390526 |
| rear mean Cd | 0.909244 |
| total mean Cd, `Cd_total_0` | 2.299770 |
| rear mean absolute Cl, `mean_abs_Cl_rear_0` | 1.069065 |
| rear Cl RMS | 1.179636 |

The coefficient `0.20` makes lift a guard rather than the primary objective. Final acceptance should additionally require no material increase in rear `Cl RMS` or mean lift bias, regardless of aggregate reward. The initial acceptance thresholds should be:

- real-CFD total mean drag reduction at least 2% against the phase-matched zero-action baseline;
- rear `Cl RMS` increase no more than 5%;
- absolute rear mean `Cl` no more than 10% of the uncontrolled rear `Cl RMS`;
- report mean `omega^2`, mean `delta_omega^2`, maximum action and maximum action rate separately.

These thresholds are engineering gates for this project, not values claimed by the cited papers.

## Why total drag, not rear drag alone

The 24/4/4 real-CFD splits show that rear drag is an excellent *variation proxy* for total drag: their correlation is 0.99989/0.99989/0.99990 on train/validation/test. Front-cylinder Cd varies much less than rear Cd (`std=0.0173` versus `0.9545` on train). However, the front cylinder contributes about 54% of mean total drag. A small control-induced change in its mean can therefore decide whether a marginal apparent rear-drag benefit is a true system benefit. Rear-only drag is acceptable for a software smoke test, not for the Stage-1 scientific objective.

## Existing CFD data coverage

The curated independent-v2 data are genuine OpenFOAM trajectories at fixed `Re=100`, `L/D=5`:

- 24 train, 4 validation and 4 test trajectories;
- 801 frames and 800 transitions per trajectory;
- 25,632 total field snapshots, of which 19,224 are in train;
- 256×128 sampled grid over `x=[8,25]`, `y=[4,11]`;
- open-loop schedules include random ramps, multisines, chirps and edge holds;
- train actions span the complete declared interval `omega in [-5,5]` and every one of ten unit-width action bins is populated;
- train action quantiles are `[-5, -4.116, -2.056, 0, 2.000, 4.340, 5]` at `[0,5,25,50,75,95,100]%`;
- sampled train `delta omega` spans `[-0.667,0.667]`, covering the PPO limit `|delta omega|<=0.5` in most of its operational domain.

This is good one-dimensional actuator coverage for the fixed geometry/Re. It is not a complete control dataset:

- every trajectory restarts from the same physical `t=80` field;
- the schedules are precomputed open loop, not sampled from PPO-visited closed-loop states;
- validation/test gain independent action histories but not independent initial physical phases at frame 0;
- there is no variation in Reynolds number, spacing, geometry or incoming disturbance;
- high-amplitude/high-rate bins have substantially larger rollout errors;
- edge-hold validation/test trajectories are mildly out of distribution in whole-trajectory action RMS.

No new CFD is required before the first total-drag surrogate retraining because all HDF5 files already contain `front_cd_cl` and `rear_cd_cl`. New CFD becomes necessary after a candidate policy is available: add several phase-shifted uncontrolled restarts and targeted trajectories around the policy's visited state/action distribution.

The matched constant-`omega=+1` coarse/medium OpenFOAM pair has also completed. Over `t=80..160`, coarse-to-medium differences are 0.50%/0.16% for front/rear mean Cd, 2.46%/1.06% for front/rear Cl RMS, and below 0.8% for both shedding-frequency estimates. This supports use of the coarse data for a first-stage method experiment, but only for this constant action; the eventual learned closed-loop policy still requires its own medium-grid replay.

## Current PhysicsNeMo FNO capability

The official PhysicsNeMo FNO receives six input channels: normalized `u/v/p`, fluid mask, current `omega/5`, and next `omega/5`. It therefore represents a continuous action-conditioned transition within the observed `[-5,5]` action domain; it is not a separate model per speed. Counterfactual tests confirm that the network uses the action inputs: replacing observed actions with zero increases held-out force MAE from 0.0706 to 0.2214 at one step and from 0.5921 to 1.5186 at 50 steps.

The present output has only five channels: next-state increments `u/v/p` and **rear** `Cd/Cl`. It cannot optimize true total drag. Extend the official FNO output to seven channels by adding front `Cd/Cl`; keep the same PhysicsNeMo architecture and APIs.

### Accuracy assessment

| Horizon | Field MAE | Rear-force MAE | Interpretation |
|---:|---:|---:|---|
| 1 step / 0.1 time | 0.00246 | 0.0706 | adequate short-step surrogate screen |
| 10 steps / 1.0 time | 0.02156 | 0.1177 | useful for short candidate ranking |
| 50 steps / 5.0 time | 0.07285 | 0.5921 | too inaccurate for a drag-control claim |

The uncontrolled rear-lift dominant period measured from the real baseline is 6.154 time units (`St≈0.1625`). The current PPO episode is 32×0.1=3.2 time units, only 0.52 shedding periods. Even the 50-step audit is only 0.81 periods. Mean drag must not be optimized or reported from such a short window.

Training ran for 20 epochs although the configuration declares 80. Validation state MAE flattened from 0.002281 at epoch 15 to 0.002223 at epoch 20, and normalized force MAE flattened around 0.033. This indicates near-convergence of the one-step objective; simply adding epochs is unlikely to fix accumulated force error. The next model needs front-force targets plus multi-step/rollout-aware validation and checkpoint selection, not merely longer one-step training.

## Required next sequence

1. Change FNO force targets from rear `[Cd,Cl]` to `[front Cd, front Cl, rear Cd, rear Cl]`; retrain from scratch with the official PhysicsNeMo FNO.
2. Validate front, rear and total drag at 1/10/50/100 steps, stratified by action magnitude and rate.
3. Require at least one full shedding period for PPO returns; prefer 2–3 periods if rollout accuracy permits. If the FNO cannot maintain acceptable force accuracy that long, use shorter model-predictive segments with periodic real-CFD correction rather than pretending a long surrogate rollout is accurate.
4. Train PPO with the normalized total-drag reward above and select checkpoints on both phase-specific and independent-history panels.
5. Run frozen-policy OpenFOAM feedback against phase-matched zero action for at least 5–10 shedding periods and report total drag, both cylinders' forces, action cost and grid sensitivity.
6. Add policy-guided CFD trajectories and independent restart phases, then retrain the surrogate if the policy visits poorly covered regions.
