# Full40 PhysicsNeMo FNO retraining plan (review gate)

## Decision

Do not start training until the finalized matched-start profile passes the new
read-only preflight. The current v4 candidate has validation-only H100 terminal
total-drag pooled NRMSE `15.371%`, worse than its v3 parent `15.047%`; both miss
the fixed `10%` Gate-B threshold. Neither result supports a PPO control-benefit
claim.

The full40 data isolate entire phase bins: train uses b00/b02/b04/b06 (20
trajectories), validation uses b01/b05 (10), and frozen test uses b03/b07 (10).
These are eight phases of one baseline limit cycle, not independent Reynolds
numbers, geometries, or noisy operating conditions.

## Blocking schema correction before finalization

The current full40 finalizer computes normalization from the explicit 20-case
train manifest, which is correct, but its planned JSON omits two fields required
by the existing DataPipe:

- `manifest.json.max_abs_omega = 0.75`; without it the DataPipe silently uses
  `1.0`, giving the action channels the wrong scale.
- full normalization compatibility fields: four-force channels/mean/std,
  rear-only `force_channels/force_mean/force_std` aliases, and
  `state_abs_normalized_channel_max_train` plus
  `state_abs_normalized_max_train`. Missing aliases break existing DataPipe,
  HydroGym, CEM, or state-divergence guards.

The finalizer must be corrected and retested before it is authorized. The new
training preflight deliberately fails closed if either field is absent. It
reads train and validation manifests, but only the frozen seal metadata; it
never opens frozen HDF5 or uses frozen outcomes for selection.

## Reproducible training stages

No new network or API is introduced. The entry points remain the project’s
existing official PhysicsNeMo 2.2.2 `FNO`, `HDF5Reader`, `DatasetBase`, and
`DataLoader` path in the pinned image ID `sha256:b40d5888...`.

1. One-step initialization: existing 5-layer, width-48, modes-32 FNO; 30
   epochs from random initialization on train20. Reusing the old checkpoint
   directly is avoided because it was optimized under a different
   normalization/action scale.
2. H20 rollout: existing rear-drag-weighted loss, 10 epochs, batch 4, all ten
   epoch checkpoints retained. This is the prior validated loss ablation, not
   a new architecture. Modes-48 remains excluded until its resource benchmark
   and a validation-only ablation justify it.
3. Every run uses the existing 20-GiB memory guard, GPU0, pinned container,
   and `--network none`. Exact historical evidence exists for one-step batch16
   only at allocator 0.20 (one completed smoke epoch), not 0.15; therefore the
   one-step config honestly uses 0.20. H20 batch4 completed five full epochs at
   allocator 0.15 and retains that cap.

The preflight recomputes SHA-256 for every train and validation HDF against its
split manifest. It does not open, hash, or enumerate frozen HDF files.

Historical wall time gives a planning range, not a guarantee: comparable
one-step 30 epochs took about 59 minutes; v4 H20 used about 65 minutes for five
epochs on 28 train trajectories. Scaling the latter to 20 trajectories and ten
epochs gives roughly 93 minutes. Including validation, budget 2.8–3.2 hours
after the complete dataset is ready.

## Shortest rigorous validation

Only full40 validation is used while choosing a checkpoint:

- evaluate H1/H10/H50/H100 with observed actions and physical-unit forces;
- primary gate: pooled H100 endpoint `Cd_front + Cd_rear` NRMSE at most 10%;
- always report macro and worst-case H100 NRMSE and persistence baselines;
- report front/rear Cl MAE and front/rear Cd MAE;
- at start=0 within each validation phase, compare each of four nonzero actions
  with the same-phase zero branch: eight strict differences total. Require
  delta-Cd MAE <=0.023 (half of the approximate 2% signal scale 0.046) and all
  8/8 non-tied signs correct. Also require exact ordering agreement for every
  non-tied pair among the five actions at each phase (up to 20 dependent
  pairwise comparisons);
- starts after zero are action-diverged states and may only be labelled
  matched-elapsed-time diagnostics, never counterfactual rankings.

Passing 10% is necessary but not sufficient because it is roughly 0.23 Cd,
larger than the approximate 0.046 Cd control signal. The joint terminal
readiness gate also requires the 0.023 action-difference bound, 8/8 signs, and
all non-tied cross-action orderings. These comparisons share only two phase
starts, so neither the eight zero-relative differences nor the pairwise
orderings are statistically independent or able to establish broad
significance. Even a joint pass permits
only the next validation/window-lift study, not a PPO or physical-benefit
claim. Front/rear Cl endpoint MAE is reported without inventing a threshold;
Cl-prime <=1.05 and mean-Cl bias <=0.10 require a time-window fidelity audit
and ultimately real CFD. Frozen-test evaluation remains a later, one-time
authorization after model and thresholds are locked.

Dry-run review commands:

```bash
python3 scripts/plan_full40_fno_retrain.py
bash scripts/run_full40_fno_retrain_spark.sh --dry-run
```

No training is started by either command.
