# Fixed nonzero action, real OpenFOAM diagnostic (2026-10-02)

This follows the [segmented zero-action smoke](SEGMENTED_OPENFOAM_ZERO_SMOKE_20261002.md). It is a bounded, hand-specified **diagnostic** sequence, not a learned policy or closed-loop control. The action is applied to the rear-cylinder `rotatingWallVelocity` boundary as a linear table from the previous interval endpoint to the next; the CFD is restarted at each 0.1-time-unit interval and the next raw 67-channel observation is read after solving. The first interval is a zero-action warm-up.

## Reproduce on DGX Spark

```bash
python3 cfd/tandem_cylinders/make_probe_feedback_case.py probe_feedback_ramp_smoke_20261002 --steps 3
python3 scripts/run_tandem_probe_feedback_schedule.py probe_feedback_ramp_smoke_20261002 \
  --actions 0,0.5,0 \
  --output artifacts/tandem_cylinders/probe_feedback_ramp_smoke_20261002
python3 scripts/validate_tandem_action_schedule.py \
  --cases-root cfd/tandem_cylinders/cases \
  --action-result artifacts/tandem_cylinders/probe_feedback_ramp_smoke_20261002/result.json \
  --zero-result artifacts/tandem_cylinders/probe_feedback_zero_smoke_20261002/result.json \
  --output docs/results/tandem_nonzero_action_diagnostic_20261002.json
```

These exact names already exist on the original DGX Spark run; use fresh suffixes to rerun. The scripts refuse to overwrite existing cases/results. The setup is the same Re=100 tandem-cylinder, 19,290-cell, t=80 OpenFOAM v2512 restart and digest-pinned container as the zero smoke. The action sequence obeys `|omega|<=5` and interval-to-interval `|delta omega|<=0.5`; each segment has 20 CFD time steps at `deltaT=0.005`.

## Result and interpretation

All three segments ended cleanly and kept maximum Courant number below 0.243 and global continuity per step below 4.4e-13. The first interval matched the zero-action reference exactly. At t=80.2 the nonzero run differed from zero by 1.22e-5 in maximum absolute wake-probe value and -0.01325 in rear-cylinder Cl; at t=80.3, after returning omega to zero, the differences were 4.43e-5 and -0.02149. The independent audit is [the JSON result](results/tandem_nonzero_action_diagnostic_20261002.json).

This verifies that bounded per-interval boundary changes reach the real CFD solver and affect measured observations. It does **not** establish reduced time-averaged drag/lift, a useful action schedule, a policy trained in HydroGym, or robust closed-loop performance. The next gate is to freeze a model-backed policy only after independent FNO test and ablation checks, then compare full real-CFD feedback trajectories to matched baselines over a meaningful horizon; report medium-mesh numerical sensitivity separately.
