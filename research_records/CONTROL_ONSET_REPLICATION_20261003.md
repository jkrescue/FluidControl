# Matched rotation-onset replication (pre-solve plan)

The earlier constant-rotation pilot *labelled* rear Cl RMS of about 1.517 at
`omega=±1`, whereas the new matched-start, two-time-unit ramp panel reported
about 1.143 for **fluctuating** Cl RMS. At the time of predeclaration, this
looked like a possible onset/window effect; the two metrics were later found
to be different. The experiment remains a useful controlled check and is
retained with its original motivation documented transparently.

Before solving, this validation-only experiment freezes two additional real
OpenFOAM cases from the **same uncontrolled `t=80` restart** as the ramp panel:
instant `omega=+1` and `omega=-1` at the first restarted solver step. All
other settings match the long ramp cases: Re=100, L/D=5, pinned OpenFOAM
v2512 `pimpleFoam`, 19,290-cell mesh, `Δt=0.005`, end time `t=160`, output
every 0.1, and primary analysis window `t=120..160`. The existing ramp and
zero cases form the paired references. Only action onset differs. Source
`U/p` hashes and action schedules are saved before solving.

```bash
python3 -m unittest tests.test_control_onset_replication
python3 cfd/tandem_cylinders/make_control_onset_replication.py \
  --audit artifacts/tandem_cylinders/control_onset_action_audit_20261003.json
bash cfd/tandem_cylinders/run_control_landscape_case.sh landscape_step_val_p100_20261003
bash cfd/tandem_cylinders/run_control_landscape_case.sh landscape_step_val_m100_20261003
```

The signed cases are not added to FNO training or the frozen test set. A
matched-window comparison must report rear fluctuating and total Cl RMS,
mean Cl, total/front/rear mean Cd, force-block drift and numerical health.
Even if the onset effect persists, two schedules at one phase and one mesh
would be a mechanism hypothesis, not a broad bistability or control claim.

## Result and corrected interpretation

Both instant-onset cases ended cleanly and passed the same 801-field,
16,000-force-row and solver-health checks as the ramp cases. The matching
`t=120..160` analysis found instant/ramp **fluctuating** rear Cl RMS ratios
of `1.0000204` (`omega=-1`) and `0.9999882` (`omega=+1`). Mean total Cd
changes versus ramp were `-0.0148%` and `+0.0097%`. Thus onset speed has
negligible effect in this comparison; it does **not** explain a 1.517-vs-1.143
gap. Artifact:
`artifacts/tandem_cylinders/control_onset_replication_result_20261003.json`
(SHA-256 `2d52f4202d89dc1c480342f66fbd0be718a1c87b542ac9bd182e3c7507f287f5`).

The retained original `control_small_p100` OpenFOAM force series resolves
the apparent gap: over its `t=80..160` window, mean Cl is `-1.004868`,
**fluctuating** RMS is `1.137038`, and **total** RMS including the mean is
`1.517437`. The old table's ~1.517 value was a total-RMS statistic despite
the ambiguous label; the new ~1.143 is fluctuating RMS. The different case
starts and windows cause only small additional numerical differences. This
is a metric-definition correction, **not** evidence for two flow attractors.
This conclusion is independently reproducible via
`python3 cfd/tandem_cylinders/audit_lift_rms_definitions.py --output
artifacts/tandem_cylinders/lift_rms_metric_audit_20261003.json` (output
SHA-256 `c0492df19626562bbb415e959a3d2094328e47c9c91f8e3ca577505b4a8670aa`).
