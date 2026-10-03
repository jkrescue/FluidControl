# Matched rotation-onset replication (pre-solve plan)

The earlier constant-rotation pilot reported rear fluctuating Cl RMS of about
1.517 at `omega=±1`, whereas the new matched-start, two-time-unit ramp panel
reported about 1.143 at the same final rotation. Their onset and averaging
windows differ. This difference is too large to wave away or to use either
result as a definitive lift-control claim.

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
