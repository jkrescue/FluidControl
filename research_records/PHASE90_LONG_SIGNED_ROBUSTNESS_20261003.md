# Two-phase long-window signed-rotation audit

Date: 2026-10-03

## Scope

This audit tests whether the long-window response to constant rear-cylinder
rotation is robust to the initial shedding phase. It is a real OpenFOAM v2512
physical-response study, not controller tuning, a closed-loop result, a Gate-B
replacement, or final physical acceptance.

Phase A is the existing matched `t=80` panel, evaluated over `t=92..116`.
Phase B is a new matched `t=90` replay, run to `t=126` and evaluated over the
predeclared `t=102..126` window. Both windows discard 12 D/U after action onset
and retain 24 D/U, about 3.9 uncontrolled shedding periods. In each phase,
zero, +1, and -1 schedules ramp from zero over 2 D/U and then remain fixed.

## Provenance and protocol integrity

Before Phase-B CFD, the source `t=90` OpenFOAM U/p hashes were compared with
the raw `t=90` fields of all cases in the current checkpoint dataset and the
independent-phase dataset. The audit covered 39 cases: 28 train, 5 validation,
and 6 frozen test. No case had an exact U+p match.

- source U SHA-256: `3684a13bc9a1a81f9f538bce4f271f25e18b0c4d4318408a30c35509ae8d6b76`
- source p SHA-256: `7db599790751368f3239d6ab0fd5280a718ccfd7b376e21453dc9f0604b05cd5`
- predeclared audit SHA-256: `976349e83531b3229c975ef92436fcc920eb84fcae9fe6587c7c81e04b7cd9af`

The original predeclared artifact mistakenly named the Phase-A reference with
the `landscape_long_val_*` prefix while assigning it the `t=92..116` window of
the intended 36-D/U `landscape_val_*` panel. This was found while all Phase-B
solvers were still running and before any Phase-B force result was analyzed.
The original file is retained unchanged. A separate pre-result correction
records the intended `landscape_val_{zero,p100,m100}` references; no Phase-B
action, window, threshold, or CFD result changed. Correction SHA-256:
`4f214c794c03eb3ba3d31b86864492fa76fb9f5ecab89d70788577aa43fdba52`.

## Numerical quality

All three new cases completed 7,200 steps and ended cleanly. Maximum Courant
number was below 0.262 and maximum absolute global continuity error per step
was below `1.5e-12`. Each fixed analysis window contains 4,801 paired
front/rear force samples. The jobs used at most 12 CPU and 24 GiB in aggregate,
did not use the GPU, and left about 98 GiB MemAvailable.

## Drag and lift results

| Phase | Action | Mean total Cd | Drag reduction vs zero | Rear Cl' RMS ratio | `|mean rear Cl| / zero Cl' RMS` | Canonical joint |
| :-- | :-- | --: | --: | --: | --: | :-- |
| t80 | +1 | 2.219536 | 3.617% | 0.9682 | 0.8113 | fail |
| t80 | -1 | 2.208797 | 4.083% | 0.9691 | 0.8650 | fail |
| t90 | +1 | 2.208649 | 3.833% | 0.9708 | 0.8281 | fail |
| t90 | -1 | 2.206371 | 3.932% | 0.9714 | 0.8378 | fail |

The total-drag effect is robust: the between-phase change in reduction is only
0.217 percentage points for +1 and 0.150 points for -1. The long-window signs
are also nearly symmetric. This shows that the much larger `-1` advantage in
the 20-step `t=86` panel was a local phase/transient ranking, not a durable
preference for one rotation sign.

Both actions reduce rear fluctuating lift slightly, but both fail the canonical
mean-lift constraint by roughly a factor of eight. Their rear mean Cl values
are about -0.96/-0.99 for +1 and +1.02/+1.00 for -1. Thus this is a robust
drag-versus-lateral-load tradeoff, not simultaneous drag and lift control.

The four 6-D/U block means remain mildly directional: for example, Phase-B +1
declines from 2.2232 to 2.1989 while -1 rises from 2.2040 to 2.2101. The full
24-D/U averages are cross-phase stable, but these block trends should remain
visible rather than being hidden behind a single mean.

These two starts come from one deterministic uncontrolled baseline and use the
same development mesh, timestep and solver configuration. They support
start-phase repeatability on this setup, not a mesh/time-converged estimate,
statistical independence, or a final multi-phase control claim.

## Rear-cylinder torque-work proxy

OpenFOAM defines

`CmPitch = M_fluid / (0.5 rho U_inf^2 A_ref l_ref)`.

With the force-on-body sign convention, the reported signed ideal actuator
proxy is `-omega*CmPitch/Cd_total_zero`. It is not measured shaft or electrical
motor power.

| Phase | Action | Signed proxy / zero drag power | Positive-only proxy | Absolute-work proxy |
| :-- | :-- | --: | --: | --: |
| t80 | +1 | -4.187% | 0.000% | 4.187% |
| t80 | -1 | -4.185% | 0.000% | 4.185% |
| t90 | +1 | -4.210% | 0.000% | 4.210% |
| t90 | -1 | -4.183% | 0.000% | 4.183% |

The sign indicates that the fluid does net work on the prescribed rotation
under this convention, so an ideal speed-holding actuator would oppose or
extract that work. Positive-only demand is therefore zero in these stationary
windows. This does **not** establish regenerative electrical output: bearing,
generator, drive, and transient losses are absent, and the sign convention has
not been validated against a physical motor model. The absolute interaction is
about 4.2% of baseline drag power, the same order as the drag reduction, so it
must be retained in any later energy accounting.

## Conclusion

Constant signed rotation produces a reproducible 3.6–4.1% total-drag decrease
across two long windows, but it is not an acceptable controller because it
creates a large mean lateral load. The experiment also resolves the short-window
ambiguity: local FNO ranking can select the better sign over 20 steps, but the
long-run signed responses are almost symmetric. A surrogate must therefore not
turn that local ranking into a global sign preference.

The next control family must be sign-balanced or phase-balanced and explicitly
penalize mean rear lift while retaining torque-work accounting. Existing
zero-mean periodic and proportional/lag-feedback trials have not passed the
joint gate, so this result does not justify another blind parameter sweep.

Machine-readable outputs:

- `artifacts/tandem_cylinders/phase90_long_signed_robustness_20261003/result.json`
- `artifacts/tandem_cylinders/phase90_long_signed_robustness_20261003/rear_cylinder_torque_timeseries.csv`

Result SHA-256:
`8c2e70274ec918df2521cce9a40af988c574315481c71ebbb5270a69865770c2`.
