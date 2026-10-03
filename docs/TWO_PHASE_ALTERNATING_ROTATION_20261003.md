# Two-phase alternating-rotation real-CFD screen

Date: 2026-10-03

## Question and frozen design

Constant `omega=+1` and `omega=-1` reduced long-window total drag by about 4%
at both t80 and t90, but induced opposite rear-cylinder mean lift near `-1`
and `+1`. This experiment asks whether equal positive and negative dwell can
cancel mean lift while retaining that drag reduction.

The schedule was fixed before CFD and was not tuned from the result:

- amplitude `|omega|=1`;
- action period 20 D/U;
- about 8 D/U at each signed plateau, with 2 D/U linear sign changes;
- `|domega/dt|<=1` and exactly zero time-mean action in every period;
- starts at t90 and t94, each with a fresh matched zero replay;
- 80 D/U simulation, with the first 20 D/U discarded;
- fixed 60-D/U analysis window containing three complete action periods and
  about 9.75 uncontrolled shedding periods.

All cases are validation-only physical screens. They are not FNO training,
frozen-test data, policy tuning, or learned closed-loop results. The unchanged
canonical checks are at least 2% lower total drag, rear Cl' RMS ratio at most
1.05, and `|mean rear Cl| <= 0.1` times zero-case rear Cl' RMS.

## Provenance and numerical quality

The pre-run leakage audit compared t90 and t94 source U/p hashes against all
current-v4 and independent-phase train, validation, and frozen-test cases: 82
comparisons covering 60 train, 10 validation, and 12 test records. Exact U+p
matches were zero.

All four OpenFOAM v2512 cases ended cleanly after 16,000 steps. Maximum Courant
number was below 0.260 and maximum absolute global continuity error per step
was below `1.9e-12`. Each analysis window contains 12,001 paired force samples.
The run used four CPU-only containers, each capped at 4 CPU and 8 GiB, and did
not interrupt GPU training.

## Fixed-window physical result

| Start | Case | Mean total Cd | Rear mean Cl | Rear Cl' RMS |
| :-- | :-- | --: | --: | --: |
| t90 | zero | 2.302561 | 0.046676 | 1.168375 |
| t90 | alternating | 2.303147 | 0.048521 | 1.513579 |
| t94 | zero | 2.301710 | -0.035165 | 1.175918 |
| t94 | alternating | 2.353775 | -0.044825 | 1.376193 |

| Start | Total-drag reduction | Rear Cl' RMS ratio | Mean-lift ratio | Canonical joint |
| :-- | --: | --: | --: | :-- |
| t90 | -0.025% | 1.2955 | 0.0415 | fail |
| t94 | -2.262% | 1.1703 | 0.0381 | fail |

A negative reduction means increased drag. The action succeeds at its narrow
purpose of cancelling mean lift: the mean-lift check passes at both phases.
It does not preserve the constant-rotation drag benefit and instead amplifies
rear lift fluctuations by 29.5% and 17.0%. The drag and fluctuation checks fail
at both starts, so the unchanged joint criterion fails unequivocally.

## Action-cycle variability

The controlled total-Cd means for the three fixed 20-D/U cycles are:

- t90: 2.4521, 2.3069, 2.1505;
- t94: 2.2211, 2.4185, 2.4217.

Rear Cl' RMS is also high in every controlled cycle. The strong opposing cycle
trends show a beat/phase interaction between the 20-D/U action and the natural
wake. No favorable subwindow was selected after the fact; the predeclared
60-D/U averages remain the reported result. The 2.24-percentage-point
between-phase drag-response difference further rules out a robust benefit.

## Torque-work proxy

The action has zero mean and `omega_rms=0.9309`. Using OpenFOAM's fluid-on-body
CmPitch convention, the ideal signed actuator proxy is
`-omega*CmPitch/Cd_total_zero`.

| Start | Signed proxy | Positive-only proxy | Absolute-work proxy |
| :-- | --: | --: | --: |
| t90 | -3.935% | 0.0666% | 4.068% |
| t94 | -3.980% | 0.0500% | 4.080% |

The negative signed value means the fluid supplies net work to the prescribed
rotation under this convention. This is not measured electrical power or a
regeneration claim: drivetrain, bearing, generator, and transient losses are
absent. The roughly 4.1% absolute interaction is retained because it is
material relative to the aerodynamic changes.

## Decision

This predeclared alternating strategy is rejected. Long equal dwell cancels
the mean lateral load, but the sign changes destroy the robust constant-action
drag benefit and amplify fluctuating lift. Because both independent starts
fail the same two physical checks, no gain, period, phase, or analysis-window
retuning is justified from this experiment.

The result should remain a validation-only negative-control dataset and must
not be added to FNO training during model selection. Its scientific value is
mechanistic: mean-lift cancellation alone is insufficient, and future control
must explicitly manage sign-transition wake dynamics, drag, lift fluctuation,
and torque cost together.

Artifacts:

- `artifacts/tandem_cylinders/two_phase_alternating_predeclared_20261003.json`
- `artifacts/tandem_cylinders/two_phase_alternating_result_20261003/result.json`
- `artifacts/tandem_cylinders/two_phase_alternating_result_20261003/rear_cylinder_torque_timeseries.csv`

SHA-256:

- predeclared audit: `34355adfe55a019c8615514c1634531c76bcc230940b9c0e80319746374e627b`
- result: `31bb77695b785b1fc2ad6ea05e6ced31dcc75b2e34ad5030e7e0f603b8d070c5`
- torque CSV: `e95931b494ac353f85c6564882d2be38720b65c43f994acab233face678583c9`
