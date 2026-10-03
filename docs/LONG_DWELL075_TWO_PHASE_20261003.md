# Two-phase low-amplitude long-dwell real-CFD audit

Date: 2026-10-03
Status: completed validation-only physical experiment; **not** model training, policy tuning, learned closed-loop evidence, or final acceptance

## Question and frozen protocol

This experiment tests one physical hypothesis after two prior negative families: short same-sign `|omega|=0.75` pulses did not jointly pass, and a faster `omega=+/-1`, `T=20 D/U` square wave removed mean lift but worsened drag and lift fluctuations. A simple duty-cycle mixture was deliberately rejected because the measured component responses cannot linearly yield the required 2% drag reduction while cancelling mean lift.

The sole predeclared candidate instead reduces amplitude and halves switching frequency:

- geometry/flow: tandem fixed cylinders, `Re=100`, `L/D=5`, rear-cylinder rotation only;
- action: `omega=-0.75` and `+0.75`, period `40 D/U`, equal signed dwell;
- per-period relative knots: `(0,0), (2,-0.75), (18,-0.75), (22,+0.75), (38,+0.75), (40,0)`;
- exact time mean: zero; maximum ramp rate: `0.375 U/D`;
- independent starts: `t=90` and `t=94`, each paired with a fresh same-state zero-action run;
- run length: `120 D/U`; fixed analysis windows `[130,210]` and `[134,214]`;
- each analysis window contains exactly two complete action periods (`80 D/U`, approximately 13 natural shedding periods using `Ts=6.154 D/U`);
- if either phase fails, stop this family without changing amplitude, period, duty, phase, or analysis window.

The immutable predeclaration is `artifacts/tandem_cylinders/longdwell075_predeclared_20261003.json`. It records SHA-256 restart hashes, full action tables, fixed windows, acceptance thresholds, and 82 comparisons against current train/validation/test profiles. No exact `U+p` initial-state match was found. These cases remain outside Curator/FNO train, validation, and frozen-test profiles.

## Canonical joint gate

A phase passes only if all three conditions hold against its same-phase zero control:

1. total mean drag reduction is at least 2%;
2. rear-cylinder lift fluctuation RMS, `RMS(Cl_rear - mean(Cl_rear))`, is no more than `1.05` times zero control;
3. `|mean(Cl_rear)|` is no more than `0.10` times the zero-control rear lift fluctuation RMS.

The two phases must both pass. Total drag is always front plus rear drag; fluctuation RMS is kept distinct from total RMS.

## Solver and numerical QC

- solver: pinned OpenFOAM container and `pimpleFoam`, `deltaT=0.005 D/U`;
- force coefficients: every `0.005 D/U`; fields: every `2 D/U`;
- worker: CPU-only worker78; four isolated containers, each capped at 4 CPUs and 8 GiB;
- solver completion, finite force records, paired timestamps, sample count, and maximum Courant number are audited from raw outputs.

## Results

| Phase | Role | mean total Cd | mean rear Cd | mean rear Cl | rear Cl' RMS | samples | max Co |
|---|---|---:|---:|---:|---:|---:|---:|
| t90 | zero | 2.298607 | 0.907412 | -0.004926 | 1.180757 | 16001 | 0.24523 |
| t90 | control | 2.268955 | 0.878370 | -0.003730 | 1.357219 | 16001 | 0.24930 |
| t94 | zero | 2.300794 | 0.909482 | 0.013164 | 1.173606 | 16001 | 0.24523 |
| t94 | control | 2.266886 | 0.876161 | 0.013575 | 1.349099 | 16001 | 0.24915 |

| Phase | total drag reduction | rear Cl' RMS ratio | abs(mean rear Cl) / zero Cl' RMS | drag gate | fluctuation gate | mean-lift gate | joint |
|---|---:|---:|---:|---|---|---|---|
| t90 | 1.2900% | 1.14945 | 0.00316 | fail | fail | pass | **fail** |
| t94 | 1.4738% | 1.14953 | 0.01157 | fail | fail | pass | **fail** |

The table reports drag **reduction as a positive number**; equivalently the controlled-minus-zero relative Cd changes are `-1.2900%` and `-1.4738%`. `Cl' RMS` is the mean-removed lift fluctuation metric, while the mean-lift bias is judged separately in the following column.

All four solvers completed 24,000 steps cleanly. Maximum absolute global continuity error per step was below `1.72e-12`. The two phase results differ by only 0.184 percentage points in drag reduction and `8.4e-5` in rear-lift fluctuation ratio.

## Rotation-work proxy

For audit only, the reported signed proxy is

`P* = -omega CmPitch / Cd_total,zero`.

Positive-only and absolute variants are also retained in the JSON and raw time-series CSV. `CmPitch` is the OpenFOAM fluid moment coefficient. These quantities are fluid-torque work proxies under the documented sign convention, not electrical motor energy; positive-only assumes no regenerative recovery.

| Phase | omega RMS | signed proxy | positive-only proxy | absolute proxy |
|---|---:|---:|---:|---:|
| t90 | 0.69819 | -0.02069 | 0.000140 | 0.02097 |
| t94 | 0.69819 | -0.02066 | 0.000124 | 0.02091 |

## Interpretation

The long-dwell action successfully cancels the large mean-lift offset seen under constant signed rotation, and it produces a small, repeatable drag reduction. It nevertheless fails the frozen joint objective in both phases: drag reduction remains below 2%, while rear lift fluctuations rise by approximately 14.95%, far beyond the 5% allowance. The close agreement between phases makes a short-window phase accident an unlikely explanation.

This is a useful negative physical result. It falsifies the specific idea that simply lowering amplitude and dwelling for several shedding periods can retain signed-rotation drag benefit while removing both mean-lift bias and fluctuation cost. The absolute rotation-work proxy is approximately 2.1% of zero-control drag power, comparable to or larger than the gross drag saving; its sign must not be interpreted as electrical recovery without an actuator model. Per the predeclared stop rule, no further period, amplitude, duty, phase, or window adjustment is made within this family.

The result is not a linear interpolation of the earlier `omega=+/-1`, `T=20 D/U` square wave: both amplitude and residence time changed, and the wake response is nonlinear and history-dependent. The two experiments may be compared as distinct predeclared action families, but neither justifies linear extrapolation to another period or amplitude.

Raw cases are under `cfd/tandem_cylinders/cases/longdwell075_*_20261003/`. Audited results are under `artifacts/tandem_cylinders/longdwell075_result_20261003/`. The result is retained whether positive or negative; it is not used to retrospectively select a favorable subwindow.

The worker and Spark copies of all 16 key raw files (front/rear force histories, solver logs, and case configs) were byte-identical after transfer. Their committed checksum manifest is `raw_key_files.sha256` (manifest SHA-256 `99c81b998e681caa243b70f6cb946bdc1e434b0b855988f51e283aff58a3f689`).
