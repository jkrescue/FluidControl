# Dynamic6 real-OpenFOAM validation-only screening results

## Scope

The six predeclared OpenFOAM v2512 cases completed 4,000 solver steps each and
passed the aggregate source-state, action-table, solver-marker, numerical-QC,
same-phase-zero and split checks. The authoritative aggregate is
`artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json`
(SHA-256 `9723203f922cbe6609f2c92b7b48d299ee9d948d694d3c6c2eab413472421d86`).
Frozen-test data was not opened or enumerated.

The table reports prescribed-action validation screening, not an FNO
prediction or PPO/closed-loop result. Positive total-Cd change means drag
reduction. The canonical columns are total drag reduction at least 2%, rear
Cl fluctuation RMS ratio at most 1.05, and absolute mean rear Cl divided by
the paired-zero rear Cl fluctuation RMS at most 0.10.

| Phase | Action | Window D/U | Total-Cd change | rear Cl' RMS ratio | mean rear Cl bias ratio | Joint |
|---|---:|---:|---:|---:|---:|---:|
| b01 | minus | 0–10 | +5.8618% | 0.9414 | 0.1477 | FAIL |
| b01 | plus | 0–10 | -7.7076% | 1.0613 | 0.0998 | FAIL |
| b01 | minus | 10–20 | +5.9680% | 1.0675 | 0.0247 | FAIL |
| b01 | plus | 10–20 | -7.9126% | 0.9289 | 0.0582 | FAIL |
| b01 | minus | 0–20 | +5.9148% | 1.0086 | 0.0855 | PASS |
| b01 | plus | 0–20 | -7.8099% | 0.9954 | 0.0787 | FAIL |
| b05 | minus | 0–10 | -7.9262% | 1.0515 | 0.1146 | FAIL |
| b05 | plus | 0–10 | +6.2023% | 0.9535 | 0.1650 | FAIL |
| b05 | minus | 10–20 | -7.9682% | 0.9161 | 0.0446 | FAIL |
| b05 | plus | 10–20 | +5.9317% | 1.0814 | 0.0019 | FAIL |
| b05 | minus | 0–20 | -7.9471% | 0.9832 | 0.0790 | FAIL |
| b05 | plus | 0–20 | +6.0674% | 1.0227 | 0.0822 | PASS |

## Interpretation guard

The b01-minus and b05-plus mapping is a **post-hoc per-phase oracle** formed
after reading validation CFD results. The two full-window PASS rows only say
that those two independently predeclared action curves pass this 20-D/U
physical screen. They do not validate a phase-aware policy, cannot be pooled
as policy benefit, and do not authorize PPO. A defensible policy must define
its phase/action rule from train phases only, freeze it, and evaluate it on
held-out validation phases before a separate final 60-D/U real-CFD acceptance
run.

The half-window failures also show why the 20-D/U mean cannot be presented as
uniform-in-time control quality. This panel is useful evidence of strong phase
sensitivity and sign reversal, but it is not the final paper statistic.
