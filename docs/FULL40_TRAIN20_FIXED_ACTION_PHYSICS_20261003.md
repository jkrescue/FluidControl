# Full40 train-only fixed-action CFD physics (2026-10-03)

This is a **training-split diagnostic**, not validation, frozen-test evidence,
HydroGym performance, or closed-loop control. The source is 20 real OpenFOAM
trajectories at Re=100, tandem-cylinder L/D=5, with rear-cylinder rotation
only. Four predeclared start phases (b00, b02, b04, b06) each have the same
five actions: -0.75, -0.375, 0, +0.375, +0.75. Each controlled trajectory is
compared only with its same-phase zero-rotation branch over its **predeclared
last 60 D/U** (12,001 force samples at dt=0.005).

| Constant rear rotation | Macro total-Cd reduction | Worst-phase reduction | Rear Cl′ RMS / zero | Abs. mean rear Cl / zero Cl′ RMS | Joint passes / 4 |
|---:|---:|---:|---:|---:|---:|
| -0.75 | 2.310% | 1.930% | 0.983 | 0.634 | 0 |
| -0.375 | 0.587% | 0.394% | 0.996 | 0.319 | 0 |
| +0.375 | 0.593% | 0.391% | 0.996 | 0.320 | 0 |
| +0.75 | 2.322% | 1.925% | 0.983 | 0.636 | 0 |

The frozen joint criterion is mean total Cd reduction ≥2%, rear Cl′ RMS
ratio ≤1.05, and absolute mean rear Cl / same-phase zero Cl′ RMS ≤0.10.
Overall **0/16** nonzero-action train comparisons satisfy it. The ±0.75
branches produce useful mean drag reduction but introduce mean lateral load
roughly six times the permitted level; their worst start phases also miss the
2% drag threshold. Halving the rotation amplitude reduces the mean-lift bias
but also reduces drag benefit below 1%.

This is direct evidence that a scalar drag-only reward would select an
unacceptable fixed-action policy. It motivates the separately declared
`canonical_joint_v1` HydroGym reward and a phase-aware *time-varying* control
hypothesis. It does **not** prove that any such closed-loop policy can pass the
physical criterion; earlier zero-mean periodic/alternating controls also
failed at least one criterion. The full40 dataset's fixed-action trajectories
alone do not validate action switching, so dynamic real-CFD train/validation
support remains a subsequent requirement.

Reproduce with `python3 scripts/summarize_full40_train20_physics.py --help`.
Audited result: `artifacts/matched_start_full40_extension/train20_physics_summary.json`
(SHA-256 `b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7`).
The script checks the full40 predeclaration, 9-case aggregate/receipts and 11
new strict receipts, and reads train case configs and raw force histories only.
