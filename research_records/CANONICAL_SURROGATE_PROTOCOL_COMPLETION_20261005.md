# Canonical surrogate receipt protocol completion

Lead decision D012, recorded 2026-10-05 Asia/Shanghai before FC-P003/P003B
complete post-evaluation. No model has been admitted by this decision.

## Missing definition, not missing successful experiments

Independent source/history review found that the original canonical PPO
window/dynamic receipts were consumer compatibility schemas with boolean
requirements, not implemented numerical producers. The 2026-10-03 document
requires causal 6.15-D/U force fidelity and rate-limited action response but
does not specify numerical surrogate-error tolerances. Do not describe those
schemas as historically quantified independent scientific experiments.

The numerical development protocol was explicitly introduced on 2026-10-04.
Adopt those already fixed criteria for the missing canonical numerical
producers now, with this date and source recorded. This is prospective
protocol completion, not retrospective preregistration or threshold relaxation.

## Unchanged computations and tolerances

Use the exact candidate, normalization and validation-only b01/b05 ×
minus/zero/plus evidence from the fixed post-evaluation protocol.

- Window: one start0 H100 free rollout per case; final 62 sampled values on
  the 0.1-D/U grid span 6.1 D/U and represent the requested trailing 6.15-D/U
  interval. For every branch separately: absolute mean-total-Cd error <=1%
  of same-window zero-CFD mean-total-Cd; absolute rear-Cl-prime RMS and mean-Cl
  errors each <=2.5% of same-window zero-CFD rear-Cl-prime RMS. All branches
  must pass; no favorable averaging or endpoint substitution.
- Canonical dynamic compatibility: adopt the existing `audit_full40_dynamic6_fno.py`
  computation predeclared in commit 72b62ac: pooled total-Cd NRMSE over ALL
  rolling H100 segments <=0.10, plus start0 four action-minus-zero Cd pair
  MAE <=0.023. Independently verify the actual HDF action sequence contract
  |omega|<=0.75 and |delta omega|<=0.1 per control step, H100, validation-only
  b01/b05 scope. Preserve all existing audit checks.
- The additional development dynamic requirement stays separate: its pooled
  Cd NRMSE uses only the six start0 H100 terminal points, not all rolling
  segments; it also requires delta-Cd MAE <=0.023 and perfect non-tie sign/order
  with the existing 1e-12 tie tolerance. Both contracts must pass. Never label
  these two differently aggregated NRMSE values as interchangeable.
- Existing validation10 endpoint gate and additional development admission
  remain required. Receipts may share underlying evidence; they are not
  statistically independent experiments or new CFD observations.

## Implementation authorization

Implement a new CPU-only producer after independent code review. Recompute
the numerical decisions from SHA-verified source evaluation/force evidence,
not from existing PASS booleans. Bind the candidate, source evidence,
producer version, this protocol and actual threshold values in both receipts.
Emit explicit FAIL when criteria fail. Missing/inconsistent input remains
an operational/schema error, not a scientific pass or a zero metric.

Use new output paths outside immutable completed posteval bundles; preserve
all original results and running scripts. Keep the legacy receipt field names
only for interface compatibility, and explicitly record this protocol's date
and shared-evidence relationship. Test with real lambda0/lambda10 negative
cases and synthetic unit fixtures clearly distinguished from scientific data.

SOTA owns implementation; Evaluation independently reviews reproducibility,
negative cases, same-candidate binding and legacy consumer compatibility.
This authorization permits CPU diagnostics/receipt generation, not PPO GPU
execution. A qualified candidate still requires separately reviewed new PPO
training, followed by real OpenFOAM paired80D/final60D physical verification.
The final >=2% drag reduction, <=1.05 lift-fluctuation ratio and <=0.10 mean
lift bias requirements remain unchanged.
