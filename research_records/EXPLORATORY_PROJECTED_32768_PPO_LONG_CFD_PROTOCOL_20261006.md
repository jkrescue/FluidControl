# Reflection-projected 32768 PPO: fixed 800-cycle paired CFD protocol

Status: **source and CPU-test preparation only; CFD execution is not approved**.

## Single hypothesis and intervention

The completed 680-step read-only audit found a large same-sign mirror defect in the
actual final 32768-step policy: `pi(o)+pi(Ro)` mean `-0.6702486725962338` and RMS
`0.8046740447610694`.  This diagnostic motivates one testable hypothesis: part of
the observed rear-lift mean bias is caused by the learned policy's violation of the
physical reflection relation.  It does not establish that projection will repair
the CFD trajectory or preserve drag reduction.

The only scientific change relative to the immutable original driver SHA
`17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa`
is the requested action:

`requested = 0.5 * (pi(o) - pi(R(o)))`.

`R` reverses the 32 probe positions, keeps probe `u` and front/rear `Cd` even,
and negates probe `v`, front/rear `Cl`, and applied rear-cylinder `omega`.  The
existing amplitude/rate filter is then called exactly once.  Neither policy,
VecNormalize, CFD solver, reward, observation, action limits, nor model weights
are changed.  Every step persists the raw `pi(o)`, raw `pi(Ro)`, projected
request, and final existing-filter record.

## Fixed execution and evaluation

- Actual policy/VecNormalize/training-result triple remains SHA-bound to the
  completed 32768-step, 24-reset policy; inference is deterministic CPU only.
- Same restart at CFD time 148, `dt=0.005`, 800 control cycles of `0.1 D/U`, and
  the same independently rerun zero-control branch.
- Same OpenFOAM image, source tree, transport functions, two solver containers,
  and cleanup ownership checks as the accepted original driver.
- Same six predeclared windows and formulas: early 12.4, its two 6.2 halves,
  primary `(168,228]` with 12,000 samples, historical companion `[168,228]`
  with 12,001 samples, and full `(148,228]`.
- The actual original result SHA
  `b425bd28ea6e1ca6786ee6ea38dd3a09e13191849a5778270b987a830584e827`
  is a historical same-protocol comparison, not a concurrent third branch.
- Original physical objectives remain unchanged: drag reduction at least 2%,
  rear-`Cl` fluctuation RMS ratio at most 1.05, and absolute rear-`Cl` mean over
  paired-zero RMS at most 0.10.  The existing 0.20 value remains sensitivity
  reporting only.  No result is automatically admitted.

## Resources and stopping

The controller remains CPU 8 GiB/no swap; each of the two CPU OpenFOAM solvers
remains 8 GiB/no swap and two CPUs.  Physical `MemAvailable` must be at least
50 GiB before startup and 22 GiB throughout, preserving the 20 GiB reserve.
The inner deadline is 3600 s and the outer systemd limit is 3750 s with 120 s
cleanup.  A failure is retained; it does not authorize a restart or protocol
change.

## Interpretation limits

The projection mathematically makes the wrapper request odd under `R`; it does
not make the finite-time CFD state symmetric, guarantee zero mean lift, guarantee
drag reduction, or prove causality.  The initial shedding phase is not required
to be symmetric.  This is an exploratory single-factor real-CFD test, not PPO
training, surrogate admission, or completion of the original 80-D/U goal before
its actual terminal metrics are reviewed.
