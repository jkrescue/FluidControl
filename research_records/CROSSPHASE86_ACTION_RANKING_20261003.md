# Validation-only t=86 action-ranking audit

Date: 2026-10-03

## Scope

This is a short, open-loop decision diagnostic. It compares an official
PhysicsNeMo FNO checkpoint with three new real OpenFOAM v2512 trajectories from
one validation-only shedding phase. It is **not** a closed-loop result, a Gate-B
replacement, a complete shedding-cycle comparison, or evidence that the
project's physical acceptance target has been met.

The panel and objective were recorded before launching CFD in
`artifacts/tandem_cylinders/crossphase86_ranking_predeclared_20261003.json`.
All cases start from the uncontrolled `t=86` snapshot, run to `t=88`, and are
ranked by mean system-total Cd at the 20 action frames `t=86.1..88.0`.

## Split and restart audit

The current checkpoint's resolved training root is
`data/curated/tandem_cylinders_gate_b_aug_v3/train`. All 26 HDF5 files under
that root carry `split=train`. The audit located the 26 training frames whose
physical timestamp is exactly 86.0 and compared each state tensor elementwise
with `phase_validation_86_multisine.h5:frame0`. There were zero exact matches.
The validation initial-state SHA-256 is
`80e01e94e6c3fc301eb80f389e7b7a4352f19fc3291135211a4ff661f4e5abba`.

The common source restart hashes are:

- U: `caf3393a152b83f5a632f2b86dcc9e4f441cff965a835f19dc0fdb7dc6d86512`
- p: `6fe8b8d54093253cccdfe20622598016b9ef9184196171896f3e2fbdf470610e`

After generation, each case's raw U file intentionally differs because the
future rear-wall action table is embedded in that file. This is not a different
initial flow state: every table begins with `omega(86)=0`, and all cases were
copied from the hashes above. The raw p files remain byte-identical.

## Frozen actions and solver quality

The three targets are zero, +1, and -1. The nonzero actions change by 0.1 per
0.1 D/U from `t=86` through `t=87`, then remain fixed. Thus the initial action
and ramp rate are common and bounded; no CFD or FNO result was used to modify
the schedules.

Each case completed 400 steps with `deltaT=0.005`. Maximum Courant number was
below 0.245, maximum absolute global continuity error per step was below
`9.4e-13`, and every solver log ended cleanly. The three jobs were limited to
4 CPU and 8 GiB each; host MemAvailable remained about 98 GiB.

## Real CFD and FNO ranking

The audited model is epoch 9 of the H20 plus rear-drag checkpoint. Inference
was performed on CPU, leaving the training GPU untouched.

| Action | CFD mean total Cd | FNO mean total Cd | CFD rear Cl mean | CFD rear Cl' RMS |
| :-- | --: | --: | --: | --: |
| zero | 2.263126 | 2.273798 | 1.303788 | 0.353141 |
| ramp to +1 | 2.409336 | 2.393302 | 1.137705 | 0.276075 |
| ramp to -1 | **2.146801** | **2.175208** | 1.486363 | 0.459963 |

The FNO gets all three pairwise comparisons correct and selects `-1`, matching
the CFD minimum with zero ranking regret. On this fixed two-unit window, the
selected CFD action has 5.140% lower mean total Cd than zero. The +1 action has
6.461% higher mean total Cd than zero.

This correct local drag ranking does not make `-1` an acceptable controller.
Its rear lift fluctuation RMS is 1.3025 times the zero case, a 30.25% increase,
so it fails the project's canonical lift non-degradation condition. Conversely,
the +1 action lowers rear Cl' RMS by 21.82% while increasing drag. The short
window is only 2 D/U, less than one approximately 6.2 D/U shedding period, and
is strongly phase biased. Its selected-action rear mean lift of 1.4864 also
does not satisfy the canonical mean-lift bound of 10% of zero-case Cl' RMS
(0.0353). This observed short-window mean cannot be treated as a cycle mean or
used to claim full-cycle physical acceptance.

The earlier `t=80` H20 plus rear-drag ranking also selected the negative action.
That same-sign choice is useful consistency, but it is not two statistically
independent complete-cycle replications: both starts come from the same
deterministic uncontrolled baseline, and the present `t=86` response covers
less than one period. The two panels must remain separately reported local
ranking diagnostics.

## Decision

The result supports a narrow claim: at a second validation phase, the current
FNO preserves the drag ordering of three small, predeclared actions over its
20-step training horizon. Together with the earlier `t=80` ranking, this is
useful evidence for guarded candidate screening.

It does not clear Gate B and does not justify unprotected surrogate PPO or a
closed-loop benefit claim. The drag/lift tradeoff shows why the next controller
must score total drag and rear lift jointly and then be verified by paired CFD
over complete shedding cycles and multiple independent phases.

Machine-readable result:

`artifacts/tandem_cylinders/crossphase86_fno_cfd_ranking_20261003.json`
