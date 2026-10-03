# H20 + rear-drag FNO validation action-ranking audit

Date: 2026-10-03 UTC

## Scope and frozen-test guard

This is a validation-only decision diagnostic on the four action schedules that
were declared and solved with OpenFOAM before model ranking. It does not select an
action on the frozen test split, change the frozen 10% Gate-B threshold, or claim
closed-loop control performance.

The audited official PhysicsNeMo FNO is the epoch-9 best checkpoint from:

`artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/best`

Its existing 100-step total-drag NRMSE remains 13.414% on validation, 15.540%
on the frozen five-case test, and 10.812% on the independent-phase test. All
three figures remain interpreted against the unchanged 10% requirement; the
model therefore does not pass Gate B.

## Predeclared action-ranking result

The audit starts from the common uncontrolled OpenFOAM `t=80` state and rolls
out each fixed action schedule for 100 interactions at `dt=0.1`. The ranking
objective is mean system total Cd over `t=80.1..90`, with lower values better.

| Validation action | OpenFOAM mean total Cd | FNO mean total Cd |
| :-- | --: | --: |
| zero | 2.301306 | 2.429976 |
| ramp then +1 | 2.303039 | 2.384393 |
| ramp then -1 | **2.267244** | **2.383203** |
| shedding-period sine | 2.558163 | 2.640028 |

The FNO selects `-1`, matching the OpenFOAM optimum among the four candidates.
Its CFD regret is exactly 0 on this panel, and the selected action reduces the
startup-window CFD total drag by 1.480% relative to zero. Pairwise ranking is
5/6 (83.33%). The one wrong pair is zero versus `+1`: OpenFOAM makes `+1`
0.075% worse than zero, while the FNO ranks `+1` as better.

The predicted `-1` versus `+1` cost margin is only 0.001191, whereas the CFD
margin is 0.035796. The correct signed-action choice is therefore encouraging
but not well separated in model space. Maximum normalized-state amplitude was
8.016 or lower for every rollout, below the fixed 39.403 state guard.

## Decision

Compared with the earlier primary H20 and rear-weighted H10 audits, which both
ranked 4/6 pairs and incorrectly selected `+1`, the joint H20 + rear-drag model
improves local validation action selection to 5/6 and chooses the correct sign.
This is evidence that the combined training objective helps decision ranking on
this predeclared panel.

It is not sufficient to launch an unguarded surrogate controller: the frozen
100-step accuracy gates still fail, the `+1`/`-1` model margin is small, and the
panel contains only one restart phase and one startup window. The validation
ranking result must remain separate from frozen-test reporting.

Result artifact:

`artifacts/tandem_cylinders/control_landscape_fno_ranking_h20_rear_drag_20261003.json`

SHA-256: `f26144b38e58455cc096133db94be8c69d229b44bdafa9756b939dbc26f4b96b`
