# Scientific-value audit: tandem-cylinder closed-loop control

Date: 2026-10-03 UTC. Scope: two fixed-centre tandem cylinders at Re=100,
L/D=5, rear-cylinder rotation; real OpenFOAM labels, official PhysicsNeMo FNO,
and a possible later HydroGym control interface. This is a research decision
record, not a claim of achieved control.

## Literature boundary and what is not novel

- Zhao et al., *Ocean Engineering* 306 (2024), 118138,
  [PolyU record](https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/):
  PPO-controlled rear-cylinder self-rotation at L/D=5 reduced rear-cylinder
  lift fluctuation by 98% in their CFD setting. Repeating rotation/PPO on the
  same geometry is therefore not a publishable novelty by itself. Their
  reported primary target is lift mitigation, not our system-total mean drag.
- He et al., *Ocean Engineering* 333 (2025), 121453,
  [article](https://doi.org/10.1016/j.oceaneng.2025.121453):
  tandem-cylinder DRL jet control already addresses drag and lift. Our
  actuator (rear rotation), objective (total drag), and verification must be
  explicit; merely saying "tandem-cylinder DRL drag reduction" is too broad.
- Ye and Elsheikh, *Physics of Fluids* 37 (2025), 093363,
  [author repository](https://github.com/gymprecice/mbrl) and
  [university record](https://researchportal.hw.ac.uk/en/publications/model-based-reinforcement-learning-for-active-flow-control/):
  PETS/MBPO with real OpenFOAM cylinder jet control report comparable control
  to PPO at lower CFD cost. Model-based flow control and multistep surrogate
  loss are established; neither is our standalone contribution.
- A 2025/2026 confined-square-cylinder study uses a surrogate and PPO with
  alternating CFD interactions,
  [paper](https://doi.org/10.1016/j.engappai.2025.112468). Thus "FNO + PPO"
  or "surrogate + RL" alone is an integration exercise.
- [HydroGym](https://github.com/dynamicslab/hydrogym) supplies standardized
  flow-control environments and RL interfaces. Using it is reproducibility
  infrastructure, not a scientific finding. Its shipped cylinder environment
  is not automatically the PolyU tandem-rotation geometry.
- Bian et al., *Applied Energy* 404 (2026), 127035,
  [author-hosted paper](https://flowphysics.ucsd.edu/assets/papers/BianEtAl_2026_AE.pdf),
  show that field error can disagree with control-landscape error. This
  supports checking whether a surrogate preserves *control ranking*, in
  addition to the existing 100-step force NRMSE. It does not validate our
  present model.

## Local physical evidence and current gap

The v3 cohort contains 35 real OpenFOAM trajectories (26 train, 4 validation,
5 frozen test); the data are not PolyU measurements. The existing
`control_objective_v3_audit_20261002.json` shows that, among the 30
train/validation trajectories, five have lower mean total Cd than their
phase-matched zero-action baselines, but **none** of those five also has
rear-cylinder mean absolute Cl at or below baseline. These are mostly
random/open-loop schedules, not an impossibility theorem for feedback. The
previous constant-rotation panel also reduced rear Cd while increasing rear
Cl RMS (see `cfd/tandem_cylinders/CASE_SPEC.md`, section 6.5).

The frozen Gate B is 100-step total-drag NRMSE <=10%, independent-phase
NRMSE <=10%, finite dynamics, persistence and action sensitivity. The best
completed model so far, an official FNO fine-tuned over 20 steps, has 15.522%
100-step validation NRMSE and 17.769% five-case frozen-test NRMSE; its
independent-phase test is 13.265%. It **fails** both NRMSE checks. A
rear-Cd-weighted ten-step FNO improved validation from 18.927% to 16.714%
but had 19.545% frozen-test NRMSE. Neither licenses surrogate-only PPO or a
physical drag-reduction claim.

## Corrected publication hypothesis and acceptance

Keep the physical problem and primary total-drag target, but do not present
"PhysicsNeMo + HydroGym" as novelty. Test this falsifiable hypothesis:
**a control-aware, uncertainty-guarded model-based controller can reduce
system-total mean Cd through rear rotation with measured lift and actuation
trade-offs, while consuming fewer CFD solver-hours than CFD-only PPO, and
retaining benefit across independent shedding phases.** If the CFD data do
not support that joint claim, report the Pareto frontier and failure mode,
or explicitly switch the primary endpoint back to PolyU-style rear lift;
never silently change the endpoint after seeing test outcomes.

For every final controller, compare paired OpenFOAM runs from the same
restart and averaging window: zero rotation, best simple periodic/constant
open-loop control, and the frozen closed-loop policy. Report front/rear/total
mean Cd, rear and front Cl RMS, action RMS/rate, estimated actuator work or
torque if available, confidence across phases, mesh/time-step sensitivity,
and CFD wall time. A decrease in drag accompanied by a large rise in lift
must be described as a trade-off, not unqualified success.

Add a *separate* control-sufficiency audit on a predeclared validation-only
action panel: ordering/rank correlation of CFD vs FNO action returns, regret
of the FNO-selected action evaluated in CFD, and out-of-distribution or
uncertainty warnings. This supplements, but does not retroactively relax,
Gate B's 10% accuracy benchmark. An active-learning round using official
PhysicsNeMo query/label protocols is justified only if this audit reveals a
localized train/validation coverage gap; OpenFOAM produces actual labels,
with equal-budget random acquisition as the baseline. Frozen test cases
never drive acquisition or hyperparameter selection.

## Immediate execution

1. The 20-step FNO epochs 9 and 10 were compared *only on validation*;
   epoch 10 is slightly better at 100 steps (15.522% vs 15.559%). No
   post-hoc test selection was performed.
2. A predeclared combination of the two validation-improving ablations
   (20-step rollout and rear-Cd loss weighting) passed a worker GPU smoke
   and began ten-epoch training on the compute-only node at 00:59 UTC.
   Primary Spark remains the sole durable artifact and test host. Final
   validation must strictly beat its parent 15.522% before frozen Gate B.
3. Independently design a small paired-action CFD control-landscape panel
   from a training/validation restart. Its actions, solver settings, and
   analysis windows must be fixed before solving. This is for feasibility
   and model ranking, not a claim of a learned policy.
