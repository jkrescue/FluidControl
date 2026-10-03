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

## Update: independent paired-action CFD check

The predeclared long validation-only OpenFOAM panel has completed and passed
solver/data QC (see `CONTROL_LANDSCAPE_LONG_CFD_20261003.md`). Relative to an
identical zero-rotation restart, `omega=+1/-1` lowered system-total mean Cd
by 4.66%/3.49% over `t=120..160`. The reduction is in rear-cylinder Cd,
whereas front-cylinder mean Cd is nearly unchanged. All six paired drag
blocks favor the rotating case, but the block trends warn against assuming
statistical stationarity. This supplies a **physical open-loop feasibility
signal**, not a learned closed-loop result.

Crucially, rear-cylinder total Cl RMS (including the nonzero mean side load)
rose by 32%/24%; rear mean absolute Cl rose 21%/10%. The late-window
fluctuating Cl RMS fell only ~3%. An apparent conflict with an older
constant-rotation table was resolved by re-reading its raw OpenFOAM forces:
the old ~1.517 was **total** RMS including mean, while its fluctuating RMS
was ~1.137, close to the new ~1.143. Instant/ramp onset made negligible
difference in a matched comparison; see `CONTROL_ONSET_REPLICATION_20261003.md`.
The research
target remains *system total drag subject to explicit side-load and effort
constraints*, with a Pareto-frontier fallback. Do not hide the side-load cost
inside a single reward or describe the open-loop result as a closed-loop gain.

The separate 100-step validation action-ranking check found that the current
best completed FNO selected `+1` rotation whereas paired CFD preferred `-1`.
Its selected action was slightly worse than zero in the startup window;
see `CONTROL_RANKING_AUDIT_20261003.md`. This is the decisive present
*model-control gap*: physical actuation can lower drag, but the current
surrogate cannot yet be trusted to select the beneficial action. Pause
surrogate-only PPO/MPC claims, preserve frozen Gate B, and prioritize
control-aware model validation and train-only real-CFD acquisition.

The canonical objective already locked a separate mean-rear-lift bound:
`|mean Cl_rear| <= 0.1 * uncontrolled Cl_rear,rms`. In the long matched panel
that limit is `0.117904`, whereas constant `+1/-1` rotation produces
`1.06594/0.91422`. Hence **neither constant action satisfies the existing
physical acceptance criteria**, despite meeting the 2% drag-reduction
threshold. This strengthens—not weakens—the need for phase-aware time-varying
control. The scientific question remains valid, but an open-loop drag-only
result must never be reported as achieving it.

Two predeclared zero-mean smooth periodic open-loop CFD controls (periods 10
and 20) have also completed. They pass the mean-lift bound but fail the joint
drag/fluctuation criteria: period 10 changes total Cd by `+0.586%` and rear
fluctuating Cl RMS by `+10.3%`; period 20 changes them by `-0.919%` and
`+13.8%`. See `PERIODIC_ROTATION_BENCHMARK_20261003.md`. This establishes
concrete baselines for a learned controller to beat, but only at one phase
and one coarse mesh. It does not rescue the current FNO's failed action
ranking or prove feedback benefit in advance.

A retrospective audit of all **30 existing train/validation** v3 CFD
trajectories at the same fixed late window found `0/30` meeting all three
locked physical checks, although four lower total Cd by at least 2%; none
of 30 meets the rear fluctuating-lift check. The five frozen test trajectories
were excluded. See `EXISTING_OPEN_LOOP_COHORT_AUDIT_20261003.md`. This is
evidence that the joint objective is nontrivial for the current open-loop
cohort, not proof that feedback is necessary or sufficient. Keep the objective
and its physical safety constraints; repair the surrogate's decision fidelity
before any learned closed-loop success claim.

## October 3 literature refresh and go/no-go decision

The September 2026 preprint by Sharma and Chakravorty,
[A Two-Stage, Model-Based Reinforcement Learning Approach for Active Flow
Control of Bluff Body Wakes](https://arxiv.org/abs/2609.08436), reports a
partially observed, model-based controller on a **single** Re=100 cylinder,
with eight pressure sensors, 44% drag reduction and suppression of lift
oscillations in its own high-order numerical setting. It makes generic
"model-based closed-loop cylinder control" an even weaker novelty claim. It
does **not** establish our tandem rear-rotation, *system-total* drag result;
nor can its percentage be compared directly to this different geometry,
actuator, drag denominator or OpenFOAM mesh. A 2026 experimental tandem
rotary-actuation [preprint](https://arxiv.org/abs/2605.20778) concerns two
**vibrating** cylinders and vibration suppression, not two fixed cylinders
under our joint drag/side-load constraint. These papers narrow novelty but
do not invalidate the physical question. PolyU's [2024 fixed-tandem
rear-rotation PPO study](https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/)
already claims 98% lift-fluctuation reduction at L/D=5, so a rear-lift-only
result would be a replication target, not a new contribution by itself.

More importantly, the [2026 HydroGym Nature paper](https://www.nature.com/articles/s41586-026-10917-6)
already controls a three-cylinder **triangular fluidic-pinball** wake by
surface rotation and targets collective drag with an absolute-lift penalty;
it reports approximately 90% drag reduction in its Re=100 2D pinball
configuration. Therefore "multi-cylinder rotation + HydroGym + joint
drag/lift reward" is **also prior art**, not our method claim. Our tandem
centres, single downstream actuator, fixed geometry and hard mean-lift
constraint are different, but merely changing geometry is not sufficient
scientific novelty. The paper would need an independently useful
decision-fidelity/CFD-efficiency finding and a physically verified,
phase-robust constrained-control result. Its pinball percentage must not be
used as a numerical benchmark for our different drag denominator or actuation.

**Present verdict: promising physical trade-off, unproven controller, no
publishable gain yet.** Real CFD shows >=2% total-drag improvement is
possible under constant rear rotation, but both signs violate the locked
mean-lift bound by roughly 8-9 times. Thirty existing open-loop
train/validation trajectories and two controlled periodic baselines give
zero joint successes. That sample is neither exhaustive nor a proof that
feedback is necessary. The best FNO fails the locked long-horizon accuracy
gate and selects the wrong signed action on a validation panel. HydroGym
compatibility and high GPU utilization do not change this verdict.

Continue the same objective, but impose a **bounded go/no-go sequence**:

1. Finish the already running official-PhysicsNeMo H20/rear-drag ablation.
   Validate first; use the frozen test only if it strictly improves the
   previous validation NRMSE, and do not infer control competence from
   force NRMSE alone. On the unchanged four-action validation panel, record
   whether it selects CFD's best sign and its CFD regret.
2. Finish the two train-only signed-pulse CFD labels and Curator v4 build.
   Before evaluating a v4 model, lock a control-decision check on the same
   four validation actions: >=5/6 pairwise orderings, select CFD's true-best
   action, and <=0.01 mean-total-Cd CFD regret. This *adds* an operational
   control-sufficiency screen; it does not replace Gate B's <=10% 100-step
   drag NRMSE or remove the independent-phase check. Run an equal-budget
   random acquisition comparator before attributing gains to active
   learning.
3. Only after both model screens pass, advance to CEM-MPC and a short
   phase-matched OpenFOAM feedback replay; then test the locked Gate-D
   long-window criteria against zero and the open-loop controls. PPO and
   HydroGym interoperability follow demonstrated feedback benefit, not
   vice versa.
4. If the corrected model cannot preserve action ranking, do **not**
   spend further CFD on surrogate-only PPO. Analyze the decision boundary
   and acquisition bias, and present the negative model-control result.
   If real closed-loop replay improves drag but repeatedly violates the
   predeclared lift constraints, report the Pareto frontier rather than
   silently redefining success. A different primary endpoint requires a
   separately versioned objective and new validation campaign.

The defensible paper contribution, if these gates succeed, is not a new
PhysicsNeMo architecture or a new HydroGym solver: it is a reproducible,
phase-robust **control-aware surrogate validation and CFD-verified
constrained tandem-cylinder closed-loop result**, with CFD-hour savings
against an explicitly matched CFD-only controller. The key unknown is
whether any bounded rear-rotation policy meets the joint physical
constraints on a mesh-converged flow; current evidence leaves that open.
