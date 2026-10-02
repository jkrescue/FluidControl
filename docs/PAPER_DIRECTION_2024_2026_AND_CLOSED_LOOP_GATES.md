# Paper direction for PhysicsNeMo–HydroGym closed-loop tandem-cylinder control

Date: 2026-10-02

The authoritative physical objective and change-control rule are maintained in
[`RESEARCH_OBJECTIVE.md`](RESEARCH_OBJECTIVE.md). If a later implementation note
conflicts with that file, the canonical objective takes precedence.

## Executive decision

The publishable direction is not simply “PPO on a surrogate.” It is a gated,
model-based active-flow-control workflow for tandem cylinders:

1. an official PhysicsNeMo FNO learns the action-conditioned flow transition and
   the forces on both cylinders from real OpenFOAM trajectories;
2. HydroGym supplies the standard Gymnasium-compatible control contract and is
   used to train/evaluate the policy in the learned environment;
3. long-horizon model error and out-of-distribution actions are measured before
   the policy is trusted;
4. every candidate policy is replayed in closed-loop OpenFOAM, followed by
   policy-distribution CFD data collection and surrogate retraining when needed.

The first scientific objective is system-level drag reduction, not vortex-induced
vibration suppression. The cylinders are currently fixed, so a VIV claim would be
physically unsupported. Rear-cylinder lift is retained as an anti-degeneracy and
load-fluctuation constraint.

## Why this is aligned with the 2024–2026 research frontier

The recent direction has moved from isolated, model-free PPO demonstrations toward
sample-efficient, robust and standardized closed-loop workflows.

- Zhao, Zhou, Ren, Tang and Wang studied the closest geometry and actuator to this
  project: a rotating downstream cylinder in the wake of an upstream cylinder.
  Their PPO policy suppressed downstream lift fluctuations by 98% at `L/D=5`, and
  their POD-guided sensor redesign improved generalization across spacings. This
  establishes the physical relevance of downstream-cylinder rotation and sparse
  wake sensing, but their target was lift mitigation rather than total drag
  reduction. [Ocean Engineering 306 (2024), 118138](https://doi.org/10.1016/j.oceaneng.2024.118138).
- Contemporary 2024 work also emphasizes robustness to varying operating
  conditions and partial measurements rather than a single nominal trajectory.
  Examples include adaptive square-cylinder control across wind direction and
  bluff-body drag control from partial measurements.
  [Physical Review Fluids 9 (2024), 094607](https://doi.org/10.1103/PhysRevFluids.9.094607),
  [Journal of Fluid Mechanics 981 (2024), A17](https://doi.org/10.1017/jfm.2024.69).
- Model-based RL became a central AFC direction in 2025. Ye and Elsheikh showed
  that PETS and MBPO can match PPO while using 2–9 times fewer environment samples
  and converging 2–7 times faster. Their ablations identify multi-step predictive
  loss, joint reward prediction and the model-utilization strategy as decisive,
  not neural-network complexity alone.
  [Physics of Fluids 37 (2025), 093363](https://doi.org/10.1063/5.0287427).
- HydroGym formalizes the solver-independent `PDEBase`, `TransientSolver` and
  `FlowEnv` interfaces and provides a common platform for data-driven flow control.
  This makes it valuable as the control/evaluation layer, while PhysicsNeMo remains
  the official learned-physics layer.
  [L4DC/PMLR HydroGym paper (2025)](https://proceedings.mlr.press/v283/lagemann25a.html),
  [official repository](https://github.com/dynamicslab/hydrogym).
- The 2026 HydroGym study extends this direction to more than 60 validated
  environments and demonstrates zero-shot transfer from inexpensive training
  environments to a substantially more expensive flow configuration. It reports
  four orders of magnitude lower exploration cost in its proof of concept, while
  explicitly leaving the breadth of transfer generalization open. Our tandem case
  should therefore treat transfer as a measured hypothesis, not an assumption.
  [Nature (2026), HydroGym](https://doi.org/10.1038/s41586-026-10917-6).
- In 2026, surrogate-based AFC work continues to emphasize hybrid environments and
  disturbance robustness. The PyFAC study combines SAC with an LSTM surrogate and
  explicitly tests free-stream disturbances, reporting a 60% training-efficiency
  improvement. This supports adding disturbance and CFD-correction experiments
  after the nominal closed loop works.
  [Physics of Fluids 38 (2026), 045140](https://doi.org/10.1063/5.0320929).

These papers do not prove that the present implementation is state of the art.
They define the comparisons and failure modes that a credible 2024–2026-style
study must address.

## Proposed paper hypothesis

An action-conditioned neural operator, trained on full flow fields and guarded by
multi-step error tests plus real-CFD refresh, can reduce the number of expensive
OpenFOAM interactions required to learn closed-loop rotation control of a tandem
cylinder while preserving the total-drag improvement under real CFD replay.

The key comparison is therefore not “FNO prediction looks plausible.” It is:

- control performance versus direct-CFD PPO and zero rotation;
- number of OpenFOAM control intervals needed to reach a fixed real-CFD benefit;
- performance before and after policy-distribution CFD refresh;
- generalization to independent shedding phases and, later, small changes in
  Reynolds number or cylinder spacing.

## Online closed-loop method to test

The recommended online method is a guarded hybrid loop, not an indefinitely long
rollout inside a frozen surrogate:

1. Bootstrap the official FNO from the existing open-loop OpenFOAM trajectories.
2. Train the HydroGym policy only on short model rollouts initialized from real-CFD
   replay-buffer states.
3. Limit the initial model rollout length to five control interactions. A 2025
   hybrid AFC study found that replacing its learned environment with CFD every
   five interactions controlled accumulated error while reducing training time by
   about 49%; five is therefore a literature-backed starting ablation, not a fixed
   truth for this flow.
   [Engineering Applications of Artificial Intelligence 159 (2025), 112468](https://doi.org/10.1016/j.engappai.2025.112468).
4. Trigger an earlier CFD anchor if the normalized state leaves training support,
   if action magnitude/rate enters a high-error stratum, or if a later official-FNO
   checkpoint ensemble disagrees beyond a calibrated threshold.
5. Append every CFD transition and physical reward to the real replay buffer;
   periodically fine-tune the world model, then re-run held-out and independent-
   phase gates before allowing longer synthetic rollouts.
6. Keep a frozen-policy evaluation path that never updates on the evaluation CFD
   trajectory. This separates online adaptation from the final control claim.

The principal ablation should compare fixed CFD anchor periods of 1, 5 and 10
control interactions with the adaptive gate. Report real-CFD calls, wall time,
total-drag benefit and failure rate at equal budgets.

## Stage gates

### Gate A — data and one-step model

- use only real OpenFOAM states/forces with documented provenance;
- predict `u`, `v`, `p`, front `Cd/Cl` and rear `Cd/Cl` with the official
  PhysicsNeMo FNO API;
- keep train/validation/test action histories independent;
- report component-wise and total-drag errors against persistence.

Passing Gate A only establishes a usable short-step world model. It does not
authorize PPO or a drag-reduction claim.

### Gate B — autoregressive control horizon

- audit horizons 1, 10, 50 and 100 at `dt=0.1`;
- require finite rollouts for every held-out segment;
- stratify total-drag error by action magnitude and action rate;
- compare observed-action rollouts with zero, sign-flipped and shuffled action
  counterfactuals to verify genuine action conditioning;
- cover at least one full baseline shedding period for reward estimation.

If one shedding period is not accurate enough, train with multi-step loss and use
short model-predictive segments with periodic real-CFD correction. Do not conceal
long-horizon drift by shortening the reported window.

### Gate C — HydroGym policy screen

Use a normalized running-window cost:

```text
J_t = mean_W(Cd_front + Cd_rear) / Cd_total_0
    + 0.20 mean_W(abs(Cl_rear)) / mean_abs_Cl_rear_0
    + 0.02 (omega / 5)^2
    + 0.01 (delta_omega / 0.5)^2

reward_t = -dt J_t
```

The return window must be at least one shedding period; two to three periods are
preferred. HydroGym should expose the policy interface, termination rules and
auditable reward components. It must label the backend truthfully as a PhysicsNeMo
surrogate, not CFD.

### Gate D — real-CFD closed loop

Freeze the candidate policy and compare it with a phase-matched zero-action case
in OpenFOAM for at least 5–10 shedding periods. Initial acceptance criteria are:

- at least 2% reduction in mean total drag;
- no more than 5% increase in rear-cylinder `Cl RMS`;
- absolute mean rear `Cl` no greater than 10% of the uncontrolled rear `Cl RMS`;
- separately report both cylinders' forces, action energy, action-rate cost and
  every termination or solver failure.

Replay the accepted policy once on the medium grid. The completed constant-action
grid check supports the coarse screening mesh, but it does not validate a learned
closed-loop trajectory.

### Gate E — policy-distribution refresh and robustness

Collect new OpenFOAM trajectories from several independent shedding phases using
the candidate policy plus bounded exploration. Retrain the official FNO and repeat
the frozen-policy CFD replay. Only then extend to disturbance, Reynolds-number or
spacing variation.

## Required ablations for a paper

| Question | Minimum comparison |
|---|---|
| Does the front force matter? | rear-only reward/model versus total-drag model |
| Does multi-step training matter? | one-step loss versus rollout-aware loss |
| Is the model exploiting open-loop data bias? | before/after policy-guided CFD refresh |
| Is action conditioning real? | observed versus zero/sign-flip/shuffled actions |
| Is HydroGym adding scientific value? | same frozen reward/policy contract on surrogate and OpenFOAM replay |
| Is the result phase robust? | multiple independent shedding-phase restarts |
| Is the benefit numerical? | coarse policy discovery plus medium-grid replay |
| Is it sample efficient? | real-CFD interactions and wall time versus direct-CFD PPO |

At least three random policy seeds should be used for the final comparison. Mean,
standard deviation and failure rate must be reported; selecting a single favorable
checkpoint is insufficient.

## Current implementation status

- The 24/4/4 OpenFOAM trajectory split already contains all four force channels,
  so no synthetic force labels are needed for the first total-drag retraining.
- The official PhysicsNeMo FNO has been extended through its supported
  `out_channels` configuration from five to seven outputs; no custom neural
  architecture was introduced.
- A seven-output smoke train, checkpoint reload, held-out rollout and total-drag
  evaluation have passed on GPU0 under the 20 GiB reserve guard.
- The 30-epoch total-drag training run completed successfully with an official
  PhysicsNeMo 2.2.2 FNO. The epoch-30 validation field MAE is `0.00126526` and the
  normalized four-force MAE is `0.0330082`.
- Observed, zero, sign-flipped, shuffled-action and independent-phase 1/10/50/100
  evaluations completed without a failed rollout segment. A fail-closed Gate-B
  audit now computes total-drag NRMSE at the full-period proxy horizon before
  authorizing CEM-MPC or routing to multi-step fine-tuning.
- HydroGym PPO remains intentionally paused. This prevents a policy from exploiting
  a surrogate before its control-horizon fidelity has passed the fixed threshold.

## Immediate next actions

1. Publish the fail-closed Gate-B NRMSE audit for the completed epoch-30 checkpoint.
2. If total-drag error fails at the shedding-period horizon, fine-tune with
   multi-step loss on the isolated second DGX Spark before any new PPO run.
3. If Gate B passes, run CEM-MPC first using the already-audited normalized
   running-window system objective and per-term reward ledger.
4. Run a frozen-policy long OpenFOAM comparison before making a control claim.
5. Generate policy-guided CFD only where the held-out/action-stratified audit shows
   insufficient support.
