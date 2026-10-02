# Six-hour execution and seven-hour acceptance window

Start: 2026-10-02 15:45 UTC. Target execution cutoff: 21:45 UTC;
acceptance review: 22:45 UTC. This schedule does **not** redefine a scientific
success threshold. The canonical project and all data/artifacts are on the
primary DGX Spark; the second Spark is temporary compute only.

## Locked research question and objective

At `Re=100`, tandem-cylinder `L/D=5`, use rear-cylinder rotation to reduce
the **time-averaged sum of front and rear drag** under front/rear lift,
rotation-amplitude and slew-rate safeguards. Compare against a phase-matched
zero-action OpenFOAM trajectory and simple-action baselines. Report separate
front/rear drag, lift RMS, rotation effort, numerical residuals and cost;
do not hide trade-offs inside PPO reward. The model observes real-CFD-derived
fields during surrogate training and sparse wake probes/forces during online
feedback. Geometry, Reynolds number and actuator remain fixed in this window.

This is an extension, not a numerical reproduction of Zhao et al. (2024):
their PolyU tandem-cylinder PPO study targeted downstream-cylinder **lift
fluctuation** under self-rotation. Its reported lift suppression cannot be
used as a target or comparator for our system-total-drag outcome.

## Immutable stage gates

1. PhysicsNeMo 2.2.2 FNO: frozen five-case real OpenFOAM v3 test (including
   untouched `expanded_test_05`) must have 100-step total-drag NRMSE `<=10%`.
   The unchanged Gate-B audit additionally requires finite rollouts,
   improvement over persistence, action sensitivity, and an independent
   shedding-phase test. Checkpoint choice uses training/validation only.
2. CEM-MPC: only a Gate-B-passing checkpoint may screen bounded action
   sequences. Report predicted total drag and every penalty separately;
   surrogate-only gains are not physical-control evidence.
3. HydroGym / PPO: train on the current **total-drag** objective and frozen
   sparse-observation/action contract, not on the older rear-cylinder reward.
   A candidate must improve independent validation/test surrogate starts and
   the exact shared `t=80` physical restart before real-CFD replay.
4. Real OpenFOAM feedback: freeze the policy, run paired phase-matched
   zero-action and controlled trajectories, and report time-averaged physical
   metrics and solver checks. A short integration smoke proves wiring only;
   sustained drag reduction needs a longer trajectory and numerical
   sensitivity. No control claim without this step.

The earlier 8192-step HydroGym PPO completed a real 32-interval OpenFOAM
feedback replay but **worsened** its earlier objective by `+0.003855` versus
zero action and increased mean drag by `+0.010697`. That run is a valid
integration precedent, not evidence for today's total-drag controller.
Current GPU jobs are FNO **multistep surrogate fine-tuning**, not PPO.

## Time-boxed execution

| Window | Work | Output / decision |
| --- | --- | --- |
| 15:45–17:30 UTC | Complete two independent 10-epoch rollout trainings and primary-host, unchanged frozen audits. | Model/seed/horizon table; fail-closed Gate-B decision. |
| 17:30–18:30 UTC | If either passes, run CEM and validate state/action/reward parity. If neither passes, localize 100-step error by case/action/phase and assess targeted CFD acquisition. | Audited control candidate or documented model/data failure mode. |
| 18:30–20:30 UTC | Passing branch: train/evaluate HydroGym PPO with the Stage-C total-drag objective; run matched `t=80` screen. Failing branch: execute only justified, train-only active CFD augmentation and retraining if the time/cost budget allows. | Frozen policy with split-level metrics or bounded active-learning experiment. |
| 20:30–21:45 UTC | Passing policy branch: guarded real OpenFOAM feedback and paired baseline, numerical checks. Otherwise finish reproducibility and error analysis. | Physical-control audit or explicit unmet gate. |
| 21:45–22:45 UTC | Preserve artifacts, verify dashboard/resource logs, push only validated code/docs. | Seven-hour acceptance record with exact successes, failures and remaining work. |

At no point does a missed deadline authorize lowering 10%, moving test cases
into training, selecting by test performance, bypassing the `t=80` gate,
or reporting surrogate reward as physical drag reduction.

## Optional active learning, only if Gate B still fails

PhysicsNeMo 2.2.2 in the pinned container exposes the official
`physicsnemo.active_learning` Driver and Query/Label strategy protocols.
Its current official [user guide](https://docs.nvidia.com/physicsnemo/26.05/user-guide/active_learning.html)
defines the train → metrology → query → label loop. The later
[surface-CFD aero recipe](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/active_learning_aero/README.html)
illustrates uncertainty-versus-random acquisition, but uses a different
model/geometry; do **not** copy its GeoTransolver/GP machinery into this FNO
case or assume its example is part of the pinned 2.2.2 image.

If the two rollout audits reveal a learnable data-coverage gap, predeclare a
pool of **new rotation histories** inside the existing amplitude/slew bounds;
score their *unlabeled* trajectories with ensemble disagreement and
training-support distance, select a small diverse batch, then run actual
OpenFOAM solves as the labeler. Reserve a same-size random-selection control
and never use frozen validation/test actions or their errors for acquisition.
Curate new outputs with the same solver/source checks, append to training
only, and compare Gate-B accuracy and CFD-hours at equal sample budget.
If the two seeds merely show generic autoregressive instability, further
unfocused CFD generation is not justified; improve the rollout loss and
validation selection first.

## Monitoring and provenance

`fluid-control-research-watchdog-20261002.service` samples both nodes every
60 seconds until acceptance time, with `Restart=on-failure`; samples and
alerts are under `artifacts/monitor/research_window_20261002/`. The dashboard
is also a restart-on-failure user service. Each GPU job retains its own
20 GiB available-unified-memory preflight and 0.20 allocator cap. The
watchdog observes and records; it intentionally does not restart a failed
training run or override a gate without diagnosis.

References: [PolyU tandem-cylinder study](https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/),
[HydroGym official repository](https://github.com/dynamicslab/hydrogym),
[PhysicsNeMo active-learning guide](https://docs.nvidia.com/physicsnemo/26.05/user-guide/active_learning.html).
