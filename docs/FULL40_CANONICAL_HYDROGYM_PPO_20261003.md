# Full40 canonical HydroGym PPO (2026-10-03)

## Scope

This is a new, isolated control path. The historical `stage_c_total_drag`
objective remains unchanged for reproducibility, but it is not the current
three-part objective: it does not separately enforce rear-cylinder fluctuating
lift and mean-lift bias. No historical PPO result is relabelled as canonical.

The new runtime uses the official HydroGym `FlowEnv`, the existing
`TandemFNOStepper`, and the PhysicsNeMo FNO selected by the full40 validation
gate. It adds no new surrogate architecture. The observation is 69D: 64
velocity-probe values, front/rear `(Cd, Cl)`, and applied rear-cylinder omega.

## Physical and reward contract

- Action support is `|omega| <= 0.75`, identical to the full40 manifest and
  FNO normalization. This must never be confused with the old v4 `/5` scale.
- The control interval is `0.1 D/U`; `|delta omega| <= 0.1`, hence
  `|d omega/dt| <= 1`.
- Episodes contain at least 100 control steps.
- A trailing causal `6.15 D/U` force window computes total drag, rear `Cl'`
  RMS, and mean rear `Cl`. The canonical diagnostic thresholds are at least 2%
  total-drag reduction, `Cl' RMS <= 1.05` times the same-phase zero baseline,
  and `|mean Cl| <= 0.10` times the baseline `Cl'` RMS.
- Until the causal window becomes complete (62 point samples at `dt=0.1`),
  force/gate reward terms are zero. Only the explicitly logged action and
  action-rate penalties are active. This warm-up is auditable in every step.

PPO reward is a surrogate feasibility/search signal. Final physical
acceptance always recomputes the three metrics from a predeclared final
`60 D/U` real-OpenFOAM window; surrogate reward cannot replace that test.

The train20 open-loop audit is an explicit warning, not positive evidence for
PPO: constant `|omega|=0.75` reduces macro drag by about 2.3%, but its mean
rear-lift bias is about `0.63` times the zero-control `Cl'` RMS, far above the
`0.10` limit; `|omega|=0.375` reduces drag by only about 0.6%. The available
branches are static actions, so a dynamic feedback policy can be an
out-of-distribution rollout even inside `|omega|<=0.75`. This is why dynamic
rate-limited validation is a hard execution gate rather than an optional
diagnostic.

## Split isolation

- Training starts: zero-action CFD restart in train phases b00, b02, b04,
  b06. Four environments are used; PPO does not select a checkpoint from
  validation.
- Validation: b01 and b05 only, evaluated once for zero, constant `-0.75`,
  constant `+0.75`, and the final PPO policy. Each report includes individual
  reward components and the canonical ledger.
- Frozen test: never opened or enumerated by the PPO entry. The manifest count
  is an identity check only. Frozen evaluation requires a later, separate
  authorization after all decisions are fixed.

## Fail-closed readiness

`scripts/train_full40_hydrogym_ppo_canonical.py --execute` deterministically
recomputes `scripts/audit_full40_validation_gate.py` from the validation report,
segments, predeclaration, exact FNO model file, config, normalization, data
manifest, and pinned PhysicsNeMo image. It requires H100 total-drag NRMSE at or
below 10%, model total-drag MAE better than persistence, and the predeclared
matched-start action-ranking checks.

Because the formal checkpoint is trained first on the frozen-blind dev30
release, formal PPO also recomputes
`scripts/verify_dev30_full40_promotion.py` and requires its exact stored receipt
to be `DEV30_FULL40_PROMOTION_PASS`. The resulting PPO readiness record binds
the dev30 manifest SHA, final-full40 manifest SHA, both byte-identical
normalization SHAs, promotion-receipt SHA, verifier SHA, and selected FNO
checkpoint SHA. Thus a validation score alone cannot bypass development-to-
formal data lineage. This extra promotion prerequisite is intentionally not
required by the isolated train-only software smoke.

That endpoint gate is necessary but not sufficient for this reward. Execution
also requires cryptographically bound b01/b05 artifacts for:

1. causal-window fidelity of total drag, rear `Cl'` RMS, and mean rear `Cl`;
2. dynamic rate-limited action response, not only constant-action endpoints;
3. six same-phase zero baselines (four train and two validation) derived from
   their predeclared final `60 D/U` evidence.

Missing or altered evidence produces `FULL40_CANONICAL_PPO_EXECUTION_BLOCKED`
before HydroGym, PhysicsNeMo, or Stable-Baselines3 is imported. A hand-written
pass JSON is insufficient because the stored endpoint gate is compared with a
fresh deterministic recomputation and all evidence/source SHA-256 values are
checked.

## Train-only software commissioning

`--train-only-smoke` is a distinct path for testing PPO/checkpoint plumbing. It
requires the pinned combined image, exact full40 FNO generation, full40 data
identity, four train starts, and train-only zero baselines. It never opens
validation or frozen data. The baseline reader is bound directly to the
existing `train20_physics_summary.json` train-only artifact and rejects a
different scope or phase set. It emits
`train_only_software_commissioning_no_control_claim`. It is not Gate C and its
score must not be presented as generalization or physical control benefit.

## Auditable outputs and execution

Every PPO checkpoint is immutable within a new output directory and is logged
with timestep and SHA-256. Its train-only diagnostics include Monitor episode
return/length, applied-action mean/min/max/RMS, rate-limited fraction, and each
canonical reward component averaged over environment steps. These curves are
for learning/debug visualization only: they never read validation, never select
a checkpoint, and are not physical-improvement evidence. A progress JSON is
atomically refreshed after each iteration. The final report records the exact FNO checkpoint SHA, individual
reward components, zero/constant/PPO validation panels, and the explicit
`surrogate_only_requires_predeclared_real_cfd_replay` status.

Use `scripts/run_full40_canonical_ppo_spark.sh`. It pins the combined
PhysicsNeMo 2.2.2 + HydroGym commit `4ab9854` image, GPU0, the 20 GiB free-memory
guard, and the project-only artifact root. No training was started while this
entry was implemented. `--dry-run` uses a CPU-only 2-CPU/4-GiB container and
does not request a GPU; only actual smoke/formal execution enters the GPU guard.
