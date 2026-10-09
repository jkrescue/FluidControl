# FC-P027 — short-horizon force-error decomposition

## Lead scope update — 2026-10-06

K4 independent terminal review has completed (FC-E038; Git52b9d69).
Root has approved isolated CPU implementation and synthetic engineering unit
tests only. The design-only wording below describes the original draft; actual
HDF/model access, GPU execution and scientific runs remain unapproved.

Historical recorded actions must remain unchanged. If their actual omega or
delta-omega violates the existing canonical cost contract, keep that case in
all force-error and physical-statistic summaries, report `cost_available=false`
with the precise reason and actual action values, and do not fabricate zero
cost or clip actions. Aggregate costs only over explicitly counted valid cases;
report unavailable counts separately without changing the44-case force
denominator. Mixed-history physical gate booleans are not admission evidence:
report diagnostic statistics/costs without presenting those booleans as a
physical pass or a P027 success label. These clarifications change no physics,
stored data, prediction threshold or control-admission standard.

Status: **DESIGN ONLY**. No implementation, data scan, model inference, GPU run,
optimization, formal evaluation, PPO, MPC or CFD execution is authorized.
Root review and a separate execution approval are required. K4 formal FAIL has
been observed; independent terminal verification remains authoritative. Neither
rejected P026 candidate becomes admissible through this diagnostic.

## Question and fixed comparison

At one fixed, fully warm origin in each of the existing44 train trajectories,
does force-prediction error already occur under true-state conditioning, or
does using frozen-flow predictions over10 steps add substantial error?

Use terminal P026 K1 and K4, unchanged official FNO models and reviewed dual
loader/history adapter. No new architecture, parameters, loss, optimizer,
checkpoint, data generation or action search. Keep normalization, masks,
residual-field interpretation, force pooling and precision unchanged. Final
implementation/source/candidate/manifest/config hashes must be pinned before
execution; this document does not substitute provisional hashes for approval.

## Data and exact times

Use all20 base,8 dynamic-train8 and16 historical direct-PPO train trajectories,
and only their train split. Their manifests declare801,201 and129 frames,
respectively. Set absolute trajectory origin `t=51` for every case and H=10.
This is a new diagnostic index selection, not a change to the original1368
training windows or their denominator. Exactly44 origins are expected. Missing
frames or identities cause failure, not silent case removal.

The required stored frames are0 through61 for past-force and future-target
statistics; field inference reads history48:51 and states needed through61.
All cases have the three actual past fields; this diagnostic has44 warm origins
and zero padded origins. It makes no cold-reset claim. In particular, the
original train16 H100 starts0–28 cannot supply52 preceding trajectory endpoints;
the present origin51 is deliberately distinct and must be reported as such.

For each transition `j=1,...,10`, both models predict force at absolute frame
`51+j`, compared with unchanged HDF force target at that frame.

- True-state H1: current state is true frame`50+j`. K4 field history ends there
  and contains its three preceding true frames. This is an offline teacher-forced
  diagnostic: later true states are not available to the free-AR prediction.
- Free AR: initialize actual field history48,49,50,51; advance only with the
  frozen flow FNO. Never replace any predicted field with true frame52 onward.
  Aerodynamic field outputs remain discarded, with no force feedback to flow.
- Both paths use the same stored prescribed-action sequence. At transitionj,
  the flow/K1 action pair is frames`50+j,51+j`; K4 uses four stored past/current
  actions ending at`50+j`, followed by the same next action at`51+j`. This is
  the existing recorded-action sampling convention, not a claim that float32
  interpolated actions are exact nominal-time online commands.
- Persistence predicts all10 future four-force vectors as the single origin
  vector, never a later truth value. State its force-source convention explicitly.

Use the existing official `HDF5Reader` through `TandemRolloutDataset` and the
reviewed `p026_state_history.py`/`p026_history_inference.py` interfaces. Do not
invent an official API or retrofit the legacy CEM caller. No validation or
frozen-test directory is enumerated or opened.

## Force timing: mandatory evidence boundary

Read-only source inspection of the preserved
`artifacts/causal_force_input_audit_20261005/timestamp_audit.py` shows that
base/train8 force curation can interpolate at float32 field times using a
right bracket after the declared nominal endpoint. The existing audit proves
six specific windows, not all44 origin51 windows. HDF force at an endpoint
therefore cannot automatically be called an online-causal measurement.

The primary diagnostic is explicitly **offline, existing-HDF engineering error
decomposition**. Its targets remain unchanged. Label HDF-based persistence and
the52-point HDF prefix as stored/interpolated references, not verified causal
online histories. No full44 causal proof is claimed or required to reinterpret
these as training targets.

Before any optional claim about an online-compatible cost, separately approve
a small raw-coefficient provenance check for these exact44 trajectories and
frames0:51. Bind case configuration, declared restart/time origin, raw source
files and hashes. For nominal time `T_i=restart_time+0.1*i`, select a unique
exact source row within1e-8 representation tolerance only when documented as
the same nominal endpoint; this tolerance never admits the next solver step.
Otherwise use the latest finite raw row whose timestamp is strictly `<=T_i`
(left hold), recording its time and age, with no right-bracket interpolation.
Reject ambiguous duplicate/restarted rows or missing samples. Restart-source
fallback is allowed only for trajectory frame0 at its exact restart endpoint,
never to fill an interior gap. Do not overwrite HDF targets or refit statistics.
If this optional audit is unavailable, retain the offline-only label; do not
silently fabricate or repeat missing prehistory. Existing six-window sidecars
are not a replacement for this44-trajectory proof.

## Outputs and anti-dilution rule

Retain every origin, action sequence and all10 predicted/target four-force
vectors, with source identities and explicit conditioning labels. Report:

1. Each lead1–10: signed residual, absolute and squared error for all four
   forces and total drag. Report aggregate MAE/RMSE/bias by lead and path.
2. Prediction-only ten-point rear-Cl signed mean error, centered residual MSE,
   predicted and true fluctuation RMS and absolute RMS error. These short-window
   statistics are not one-shedding-period admission quantities.
3. The mixed62 endpoints:52 true stored force vectors at frames0:51, followed
   by predicted52:61, versus the all-true0:61 window. Frame51 appears once.
   Apply the existing canonical statistic implementation with identical sampling,
   weighting and time-window conventions; do not replace it with convenient
   arithmetic if its integration contract differs. Report mean total drag,
   rear-Cl mean and centered RMS, errors, and cost components; any required
   baseline normalization must come from the existing pinned contract.
4. K1/K4/persistence comparisons, separately for H1 and free AR where applicable;
   family, phase, and individual episode/case rows. Report both trajectory-macro
   overall44 summaries and family summaries so base20 cannot hide train16 errors.
   Train16 covers only historical b00/b02 exploratory episodes, not final-policy
   samples or four-phase coverage.

The52 identical reference points can dilute mixed-window error. A good mixed62
cost with poor predicted10 behavior is not success. No new admission threshold,
PASS label, best-model selection or post-hoc horizon/phase filtering is allowed.
Record H1–AR paired differences and full per-case distributions rather than
asserting an arbitrary significant improvement cutoff.

## Interpretation and next-action limits

- Poor H1 and AR: shortening rollout alone does not repair conditional force
  prediction under this representation. Do not start short-horizon control.
- H1 appreciably better than AR: supports a contribution from rolled-state input
  error/distribution shift. It does not identify a unique mechanism or prove
  that direct force modeling is otherwise sufficient.
- Good H10 with prior H100 failure: supports a horizon-sensitivity hypothesis,
  not a matched-origin causal proof of long-horizon accumulation. Existing
  formal H100 origins differ; additional evidence requires a separate design.
- No recorded trajectory supplies alternative-action outcomes at the same state.
  This cannot prove counterfactual action ranking or MPC benefit.

The useful result is selecting the next error source to address, not another
small learning-rate/weight sweep. H10 results cannot override H100 admission or
authorize surrogate PPO. The main route remains accepted surrogate → newly
trained compatible69D PPO → separately verified real-CFD feedback. That deployed
policy does not require a full-field CFD-to-FNO bridge; this design neither
builds one nor diverts the project automatically into MPC.

## Bounded future execution contract

Proposed cap: one sequential read-only job,44 origins, two models, ten leads,
H1 and AR:1760 aerodynamic state evaluations (batching may reduce call count).
At most440 frozen-flow transitions if shared across the two arms, or880 if the
reviewed implementation recomputes and verifies equality. No gradients, backward,
optimizer, model save, candidate export or repeated training. Persistence has
no model calls. Fix the implementation choice before approval; do not silently
double work or infer bitwise equality across changed batching.

Use one-window device staging, bounded CPU buffers, allocator cap0.06 and
container memory12GiB, subject to implementation/resource review. Proposed
whole-job timeout900s with bounded owned-container cleanup; fail closed rather
than increase budget automatically. Require startup MemFree>=30GiB and
MemAvailable>=50GiB, continuous both>=20GiB plus the existing CUDA guard, exact
owned-container/source/image evidence and no concurrent GPU job. If that
budget cannot support the reviewed interfaces, return to Root before execution.
Hash verification/cache handling must be separately authorized and logged;
this design itself authorizes neither scans nor cache maintenance.

Before running, Root must approve immutable source closure,44 train identities,
candidate/protocol hashes, output schema, precision, resource guards and exact
execution command after independent CPU tests of timing and H1/AR isolation.
All results remain diagnostic, not formal admission or control success.
