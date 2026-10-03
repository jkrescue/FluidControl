# Validation-only FNO control-ranking audit

Date: 2026-10-03 UTC. This diagnostic tests whether the best completed
PhysicsNeMo FNO preserves the *choice* among four predeclared OpenFOAM action
schedules. It does not replace the frozen 100-step Gate B, which this model
still fails (validation total-drag NRMSE 15.522%; five-case test 17.769%).

## Reproducible comparison

`scripts/audit_tandem_validation_action_ranking.py` loads the official
PhysicsNeMo FNO checkpoint from the primary-seed H20 run and the training-set
normalization. Its initial state is frame zero of `expanded_train_24.h5`,
whose metadata identifies the same uncontrolled OpenFOAM `t=80` restart as
the paired validation CFD panel. The four action tables were fixed and CFD
solved before running this ranking audit. FNO is rolled out autoregressively
for 100 actions at `Δt=0.1`; physical front/rear Cd and Cl are decoded using
the training normalization. The comparison uses exact OpenFOAM coefficient
samples at `t=80.1..90`, a startup window within the trained 100-step horizon.
No frozen test trajectory is used for action selection.

| Rear action | CFD mean total Cd | FNO mean total Cd |
| :-- | --: | --: |
| zero | 2.301306 | 2.504595 |
| ramp then +1 | 2.303039 | **2.453619** |
| ramp then -1 | **2.267244** | 2.456766 |
| sine | 2.558163 | 2.707151 |

FNO pairwise ranking accuracy is 4/6. It selects `+1` but OpenFOAM selects
`-1`: CFD regret is 0.035796 mean total Cd (about 1.58% of the true best),
and the selected `+1` is 0.075% worse than zero rotation in this startup
window. The +1/-1 FNO cost margin is just 0.00315, much smaller than its
absolute drag bias (~0.15–0.20). None of the four rollouts exceeded the
training-state-amplitude guard; thus that guard alone cannot detect this
ranking failure. Artifact:
`artifacts/tandem_cylinders/control_landscape_fno_ranking_h20_20261003.json`
(SHA-256 `99577137659ea7acabd117f681b816197bf577b7ac846383414fb9014ce36a13`).

This is **one restart phase and a ten-time-unit startup window**, not the
late `t=120..160` physical objective. It cannot prove that all FNO control
rankings fail, but it is a concrete counterexample to using this checkpoint
for unguarded PPO/MPC. The long-window matched CFD panel still shows 3.49–
4.66% open-loop drag reduction for ±1, with side-load penalties. Startup and
long-run rankings need not coincide. Both must be checked separately.

A completed, independently trained rear-Cd-weighted H10 FNO was checked on
the **same unchanged validation panel** after restoring primary CUDA
headroom. It also ranks only 4/6 action pairs correctly and selects `+1`
instead of CFD's `−1` (artifact
`control_landscape_fno_ranking_rear_weighted_h10_20261003.json`, SHA-256
`622f0e25abf0653f8994156fd45a809accc29722fc8de3f8e8abd16b827f0c4f`).
Its predicted +1/-1 total-Cd costs are `2.403978/2.405374`, a still smaller
margin than H20's `2.453619/2.456766`. Thus simple rear-force reweighting
does not resolve this local control-ordering failure. This is an ablation
diagnostic, not another frozen-test selection or a proof of global failure.

## Decision and next experiment

Do not launch a surrogate-only learned controller on this failing checkpoint.
Keep the frozen Gate B thresholds and held-out split unchanged. The next
model-development gate is *validation control ranking and signed action
sensitivity*, alongside 100-step force accuracy. If the in-progress H20 +
rear-drag-weighted ablation fails to improve validation, do not repeatedly
probe the frozen test; instead inspect train-only action coverage and acquire
new **distinct** OpenFOAM training schedules focused on smooth signed
rotation near the observed ranking ambiguity. Keep these four action traces
and their CFD labels out of training, and keep the independent-phase test
untouched. Compare any uncertainty/disagreement-based acquisition with an
equal CFD-budget random acquisition baseline.

PhysicsNeMo 2.2.2 in the pinned container was checked to contain the official
`physicsnemo.active_learning.protocols` interfaces (`QueryStrategy`,
`LabelStrategy`, `MetrologyStrategy`, `LearnerProtocol`, and `DriverProtocol`).
Its [official active-learning guide](https://docs.nvidia.com/physicsnemo/26.05/user-guide/active_learning.html)
describes train/query/label/evaluate iterations; the
[official surface-CFD example](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/active_learning_aero/README.html)
is an architectural reference, **not** a drop-in tandem-cylinder solution.
An adaptation must use the installed protocol signatures and real OpenFOAM
labels; no invented PhysicsNeMo API or synthetic force labels are acceptable.
