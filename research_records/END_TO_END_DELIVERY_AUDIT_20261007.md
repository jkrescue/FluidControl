# End-to-end delivery audit — current bounded case

Read-only repository/artifact review, 2026-10-07. No model, CFD or data conversion was executed. Historical approvals are evidence, not permission for a new run.

## Current conclusion through E112

Current execution note: E112 physical-y-reflection training has ended and passed
independent engineering review (256 original windows/512 branches/32 updates).
Both original fixed-six retention objectives worsen B: H1+4.1991%, AR100+1.4529%.
The candidate is not adopted;80-endpoint development evaluation was not run and
its metrics remain unknown. See the [terminal review](P064_Y_REFLECTION_PAIRED_TERMINAL_REVIEW_20261007.md).
No training or CFD currently runs; next data diagnosis is preparation only.
B remains default, prior failures and final acceptance standards are unchanged.
The delivered real-feedback chain below remains usable independently of this
negative prediction study.

The fixed tandem-cylinder operating case has an implemented, independently
audited official-component chain: **existing real curated data and pretrained
K1** → official PhysicsNeMo FNO fine-tuning → HydroGym/SB3 canonical PPO → CPU
policy feedback to real OpenFOAM. This is not a raw-data-only rebuild and does
not deploy online FNO/MPC. E095 reproduces the retained B b01 loop; E109 extends
the same fixed case to a predeclared later interval. Both pass all six unchanged
physical windows, so the basic real closed-loop delivery does not depend on
relaxing the original2% /1.05 /10% criteria.

The former canonical rebuild-documentation gap is closed at the bounded handoff
level by [the multistage runbook](CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md) and
[machine-readable inventory](CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json),
commit `82f5230cf0605f749bdcbb96926002ea571b8a44`. The readonly checker verified906
hashes, including all55 payloads, under invocation
`210169b4918a41d9bb47b02396851a06`. The inventory starts from installed
runtimes, existing curated data and pretrained K1; it does not claim a fresh
raw→upstream-flow→K1 reconstruction, provision payloads from Git or constitute
a new three-stage execution.

E107–E111 add bounded exploratory evidence without changing that default.
E107 absolute64 exactly matches retained B at update32 and continues to64
updates. E108 improves fixed-development H1/H5 summaries but regresses both
fixed-six H1 and AR retention, so prediction selection remains FAIL. E110's
same-protocol exploratory PPO nevertheless produces an E111 b01 physical run
that passes all six original windows; the primary values are4.136238% drag
reduction,0.810196 lift RMS ratio and1.681700% bias, descriptively better than
matched B/G. This single exposed b01 result is not statistical superiority,
independent-condition generalization or full forecasting acceptance, and it
does not silently replace B. Some early/full-window bias comparisons are worse
than B, while mean omega² is0.257480967 versus B0.240447471; omega² is an action
proxy only and is not verified mechanical power or net-energy performance.

Therefore the **basic official-component real-feedback case is delivered**, but
the broader high-accuracy surrogate objective remains incomplete. B formal
prediction and later candidate retention failures stay FAIL; early-window and
seed failures remain part of the record. For the present fixed operating-case
deliverable, no additional architecture, threshold sweep, new Reynolds number
or online MPC is a mandatory missing element. Broader generalization would need
a separately preregistered independent condition rather than selecting among
already opened phases.

Evidence: [E107 engineering review](P064_ABSOLUTE64_TERMINAL_ENGINEERING_REVIEW_20261007.md),
[E108 development review](P064_ABSOLUTE64_DEVELOPMENT_REVIEW_20261007.md),
[E109 future-time B review](P064_B_FUTURE_TIME_CFD_TERMINAL_REVIEW_20261007.md),
[E110 PPO review](P064_ABSOLUTE64_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md)
and [E111 physical review](P064_ABSOLUTE64_B01_CFD_TERMINAL_REVIEW_20261007.md).
All sections below are retained historical audit text; their earlier
"current" or "missing runbook" statements are superseded by this section and
must not be read as live execution status or authorization.

## Historical conclusion before the canonical inventory and E107–E111

The basic official-component → learned-policy → real CFD feedback case exists and has been reproduced. **The current safe entrypoint reproduces CFD from an existing frozen policy; it does not rebuild the FNO and policy from raw data.** A data-to-model-to-policy handoff guide remains the main engineering delivery gap. This is separate from the scientifically unmet prediction criteria; no threshold is relaxed here.

| Segment | Existing evidence / usable entry | Delivery boundary |
|---|---|---|
| Real CFD → curated fields | `scripts/curate_tandem_cfd.py`, full40 Curator orchestration; `docs/B00_CONTROLLED_TRAIN_CONVERSION_TERMINAL_REVIEW_20261006.md` and `docs/P064_B02_CONTROLLED_TRAIN_CONVERSION_TERMINAL_REVIEW_20261007.md` | Official Curator/VTK sampling and HDF production were executed. Their custom case selection, grid, action/force clock and provenance adapters are project code, not an official turnkey tandem-cylinder example. New conversion needs new output and explicit authority. |
| HDF → model input | `src/fluid_control/tandem_datapipe.py`; original dataset manifests and train normalization; b00/b02 train-only views | Official HDF5Reader is wrapped by custom window/action/force alignment. Whole trajectories have split provenance, not random frame splits. Original full40 phase assignment was train0/2/4/6, validation1/5, test3/7; later opened b01/b03 development is not an untouched holdout. The 44-trajectory training extensions and train-only controlled additions must be enumerated, not inferred from a directory glob. |
| Existing K1 → B FNO | `scripts/train_fcp064_controlled_aero_ab.py`; `docs/FC_P064_ARM_B_TRAINING_APPROVAL_20261006.json`; B `training_protocol.json` and engineering review | Actual K1-parent fine-tuning: 256 windows/32 updates,192 original+64 b00; 28 aerodynamic-FNO parameter tensors trainable, two biases and separate flow FNO frozen. Not merely a final-layer readout, not a fresh complete flow-model training run. Official save/reload was verified. |
| Upstream model reconstruction | `scripts/train_tandem_fno.py`; `docs/FC_P026_K1_TERMINAL_REVIEW_20261006.md`; K1 execution approval/candidate protocol | K1 has actual1,368 windows/171 updates and official reload evidence, but itself retains a frozen upstream flow parent. Current quickstart does not give a complete ordered raw-data→upstream-flow→K1→B reconstruction recipe. Existing lineage evidence is not a newly demonstrated data-only rebuild. |
| B FNO → canonical PPO | `docs/P064_B_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json`; E082 terminal review and immutable runner | Actual HydroGym environment interface plus SB3 PPO:32768 transitions,512 optimizer hooks,256 epochs,H5/24 reset states/69 observations, frozen B manifest927669…; canonical symmetry/reward/reset/safety adapters are custom. E082 policy5c056… and Vec1d250… are the retained default pair. |
| Frozen policy → real feedback | `scripts/reproduce_canonical_closed_loop.py`; `docs/CANONICAL_CLOSED_LOOP_QUICKSTART.md`; B/G profile guide | Default B readonly preflight and explicit separately-authorized execution exist. E095 completed800 paired feedback intervals and reproduced E085 action/observation records. G is an explicit exploratory profile, not a replacement default. CPU policy sees real OpenFOAM observations; no online FNO/MPC is required for this basic chain. |
| Result presentation | Existing dashboard `/api/state`, B reproduction curves and real CFD U/p figures; G result/report bindings | Bound invocation/result/report metadata exist. At this audit, G was a current text summary while `#canonical-reproduction` plotted B rows: label/curve separation is a concrete UI fix, not a scientific gap. Instantaneous field pictures cannot establish average drag reduction. |

## Reproducibility scope and dependency identity

The E082 actual approval records PhysicsNeMo2.2.2, torch2.14.1, SB3 2.7.1, Gymnasium1.2.3, NumPy2.5.3 and h5py3.16.0, with source/runtime/import pins. The historical reproduction guide additionally records Curator0.1.0, vendored HydroGym upstream commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`, official Reader source hash, OpenFOAM2512 image and the distinct b40 formal-evaluation image. These are per-execution identities, not interchangeable environments. `pyproject.toml` uses broad lower bounds and is not a complete lockfile or payload installer.

Git alone does not provide the large CFD/HDF/checkpoint payloads, container images or installed runtimes. The current safe launcher checks their existing identities on Spark; it does not provision them. `docs/CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md` provides historical training→PPO→CFD commands from existing data **and K1**, but its commands target the old projected policy and historical outputs. It explicitly lacks a guide-driven fresh three-stage replay. The new canonical quickstart deliberately covers only the last segment. Thus neither guide currently constitutes a complete safe canonical model-and-policy rebuild workflow.

## Current claims and limits

- B canonical's two specified seeds pass the b00 primary window; early bias failures remain, including21.18%. B b01 and its E095 engineering reproduction pass all six original windows. This does not establish arbitrary-seed robustness.
- G b01 independently passes six windows: primary drag reduction3.9513458%, rear-lift RMS ratio0.8163775, bias ratio0.0307169. B matched b01 drag reduction is4.0090689%; G is not superior on drag and remains exploratory. G result `f6319771a93541275f4fe7183d49fba9d607100fcdb7d9bf31b9fc92946594a4`, report `P064_G_SYMMETRY_CANONICAL_B01_CFD_TERMINAL_REVIEW_20261007.md` SHA `1d7979cace631eb02ed5fe2f76f5f00b4ed6eaa3eeee6bb235d3fb4f6b132022`.
- B complete prediction acceptance and G's original retention selection remain FAIL. Old noncanonical seed failures and rejected candidates remain valid records. Physical thresholds remain2% drag/1.05 lift RMS/10% bias. CFD wall time is not physical real-time speed; action-square cost is not verified mechanical energy.

## Smallest useful engineering completion

1. Correct current G/B dashboard curve provenance and historical labels; retain existing service and actual unit/result bindings.
2. Add one canonical **rebuild runbook** with a machine-readable input inventory: raw/curated split manifests, normalization, upstream flow/K1/B parent lineage, exact configs/runtime images, then B training→canonical PPO→CFD handoff. State explicitly whether the starting contract includes pretrained K1. Every stage defaults to preflight and allocates a new output; old approvals must not grant new execution.
3. Validate that runbook's paths/argv/schema/imports using readonly/CPU checks first. A full fresh rebuild would be a separately budgeted scientific execution, not implied by documentation work. Do not delay delivery of the already proven basic case on new architecture/MPC, additional seeds, or automatic retraining.

Evidence links: [safe quickstart](CANONICAL_CLOSED_LOOP_QUICKSTART.md), [B/G profiles](CANONICAL_B_G_REPRODUCTION_PROFILES_20261007.md), [historical multistage guide](CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md), [B full prediction review](P064_B_FORMAL_TERMINAL_REVIEW_20261006.md), [G development selection](P064_AR5_RESET_G_DEVELOPMENT_REVIEW_20261007.md), [G physical review](P064_G_SYMMETRY_CANONICAL_B01_CFD_TERMINAL_REVIEW_20261007.md).
