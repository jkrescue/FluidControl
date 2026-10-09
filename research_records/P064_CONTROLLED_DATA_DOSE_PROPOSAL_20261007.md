# P064 controlled-data dose proposal (preparation only)

Date: 2026-10-07. Status: **research proposal only; no training, inference, or CFD execution is approved**.

## Question and single factor

The smallest remaining training hypothesis is that P064-B improved one-step force readout because real controlled states were represented in training, but its fixed 25% substitution supplied insufficient temporal density. Test one dose change only: replace 50% rather than 25% of the same 256-window budget with the already converted b00 controlled trajectory. Keep the official FNO architecture, loss, precision, parent, optimizer, trainable parameters, total updates, and all data partitions unchanged.

This is not a new trajectory or independent phase. It densifies one existing b00 trajectory and therefore cannot by itself establish phase generalization.

## Existing evidence

- The independently reviewed fixed b01/b03 development panel reports A to B pooled H1 rear-Cl MAE `.156116880 -> .138998317` (10.9652% lower) and total-Cd MAE `.039952166 -> .038065374` (4.72263% lower). H5 rear-Cl improves 6.85934%, while H5 total-Cd improves only 0.095625%. Source: `docs/P064_AB_DEVELOPMENT_COMPARISON_REVIEW_20261006.md`.
- B is still worse than force persistence at H1: rear-Cl `.138998317` versus `.090333519`, total-Cd `.038065374` versus `.020318218`. Its original-six H1/AR objectives also regress relative to A: `.003437512894 -> .003976855262` and `.008827898137 -> .008946200483`. Thus controlled coverage has a directional signal and a retention tradeoff, not broad accuracy.
- Batch-1 true-state H1 retains a rear-Cl MAE of `.045200950125`; b01-plus and b05-minus retain amplitude errors beyond the prior 2.5% diagnostic scales. Free AR worsens all four rotating-case trailing-62 RMS-error magnitudes, but true-state errors remain. Source: `docs/P064_TEACHER_FORCED_H1_TERMINAL_REVIEW_20261006.md`.
- The H25 flow-only candidate worsens pooled 600-endpoint rear-Cl MAE `.0623860 -> .0830123` and total-Cd MAE `.0207225 -> .0467238`; all six cases worsen in total-Cd and mean/final velocity and pressure errors. Source: `docs/P064_B_H25_QUICK_AR_TERMINAL_REVIEW_20261006.md`. This argues against another horizon or architecture change as the next test.

## Data identities and partitioning

Use the existing train-only inputs; do not regenerate or relabel data:

- P026-K1 parent manifest: `artifacts/fcp026_history_training_k1_20261005/candidate/dual_model_manifest.json`.
- b00 view: `artifacts/b00_controlled_train_dataset_view_20261006`, manifest SHA `97a82e2a157b87ecf9b9ea626ad079852a8d29ba89628ca5034ef99545929a5f`.
- b00 conversion receipt: `artifacts/b00_controlled_train_conversion_20261006/result.json`; converted HDF: `artifacts/b00_controlled_train_conversion_20261006/b00_projected_ppo_train.h5`.
- Original train-only views: `artifacts/p064_host_train_only_views_20261006/{base,train8,train16}`, manifest SHAs respectively `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2`, `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35`, and `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b`.

b00 remains training data and train-fit reporting only. The already opened b01/b03 16-origin panel remains development only. The dynamic-six panel and original-six retention panel remain opened diagnostics. None may be moved into training, used to select a checkpoint, or used to increase the budget after seeing results. No untouched surrogate test is created by this proposal.

## Fixed candidate C protocol

- Fresh load from the same frozen P026-K1 parent; do not continue P064-B optimizer state or weights.
- Train only the same 28 aerodynamic parameters; keep flow and the two lift-bias parameters frozen.
- Fresh AdamW with learning rate `1.5625e-7`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, and the existing gradient clipping and equal H1/AR objective.
- Seed `20261003`; 32 optimizer updates, eight accumulated windows per update, exactly 256 consumed windows total.
- Candidate C substitutes four mechanically fixed positions per eight-window group, `[0, 2, 4, 6]`: 128 b00 windows and 128 original-train windows. Select the 128 b00 starts mechanically and inclusively over valid starts `0..700` using the existing integer `evenly_spaced_indices` rule. No random resampling or added optimization occurs.
- Save and evaluate only the terminal update. Do not select an intermediate checkpoint.

## A/B comparability evidence and caveat

The existing A and B are strictly matched in the reviewed producer, not merely described as matched:

- `scripts/train_fcp064_controlled_aero_ab.py` seeds Python, NumPy, PyTorch, CUDA, and the official data loaders with `20261003` before the arm-specific dataset is composed; it loads the exact P026-K1 parent and creates a fresh AdamW after scheduling.
- `src/fluid_control/p064_controlled_aero_ab.py` compiles both arms from the same verified 1368-window permutation and its identical 256-window prefix. Arm B changes only positions `0` and `4` of each eight-window group, using 64 mechanically spaced b00 starts.
- Independent terminal review verified 32 records x 8 windows, all 28 Adam states at step 32, frozen lift biases equal to K1, and copied flow files byte-equal to the parent (`docs/FC_P064_ARM_B_TERMINAL_ENGINEERING_REVIEW_20261006.md`).

Therefore existing B is a valid equal-budget comparator **only if** a future C implementation is a source-closed schedule-only extension of the same reviewed producer and all parent, seed, optimizer, objective, precision, and data identities remain exact. If implementing C requires any other producer or numerical change, B must be rerun from K1 under that exact producer; otherwise the comparison is confounded.

## Fixed evaluation and interpretation

Retain the existing P064 development protocol rather than inventing a new gate:

1. Primary development comparison: pooled b01/b03 H1 rear-Cl MAE and total-Cd MAE must both move downward from existing B, with every phase and origin still reported.
2. Report H2-H5, field errors, persistence, original-six H1/AR retention, and same-six teacher-forced/free-AR diagnostics exactly as secondary evidence. Do not hide regressions or convert these into post-hoc selection thresholds.
3. Report b00 only as train-fit. A lower b00 error is not generalization.
4. A directional development result is not formal surrogate admission and does not alter the existing H100/formal failure. Any future physical test requires a separately predeclared controller/phase and approval.

## Cost and decision boundary

Use the reviewed P064 A/B envelope: one GPU, 12 GiB cgroup memory, no swap, `MemAvailable` guards 50 GiB at startup and 22 GiB at runtime, preserving the 20 GiB physical reserve. The matched 32-update run is expected to be roughly the same order as prior P064 A/B training (about ten minutes); fixed development replay is minute-scale. These are planning estimates, not authorization.

This candidate is optional and bounded. The 64 existing b00 windows already span `0..700`; doubling their count adds temporal density within the same trajectory, not independent action-phase coverage. If current delivery takes priority, pausing is scientifically reasonable. Await E085/E086 and explicit Lead approval before any implementation or execution.
