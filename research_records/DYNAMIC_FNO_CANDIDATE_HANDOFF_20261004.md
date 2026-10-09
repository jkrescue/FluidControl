# Dynamic-FNO candidate validation handoff (2026-10-04)

This handoff adds a fail-closed path for the `train20 + dynamic-train8` H50 and
H100 candidates. It does not alter the old train20 evidence, access frozen data,
authorize PPO, or relax the final physical acceptance criteria.

## Evidence layers

1. `audit_dynamic_fno_candidate_lineage.py` recomputes the candidate identity:
   resolved H50/H100 configuration, zero teacher forcing, H100 validation,
   train20/train8 manifests and shared normalization, unique best epoch, model
   and optimizer-state hashes, immutable parent, implementation hashes, and the
   pinned PhysicsNeMo image. For H100, training implementation hashes are read
   from the recorded launch commit. Current recovery/validation hashes are kept
   in a separate field and are never relabelled as historical training code.
   The H50 launch did not record a git source snapshot; its launch-code hashes
   remain null and the receipt states this limitation rather than substituting
   the current worktree.
2. `run_dynamic_fno_formal_validation_spark.sh` evaluates validation10 with
   H1/H10/H50/H100 and runs the existing endpoint Gate against the exact
   candidate. It mounts no frozen-test directory. Dry-run is the default;
   execution requires the explicit approval token.
3. `audit_dynamic_fno_development_gates.py` independently recomputes the new
   2026-10-04 *development admission* from the dynamic6 stepwise force evidence.
   This standard was not part of the original predeclaration and is not a paper
   or final physical acceptance Gate.

The development window uses the final 62 samples on the 0.1-D/U grid. Those
samples span 6.1 D/U and discretely represent the requested trailing 6.15-D/U
continuous interval. Every b01/b05 × minus/zero/plus branch must separately meet:

- total-Cd absolute error ≤ 1% of same-window zero-CFD total Cd;
- rear-Cl fluctuation-RMS absolute error ≤ 2.5% of same-window zero-CFD rear-Cl RMS;
- rear mean-Cl absolute error ≤ 2.5% of same-window zero-CFD rear-Cl RMS.

The dynamic endpoint checks retain pooled total-Cd NRMSE ≤ 0.10,
zero-relative delta-Cd MAE ≤ 0.023, and perfect non-tie sign and cross-action
ordering with fixed numerical tie tolerance `1e-12`. The producer records every
metric, threshold, and gap even when the joint result is FAIL; branch averaging
cannot hide a failure.

## Reviewed sequence

Create the immutable lineage receipt first:

```bash
python3 scripts/audit_dynamic_fno_candidate_lineage.py \
  --candidate-root artifacts/<exact-dynamic-candidate> \
  --output artifacts/tandem_cylinders/dynamic_fno_candidate_lineage_<id>.json
```

Then inspect the candidate-specific command without executing it:

```bash
DYNAMIC_FNO_CANDIDATE=artifacts/<exact-dynamic-candidate> \
DYNAMIC_FNO_LINEAGE=artifacts/tandem_cylinders/dynamic_fno_candidate_lineage_<id>.json \
VALIDATION_OUTPUT=artifacts/tandem_cylinders/dynamic_fno_formal_validation_<id> \
scripts/run_dynamic_fno_formal_validation_spark.sh --dry-run
```

After the separate fixed dynamic6 force-window diagnostic exists, recompute its
development evidence with the exact checkpoint SHA. A later handoff audit may
bind that result using `--development-gate`; a PASS only makes a separate,
reviewed candidate-specific PPO launcher eligible for implementation review.
It still sets `ppo_authorized=false`.

The validated surrogate horizon is exactly 100 control steps (10 D/U), and the
candidate handoff therefore requires surrogate PPO episodes of exactly 100
steps with reset to a real curated frame-0 state. It must not be interpreted as
validated free recurrence for the 800-step/80-D/U real-CFD evaluation. Final
acceptance remains paired real OpenFOAM over the dense final 60 D/U: total drag
reduction ≥2%, rear-Cl′ ratio ≤1.05, and absolute rear mean Cl divided by zero
rear-Cl′ RMS ≤0.10.

Field relative-L2 and force accuracy remain distinct. A small total-Cd error
does not justify a claim of percent-level full-field accuracy.
