# Dynamic FNO to canonical PPO handoff gaps

Control/Evaluation review, 2026-10-05. This is an implementation-readiness
note. It does not promote a checkpoint, change a threshold, authorize PPO,
open frozen data, or supersede the historical audit in
`docs/DYNAMIC_FNO_FORMAL_HANDOFF_AUDIT_20261004.md`.

## Four separately consumed surrogate evidence contracts

The existing canonical PPO preflight requires three checkpoint-bound receipts.
They are separate fail-closed interface contracts, but source/history review
after this note was first written found that receipts 2 and 3 had only consumer
schemas and boolean fixtures: no historical numerical producer or surrogate-
error tolerance had been defined. They must not be described as historically
quantified independent scientific experiments:

1. `FULL40_VALIDATION_SURROGATE_READINESS_PASS`: validation10 endpoint and
   H100 force/action-difference readiness produced by the existing full40 gate
   audit.
2. `FULL40_VALIDATION_CANONICAL_WINDOW_FIDELITY_PASS`: the canonical causal
   6.15-D/U window receipt, including total-drag, rear-Cl fluctuation and
   rear-Cl mean-bias fidelity. Its producer and evidence must be SHA-bound.
3. `FULL40_VALIDATION_DYNAMIC_ACTION_PASS`: the canonical dynamic-action
   receipt for the existing `|omega|<=0.75`, `|delta omega|<=0.1` and at least
   H100 contract. Its producer and evidence must also be SHA-bound.

The newer `development_gate.json` is a fourth, additional development-admission
contract. It combines the fixed Dynamic6 start0 endpoint response with a
six-branch sampled force-window check. It has a different status, evidence
schema, aggregation and stated scientific scope. In particular, its pooled
dynamic NRMSE uses six start0 terminal points, while the D012 canonical dynamic
compatibility check uses all rolling H100 segments. The two quantities are not
interchangeable.

Lead decision D012, recorded before FC-P003/P003B post-evaluation completed,
prospectively fills the missing numerical producer definitions using the
already fixed 2026-10-04 development tolerances. A producer must recompute the
numbers from SHA-verified evaluation/force evidence and record the D012 source
and adoption date. It may share evidence with the development gate, but may not
rename that gate, copy its PASS booleans, or claim the definitions were part of
the historical predeclaration. A future candidate must pass both the D012
compatibility receipts and the unchanged development admission.

## Minimum candidate-aware adapter still needed

Use new files and keep the old H20 entry intact. A CPU-only dry-run adapter
must fail closed unless all of the following hold:

- one immutable candidate generation contains an exact PhysicsNeMo model and
  matching checkpoint state; their SHA-256 values, epoch and predeclared
  selection rule are bound to the complete training receipt;
- resolved H100 config and its parent configs, source snapshot, official image,
  dev30/train8/train16 or other approved train-only manifests, fixed
  normalization and frozen-access flags are content-verified;
- validation10 evaluation, segments, endpoint receipt, canonical-window
  receipt, canonical-dynamic receipt and development receipt all bind the same
  checkpoint, normalization, validation split and immutable producer code;
- the adapter reconstructs the existing seven-output FNO from the candidate's
  resolved config rather than the old wrapper's hard-coded H20 config;
- full40 promotion identity, validation b01/b05 zero baselines, the 69D
  observation contract, four-force order, `omega` scale and rate limit are
  recomputed without enumerating or opening frozen HDF; and
- dry-run output distinguishes missing evidence, scientific failure and an
  operational/schema error. It writes no PPO policy and performs no GPU work.

This gap note itself introduces no threshold. D012 now supplies the prospective
numerical completion: per-branch 1% zero-Cd-scaled drag error and 2.5% zero-
Cl-prime-scaled lift RMS/mean errors for the sampled window; and, for canonical
dynamic compatibility, all-rolling H100 total-Cd NRMSE <=0.10, start0 action-
minus-zero Cd MAE <=0.023, plus explicit action/rate/H100/split verification.
The separate development dynamic check retains its six-start0-point NRMSE and
perfect non-tie sign/order requirements. Candidate-specific producers may
reuse the same evidence only through audited, checkpoint-bound recomputation.

## Existing code that can be retained

`Full40CanonicalSurrogateFlow`, `TandemFNOStepper` and the HydroGym `FlowEnv`
wrapper already enforce the 69D observation, four force channels, bounded rear
rotation, rate limiting and component-wise canonical reward ledger. The SB3
training loop already creates a new policy and reports surrogate-only scope.
These interfaces do not need a new model architecture.

The runner still needs a candidate-aware wrapper and immutable receipt chain;
the old `run_full40_canonical_ppo_spark.sh` hard-codes an H20 config and legacy
checkpoint location. A newly admitted FNO must train a new PPO policy and a new
VecNormalize artifact. No policy trained against an earlier FNO may be reused.

## Real-CFD handoff after surrogate PPO

Surrogate admission and PPO completion are not physical success. The existing
OpenFOAM feedback framework can be adapted only after it verifies the new FNO,
policy and VecNormalize SHA chain, observation/action parity and the unchanged
same-start zero-control comparison. Final evidence remains paired real CFD for
80 D/U with the last 60 D/U reported against the original physical criteria:
total-drag reduction at least 2%, rear-cylinder Cl-prime ratio at most 1.05,
and normalized absolute mean rear lift at most 0.10. The CFD-only PPO result is
preserved as a separate baseline and is not relabelled as FNO-assisted.
