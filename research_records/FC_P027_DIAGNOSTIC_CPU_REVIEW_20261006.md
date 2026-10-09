# FC-P027 independent diagnostic CPU review — 2026-10-06

Verdict: no remaining blocker found for the reviewed diagnostic's CPU engineering preparation. This is not execution approval, a real-model result, surrogate admission, or permission to start PPO. Resource launcher review and final immutable dependency closure remain separate prerequisites.

## Reviewed identities and checks

- Staged diagnostic: `/tmp/p027-review.EsQQrA/scripts/diagnose_p027_short_horizon_errors.py`, SHA256 `f398c86f66cb2f30df9b1832b3081d851bac8fd0f9dad3a3d150cc9d25ce99d2`.
- Tests: `tests/test_p027_short_horizon_errors.py` in that stage, SHA256 `6d0788842371d1f786a1ce9ee8ca7f03c50c8a619f72750ca42b562db713979c`.
- Independent CUDA-hidden CPU run: **22 passed in 0.56 s**, using the existing env_isaaclab Python, bytecode/cache writing disabled, staged scripts plus canonical scripts/src on PYTHONPATH.
- Read the complete diagnostic and tests, including the execution path. No diagnostic execute call, official model load, actual HDF reader invocation, HDF byte scan, Docker operation, or GPU computation was performed in this review.

The existing source-phase map was independently SHA-checked as `57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92`. A small-JSON-only check called the actual mapping and baseline validators with the preparation spec: exactly 44 unique origins; base20 has 5 per canonical phase b00/b02/b04/b06, train8 has 2 per phase, and train16 has 8 each in b00/b02. All phases have the pinned existing train20 zero reference. The 44 reported HDF hashes in the spec equal the existing pinned audit's map; this check did not rehash HDF files. The spec remains PREPARATION_ONLY, not an approval.

## Numerical and temporal review

The ten targets are frames 52–61. H1 uses true field history ending at each preceding frame; free AR initializes at frame 51 and advances only through the shared frozen-flow predictions. K4 starts with fields/actions 48–51 and uses the next stored action for each transition. Aerodynamic field outputs do not drive recurrence. Synthetic future-truth poisoning changes H1 but leaves AR unchanged. Reader fixtures check all 62 requested indices, retained 14 field frames, normalization, force target alignment, and the 0.1 time grid with an explicit float32 timestamp quantization allowance.

Per-lead four-force and total-drag errors, mean-case versus pooled RMSE, paired AR-minus-H1 summaries, family/canonical-phase groups, and raw per-case predictions are separate. The mixed window is exactly 52 stored truth points plus ten predicted points; its errors cannot replace prediction-only ten-point errors. No physical gate boolean is exported as admission. The endpoint cost uses actual omega at frame 61 and its difference from frame 60; invalid action/slew retains the case and reports cost unavailable rather than clamping, substituting zero, or removing its force denominator. It makes no claim that all ten actions are admissible.

The final revision resolves the earlier wrong configuration-loader import and filename/local-phase inference. The pinned map, exact 44-member set, three manifest identities, normalization and baseline metadata are validated before model/HDF loading. Baseline selection uses canonical physical phase, not an unverified family-local label. The zero reference is explicitly the existing long-term final-60 reference, not a fitted origin-51 reference. Stored interpolated HDF forces and prescribed actions remain offline descriptive evidence, not a newly proven online-causal input stream.

## Limits and execution prerequisites

Tests use synthetic tensors, a mock reader and mock models; they do not establish real official-reader/model execution, GPU peak memory, runtime completion within 900 seconds, or actual 440/1760 call counts. Those counts describe the reviewed fixed loops and must be checked against a separately approved actual run. The eight-file preparation source list is a draft, not a demonstrated complete import closure; the execution approval must bind the final immutable closure, actual image, candidates, data and launcher. The launcher creation-timeout cleanup issue is handled in a separate review and is not approved by these diagnostic tests.

No models, targets, normalization, datasets, scientific thresholds or canonical production code were changed by this review. K1/K4 remain scientifically rejected under the completed original formal protocol; the physical mean-lift 10% requirement is unchanged.
