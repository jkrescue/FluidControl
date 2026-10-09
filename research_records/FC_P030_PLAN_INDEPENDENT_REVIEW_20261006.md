# FC-P030 independent design review

Verdict: the revised design is suitable for separately authorized, isolated CPU implementation and synthetic tests. This is not permission to load real model/HDF payloads or run a GPU diagnostic. P029 formal evaluation and its unchanged admission decision remain separate.

Reviewed plan: `docs/FC_P030_TRAIN_HORIZON_DIAGNOSTIC_PLAN_20261006.md`, SHA `07febacc623df47923db15fc8e560770c7b34f3643aa61a133efd65b022dc0b8`. The initial review found and corrected two material ambiguities: formal at-lead metrics must not be replaced by cumulative-prefix averages, and the existing timestamp check is a quantization-aware whole-grid comparison, not an asserted historical fixed 2e-5 tolerance. The final plan now matches the existing check: compare101 finite, strictly increasing timestamps against t0+0.1*arange(101), using max(2*max(float32 spacing(abs(times))),1e-7), and record the applied tolerance.

## Source and metadata checks

Read current operating instructions and project context, the actual `TandemRolloutDataset` index/load implementation, P029 dataset composition, evaluator field helpers, and small existing manifests/normalization JSON. No HDF contents, checkpoint payloads, model imports, GPU work, cache advice, or implementation execution occurred.

The dataset enumerates `range(0, count-100, stride)`, and the actual P029 composition uses base stride20 and additional-family stride2. Manifest frame counts 801/201/129 make start0 legal for all20/8/16 trajectories, including train16. Each selected window contains q0, targets q1..q100, actions omega0..omega100 and force targets F1..F100. This is the earliest dataset window per file, not its first shuffled encounter. It is not an unbiased sample of the full1368-window distribution.

Actual small-file hashes match the plan: dataset source `c939e455...93ae`; base manifest `5213c7bb...ddd2`; train split `1eaa84f1...96b89`; train8 `a0bd0e3b...53f35`; train16 `7c62dab9...cf5b`; normalization `f1b4607e...92bc1`; objective `904fc903...24516`; runner `e7dd8b9a...c4aa0`. Full digests remain in the reviewed plan. Evaluator source checked: `daef4a3a6eb1b9190f5cde728581b0c1b4b9656f56e865cf61a53deb1f37de88`.

## Required semantics and CPU test coverage

- One uninterrupted free-AR H100 per model: current qhat_j with omega_j/omega_j+1 predicts F_j+1, while the flow advances to qhat_j+1. Future truth is a target only. Poisoned-future-state tests must not change predictions; horizon reporting must not reset rollout state.
- At-lead H1/10/25/50/100 are primary. Optional cumulative-prefix statistics need distinct names. Physical masked field SSE/reference sums must be pooled before square-root ratios; zero reference yields undefined/None, not zero or an invented epsilon. Physical force means/std and four-channel ordering must match the original evaluator, including totalCd formed before absolute error.
- Lead1 force identity is expected because both flows start from identical true q0 and share the same frozen aero. It is not a claim that all teacher-forced or AR forces remain identical. Lead1 fields may differ.
- Synthetic loader tests must cover all family frame counts and original stride membership, exact101 action/100 target indexing, normalization/mask preservation, nonfinite/shape rejection, and the original whole-grid float32 timestamp-quantization check. Actual phase labels must come from the pinned source mapping, not names or absolute timestamps.
- Store per-lead sufficient statistics/raw physical forces so independent recomputation covers all44 rows, groups and sign counts without extra forwards. Endpoint-vs-prefix and mean-of-ratios-vs-pooled-ratio tests are necessary.

The theoretical budget is 8,800 flow plus8,800 aerodynamic evaluations. These are loop-contract counts, not measured profiler counts or evidence of resource feasibility. Use one-window staging and no-grad; avoid retaining all full-resolution100-frame predictions on GPU when sufficient statistics suffice. A separate reviewed launcher/resource approval is required.

## Interpretation limits

Start0 and historical origin51 differ; old H10 values cannot populate this panel or serve as an exact reproduction requirement. Phase/family groups are imbalanced (15/15/7/7 by phase), train16 covers only b00/b02, and only the earliest100 transitions are selected. Full paired deltas and exception counts support descriptive consistency/inconsistency/inconclusive statements, not a new majority gate or causal proof. Better or worse long-horizon training-panel behavior cannot establish held-out generalization, physical control benefit, or PPO admission. No physical10% criterion or existing surrogate gate changes.
