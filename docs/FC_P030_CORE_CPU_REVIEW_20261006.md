# FC-P030 independent core CPU review — 2026-10-06

Verdict: ACCEPT for subsequent isolated execution-integration preparation only. No remaining blocker was found in the reviewed pure diagnostic core. This is not authorization for GPU execution, a scientific result, or PPO admission.

## Reviewed identity and independent test

- Plan: `docs/FC_P030_TRAIN_HORIZON_DIAGNOSTIC_PLAN_20261006.md`, SHA256 `07febacc623df47923db15fc8e560770c7b34f3643aa61a133efd65b022dc0b8`.
- Preserved core: `artifacts/fcp030_core_cpu_stage_20261006/scripts/p030_train_horizon_core.py`, SHA256 `90d99902d87fb07494eb87e7e5f91c6420e70c5ee773dd3b55af77e0541a3560`.
- Tests: `artifacts/fcp030_core_cpu_stage_20261006/tests/test_p030_train_horizon_core.py`, SHA256 `1be2a50549df44ae1fd511551e038436374bad6e3cc2ab3d6649d47e42fd9725`.
- Independent rerun from that preserved artifact directory: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider tests/test_p030_train_horizon_core.py`: **15 passed in 0.99 seconds**. Both file hashes were independently recomputed immediately before the run.

## Substantive findings

The final revision resolves the initial review findings: it validates the exact family-by-phase distribution, fixed dataset indices and exact integer fields; binds each paired record to case/start/dataset identity and equal truth/reference arrays; and reports paired sign summaries within groups. Negative fixtures reject reordered cases, altered force truth or reference sums, malformed sufficient-statistic shapes, negative sums and undefined paired velocity ratios.

The core selects 44 start-zero H100 windows from supplied original membership, with base/train8/train16 counts 20/8/16 and phase counts 15/15/7/7. Train16 is restricted to eight b00 and eight b02 trajectories. The continuous recurrence consumes predicted state q_j with actions j and j+1, compares force against target j+1, and advances the flow without truth resets at reporting leads. Synthetic poisoning of future truth leaves predictions unchanged. Each tested window performs 100 flow and 100 aerodynamic calls under no-grad, with eval/frozen-parameter checks.

Primary at-lead H1/10/25/50/100 and separately labeled cumulative-prefix summaries remain distinct. Field errors are physically unnormalized and masked; sufficient squared sums are pooled before taking relative square roots. Zero references remain undefined rather than receiving an epsilon. Total drag is summed across front/rear before absolute error. Physical force normalization is retained. The whole-grid timestamp check follows the specified float32-quantization tolerance. The core provides a separate exact H1-force identity assertion for the matched frozen aerodynamic model.

## Limits and integration obligations

These tests use synthetic tensors, toy model callbacks, and locally implemented versions of the canonical metric callbacks. They do not exercise the official reader, real normalization files, official model loading, GPU memory, or actual 44-file source provenance. Hash-shaped metadata and supplied membership do not independently authenticate their sources.

The future execution layer must bind the actual original sampler/source-phase map, candidate and normalization bytes; supply the real canonical input/predict/field-metric callbacks; obtain all 101 timestamps explicitly (the ordinary dataset sample's initial timestamp alone is insufficient); invoke selection and matched-H1 checks; and protect actual model identities and resource floors. The 8,800 flow plus 8,800 aerodynamic calls across two arms are the planned loop total, not observed real-model execution evidence. No HDF/model payloads were opened and no active formal process was changed during this review.

This diagnostic can describe horizon-dependent paired errors on these 44 training windows. It cannot establish physical closed-loop success, identify a unique cause, select an accepted model, or alter the original physical mean-lift 10% criterion or formal numerical gates.
