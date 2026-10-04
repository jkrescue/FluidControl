# FC-P003C fixed-feature force-readout diagnostic

This bounded train-only diagnostic asks whether the existing C epoch-2 FNO hidden representation can express the four forces through its existing affine output projection. It is not a new model, a candidate, or an admission result.

The official model is held in evaluation mode and is never mutated or saved. A pre-hook reads the input of `decoder_net.final_layer.linear`, whose verified shape is 128 to 7 with an identity final activation. The same fluid mask used by the project prediction function spatially averages the 128 hidden features. Before the first forward, this isolated diagnostic disables CUDA/CUDNN TF32 and selects PyTorch `highest` float32 matmul precision. Both the prior and effective settings are recorded. This deliberately differs from the historical C training/formal-evaluation default numerical protocol and no bitwise equivalence to those predictions is claimed. Under the fixed diagnostic protocol, rows 3:7 of the original affine layer must reproduce the model's four normalized force outputs within `2e-5`; this is a numerical wiring check, not a scientific threshold.

Eight dynamic train actions and four unique same-phase zero trajectories are extracted once over targets 1..200. A float64 affine least-squares readout (`numpy.linalg.lstsq`, fixed `rcond=1e-10`) is fit only on targets 1..100. For exact symmetric pair weighting, each action contributes 100 rows and its phase zero contributes another 100 rows, so 1,200 unique rows become 1,600 weighted rows: 800 action and 800 repeated-zero rows. Targets 101..200 are a strictly later-time check. Metrics remain separate for 800 actions, 400 unique zeros, and 800 action-minus-zero endpoints, with all four channels and every pair reported. Rank, all 129 singular values, the `rcond*s_max` cutoff, retained-singular-value condition number, raw smallest singular value, and coefficient norm are recorded.

Interpretation is limited:

- large prefix residual means the fixed masked-mean features are not linearly sufficient for this force target;
- low prefix but high late error indicates a time/feature-distribution transfer problem;
- low errors in both windows would show only that the fixed features admit a better train-only linear readout than the current rows. It would not establish that output-row optimization is the primary or unique cause.

The late window was included in regular training and is neither independent generalization nor validation evidence. The fit is performed in normalized-force space and metrics are converted back to physical coefficients. A fixed positive per-channel weighting would not change four independent least-squares solutions; no validation-derived weights are introduced.

None of these outcomes changes the canonical thresholds or establishes field, rollout, control, or PPO readiness. The result explicitly records zero optimizer steps, no candidate or deployable checkpoint save, no validation/frozen access, no PPO, unchanged model tensor SHA, and a same-batch repeated-forward state-output identity check. Fitted diagnostic coefficients are retained only for reproducible offline analysis and are not installed in the checkpoint.

After Lead review, the intended single-GPU command is run inside the pinned PhysicsNeMo image under the existing continuous 20 GiB guard (allocator 0.15), with only train leaves mounted:

```bash
python /workspace/scripts/diagnose_fcp003c_fixed_feature_force_readout.py \
  --train8 /workspace/train8 \
  --base /workspace/base \
  --pair-manifest /workspace/evidence/dynamic_pair_manifest.json \
  --normalization /workspace/base/normalization.json \
  --config /workspace/config/resolved_config.yaml \
  --checkpoint-dir /workspace/checkpoint \
  --output-dir /workspace/output \
  --chunk-size 10
```

Expected GPU extraction is roughly 2–4 minutes; the float64 CPU solve is sub-second. The compressed cache contains 2,400 unique 128-dimensional feature rows plus targets/predictions and is expected to be only a few MiB. This plan does not authorize execution.
