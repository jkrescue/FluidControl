# Candidate-aware PPO readiness adapter

`scripts/audit_candidate_ppo_readiness.py` is a CPU-only bridge between a new
H100 PhysicsNeMo FNO candidate and the existing canonical PPO implementation.
It does not run PPO, create a policy, use a GPU, inspect the frozen split, or
claim physical control success.

The adapter returns `CANDIDATE_PPO_CPU_DRY_RUN_READY` only when one candidate
generation is bound to all of the following independent evidence:

- exact model ZIP payload, training state, resolved H100 config, training
  receipts, train-only manifests and normalization from the candidate lineage;
- official PhysicsNeMo image identity and the existing validation10 endpoint
  readiness gate;
- the historical canonical causal-window fidelity gate;
- the historical canonical dynamic-action gate; and
- the additional development-admission gate and its receipt-bound force-window
  evidence.

The development gate cannot replace either historical canonical gate. Missing
files are reported as `MISSING`, a validly formed failed gate as
`SCIENTIFIC_FAIL`, and inconsistent identities or schemas as `SCHEMA_ERROR`.
Even a READY result records `ppo_execution_authorized=false`: a separate,
reviewed launcher must consume it and train a new policy and VecNormalize for
that exact FNO.

Example (paths are deliberately explicit):

```bash
python scripts/audit_candidate_ppo_readiness.py \
  --candidate-root artifacts/<candidate> \
  --lineage artifacts/<candidate>/posteval/lineage.json \
  --posteval-receipt artifacts/<candidate>/posteval/receipt.json \
  --endpoint-gate artifacts/<candidate>/posteval/validation10/endpoint_gate.json \
  --window-gate artifacts/<candidate>/canonical_window_gate.json \
  --dynamic-gate artifacts/<candidate>/canonical_dynamic_gate.json \
  --development-gate artifacts/<candidate>/posteval/development_gate.json \
  --validation-manifest data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json \
  --normalization data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json \
  --data-artifact dev30=data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json \
  --data-artifact train8=data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json \
  --data-artifact train16=data/curated/tandem_cylinders_directppo_train16_v1/manifest.json \
  --data-artifact paired=artifacts/train20_paired_stat_datapipe_v1/manifest.json \
  --output artifacts/<candidate>/candidate_ppo_readiness.json
```

The validation manifest argument must be the full40 manifest named by the
endpoint gate; it is not the development-only dev30 manifest. Candidate-specific
`--data-artifact` keys must exactly equal the lineage
`data_lineage` keys other than `normalization`; no unrecorded source is silently
accepted. Relative producer/evidence paths are resolved from `--repo`, matching
the existing canonical runner. The command never walks the dataset tree.
