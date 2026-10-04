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

## Candidate launcher

`scripts/run_candidate_full40_canonical_ppo.py` consumes this audit and calls
the existing `train_full40_hydrogym_ppo_canonical.py`; the historical shell
entry remains unchanged. The supported host entry is the project-built,
content-pinned image launcher
`scripts/run_candidate_full40_canonical_ppo_spark.sh`, which verifies the
image ID, disables networking, applies container capability and PID limits,
and places execution behind the existing 20-GiB GPU guard. The image contains
pinned official PhysicsNeMo and HydroGym libraries; it is not described as an
officially distributed HydroGym container. Direct-host
`--execute` is rejected. With no `--execute` flag the Python layer only runs both CPU
preflights and writes `CANDIDATE_CANONICAL_PPO_DRY_RUN_READY` or `...BLOCKED`.

Execution additionally requires `--approved-preflight` whose complete nested
readiness and command contract must exactly equal a fresh recomputation. It
always initializes a new SB3 policy for the exact candidate. The candidate
path asks the legacy trainer to save `vecnormalize.pkl` with
`norm_obs=false,norm_reward=false`; this identity wrapper preserves the old PPO
numerics while giving downstream real-CFD evaluation an explicit, hashed
policy/environment pair. The final binding receipt hashes both artifacts and
still states that real-CFD validation is incomplete.

The wrapper fixes the existing contracts at H100, 69 observations,
`|omega|<=0.75`, `|delta omega|<=0.1`, and unchanged `canonical_joint_v1`
reward. Current candidates lack the two standalone canonical evidence gates,
so their real dry-run remains BLOCKED and no PPO process is started.

Focused tests are collected with `PYTHONPATH=src pytest`; the project runtime
image does not itself bundle pytest. The reviewed CPU probe receipt is
`artifacts/hydrogym/candidate_wrapper_lambda10_blocked_review_20261005.json`.
