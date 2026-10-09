# Gate-B audit: epoch-30 total-drag PhysicsNeMo FNO

Date: 2026-10-02
Decision: `GATE_B_NEEDS_MULTISTEP_RETRAINING`

This audit follows the locked objective in
[`RESEARCH_OBJECTIVE.md`](RESEARCH_OBJECTIVE.md): reduce the total drag of the
two-cylinder system in real CFD while constraining lift fluctuations. It assesses
surrogate fitness only and is not a closed-loop drag-reduction result.

## Reproducibility identity

- Primary node: `spark-269b` (`USER@SPARK_HOST`)
- Official PhysicsNeMo source reference: commit
  `7468c3e4dfbe5cf635ac39f040343e7b3660332a`
- Runtime: NVIDIA PhysicsNeMo `2.2.2`, official FNO API
- Container image ID:
  `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`
- Data profile: `tandem_cylinders_expanded_independent_v2`
- Model outputs: `u`, `v`, `p`, front `Cd/Cl`, rear `Cd/Cl`
- Training seed: `20261002`
- Best model SHA-256:
  `b66f7b030d0c1ea297ec60df9b7d7129a8abeaaa68256b86c74cf7aa46040276`

The 30-epoch run completed after one recorded CUDA-context stall at epoch 25. The
hung process was diagnosed, stopped, and resumed from the immutable epoch-20
checkpoint. No OOM or GPU Xid occurred. The final history contains exactly 30
epochs and the final checkpoint reload passed.

## Final training metrics

| Metric | Epoch 30 |
| --- | ---: |
| Training loss | `4.62052e-05` |
| Validation field MAE | `0.00126526` |
| Validation field RMSE | `0.00212850` |
| Normalized four-force MAE | `0.0330082` |

The memory guard observed at least `108.89 GiB` system/unified memory available
during training, well above the fixed `20 GiB` reserve requirement.

## Autoregressive total-drag evidence

Total-drag NRMSE uses the RMS of the true `Cd_front + Cd_rear` as denominator.
The acceptance ceiling was fixed at 10% for the 100-step full-period proxy before
this audit was run.

| Evaluation | 1 step | 10 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: |
| Held-out observed actions | 3.24% | 4.13% | 11.19% | **16.18%** |
| Independent shedding phase | 0.66% | 1.57% | 6.42% | **8.62%** |

All evaluated segments remained finite through 100 steps. The independent-phase
panel passed the full-period threshold, but the standard held-out panel did not.
The gate therefore fails closed.

## Action-conditioning audit

Observed actions must outperform deliberately incorrect inputs. Total-drag MAE:

| Action supplied to model | 1 step | 10 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: |
| Observed | 0.0559 | 0.0755 | 0.2154 | 0.3304 |
| Zero | 0.1548 | 0.3091 | 0.7873 | 0.8226 |
| Sign flipped | 0.3004 | 0.6250 | 1.2202 | 1.0382 |
| Shuffled | 0.1926 | 0.3360 | 0.8116 | 0.8626 |

Observed actions win at every horizon against all three counterfactuals. This is
evidence that the surrogate uses the rotation input, but it does not override the
failed full-period accuracy threshold.

## Decision and routed next experiment

Only `heldout_full_period_total_drag_nrmse` failed. The next experiment is therefore
10-step rollout-loss fine-tuning of the same official seven-output FNO, initialized
from this epoch-30 checkpoint. It does not change the geometry, objective, force
channels, data provenance, or Gate-B threshold.

The isolated worker `spark-3a22` (`WORKER_HOST`) has the exact same container
image. Thirty train/validation input files and the initial checkpoint were verified
with SHA-256 after transfer. A real 10-step backward-pass preflight succeeded for
batch sizes 1 and 4 while retaining more than 111 GiB available unified memory.

The first preflight attempt failed before training because worker UID 1042 was not
present in the container password database. The runner now supplies `USER` and
`LOGNAME`; the failed attempt is retained under
`artifacts/distributed_runs/gateb_multistep_20261002/preflight/`.

The formal worker configuration is:

- seed `20261003`;
- 10 rollout steps;
- batch size 4;
- 10 epochs;
- deterministic train-window stride 10 over all 24 real CFD trajectories;
- teacher forcing 0.5 to 0 over 5 epochs;
- at least 20 GiB available unified-memory guard.

No PPO or CEM-MPC result will be accepted until the fine-tuned checkpoint is
re-evaluated with the same observed/counterfactual/independent Gate-B suite.

## Canonical artifacts on the primary Spark

```text
artifacts/tandem_fno_total_drag_spark_30epoch/
├── best/FNO.0.30.mdlus
├── training_history.json
├── heldout_evaluation.json
├── heldout_evaluation_zero.json
├── heldout_evaluation_sign_flip.json
├── heldout_evaluation_shuffle.json
├── heldout_evaluation_phase_independent.json
├── gate_b_audit.json
└── gate_b_audit.md
```

Evaluation JSON SHA-256 values are retained with the run record. Large checkpoints,
HDF5 data and generated figures remain outside Git and are reproducible from the
documented commands and manifests.
