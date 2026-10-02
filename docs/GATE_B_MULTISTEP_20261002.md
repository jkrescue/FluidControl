# Gate B: formal multistep surrogate result

Date: 2026-10-02. Scientific objective: reduce mean **total** drag of the
two fixed tandem cylinders by bounded downstream-cylinder rotation, subject to
front/rear lift constraints. The result below assesses surrogate fitness only.

## Training and provenance

- Data: actual OpenFOAM tandem-cylinder trajectories in
  `data/curated/tandem_cylinders_expanded_independent_v2`. The 24 training
  trajectories and validation data were transferred temporarily to the compute
  Spark at `WORKER_HOST`; the canonical dataset and artifacts remain on the
  primary Spark.
- Model: official PhysicsNeMo 2.2.2 seven-output FNO, initialized from the
  30-epoch one-step checkpoint. Four force channels are retained separately.
- Configuration: `conf/tandem_fno_total_drag_rollout.yaml`; ten-step loss,
  batch 4, ten epochs, training stride 10, seed 20261003, teacher forcing
  decreasing from 0.5 to 0 in the first five epochs.
- Formal run: ten of ten epochs completed with exit code 0. Best saved
  checkpoint is epoch 10. The GPU guard sampled memory 679 times; minimum
  system `MemAvailable` was 106.81 GiB and minimum CUDA-free was 56.74 GiB,
  both exceeding the 20 GiB reserve.
- Canonical artifacts: `artifacts/distributed_runs/gateb_multistep_20261002/formal/tandem_fno_total_drag_rollout_seed20261003/`.

## Independent audit

The frozen audit uses real CFD holdout trajectories, observed, zero,
sign-flipped and shuffled actions, plus an independent shedding-phase case.
All rollouts are finite; observed actions outperform the altered-action
counterfactuals and state persistence on the prescribed checks. The decisive
100-step total-drag NRMSE is:

| Model | Standard holdout | Independent phase | Gate limit |
| --- | ---: | ---: | ---: |
| One-step, 30 epochs | 16.18% | 8.62% | 10.00% |
| Multistep, 10 epochs | **13.56%** | **7.90%** | 10.00% |

The multistep model improved both reported cohorts but **Gate B remains
failed**. The standard holdout's case-level 100-step NRMSE is 7.69%, 11.47%,
6.93% and 28.13% for `expanded_test_00`, `_01`, `_02` and `_04`, respectively.
`expanded_test_04` is a distinct `edge_hold` control schedule created during
the train/test split repair; its high error merits an action-history and state
support diagnosis. These per-case values are diagnostic, not a revised gate.

The stage-C CEM job was automatically skipped by the gate. Current-objective
PPO has not started. No surrogate-only result is interpreted as real-CFD drag
reduction. Next work is to locate the error source, add independent real CFD
training cases if coverage is lacking, retrain and repeat the unchanged audit.

The live dashboard at `scripts/serve_live_research_dashboard.py` shows the
formal curves, this gate, current-model CFD comparison images and both Spark
nodes' CPU/GPU activity. Its monitoring instructions are in
`docs/LIVE_DASHBOARD_20261002.md`.
