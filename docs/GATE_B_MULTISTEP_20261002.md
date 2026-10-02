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

A separate batch-one timing run on a real `expanded_test_00` action sequence
measured the official FNO's model-forward latency at 6.31 ms median per step
and 0.639 s for 100 synchronized sequential steps on the primary GB10. The
CUDA peak allocation was 0.528 GiB. These numbers exclude model loading,
OpenFOAM solve time, data transfer and controller optimization; they do not
measure end-to-end closed-loop latency. The record is
`artifacts/monitor/fno_inference_benchmark_seed20261003.json`, produced by
`scripts/run_tandem_fno_inference_benchmark_spark.sh`.

The live dashboard at `scripts/serve_live_research_dashboard.py` shows the
formal curves, this gate, current-model CFD comparison images and both Spark
nodes' CPU/GPU activity. Its monitoring instructions are in
`docs/LIVE_DASHBOARD_20261002.md`.

## Second-seed replication (completed 2026-10-02)

A separate ten-epoch rollout run used seed `20261004` on the same audited v2
training/validation split and the same official PhysicsNeMo FNO. Its best
checkpoint is epoch 10. The worker GPU guard recorded minimum system
`MemAvailable` of 106.64 GiB. The model and logs were copied back to the
primary Spark with SHA-256 verification. All 42 frozen Gate-B checks were
evaluated; the only failure was the unchanged 100-step standard-holdout
total-drag NRMSE threshold:

| Rollout seed | Four-case holdout 100-step NRMSE | Independent phase | Gate |
| --- | ---: | ---: | --- |
| 20261003 | 13.56% | 7.90% | fail |
| 20261004 | 13.56% | 7.49% | fail |

For seed `20261004`, case `expanded_test_04` remains the dominant error at
28.06% (other cases: 7.48%, 11.54%, 7.18%). Thus changing the random seed
did not resolve the strong-rotation coverage problem. Neither seed authorizes
CEM or HydroGym PPO for the current total-drag objective. The canonical
second-seed report is under
`artifacts/distributed_runs/gateb_multistep_seed20261004_20261002/formal/`.
The next frozen test is the augmented v3 dataset with 26 train, 4 validation
and 5 test CFD trajectories, including the untouched `expanded_test_05`.
