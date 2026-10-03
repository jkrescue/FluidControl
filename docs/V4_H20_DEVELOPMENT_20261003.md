# v4 H20 rear-drag FNO development ablation

Date: 2026-10-03 UTC. Script:
`scripts/run_control_gap_v4_h20_development_spark.sh`.

This **validation-only** development experiment uses the existing official
PhysicsNeMo 2.2.2 FNO, not a new model. It initializes from the completed
v3 H20/rear-drag best checkpoint (epoch 9) and fine-tunes on the Curator v4
28-case *training* split. The four validation and five frozen-test HDF5
trajectories are the same as v3. v4 recomputes normalization from 28 training
cases rather than reusing the v3 26-case statistics, so the experiment is a
warm-start adaptation to both new schedules and slightly changed scaling.
It is not an architecture or fixed-normalization ablation.

The predeclared primary configuration is five epochs, 20-step autoregressive
training, batch four, seed 20261003, and rear-Cd channel weight four. It uses
the pinned isolated `fluid-control-physicsnemo:2.2.2` image ID
`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`,
container memory cap 64 GiB and a GPU allocator cap of 15% of unified memory.
The existing GPU supervisor stops work if host `MemAvailable` falls below
20 GiB. The primary Spark remains the durable host; the compute-only Spark
separately runs a v4 ten-epoch one-step FNO.

Before the five-epoch run, a **full 20-step horizon, one-batch** smoke
completed in the same image, loaded the parent checkpoint, produced finite
training/validation metrics and checkpoints, and exited with
`gpu_guard_complete/exit_code=0`. Minimum sampled `MemAvailable` was
99.475 GiB. The smoke is a plumbing/memory test only; its metrics are not
eligible for model selection or physical claims.

The five-epoch run is recorded under
`artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch`, with
systemd log `artifacts/tandem_cylinders/v4_h20_5epoch.log`. Its result must
first be evaluated on the unchanged four validation cases for 100-step
total-drag NRMSE and the predeclared four-action validation CFD ranking panel.
Do not access the frozen test simply because a training loss improved. A
five-epoch warm-start result is not an equal-budget comparison with the
ten-epoch v3 H20/rear-drag model or the separate 30-epoch v3 one-step model.
Any such difference must be labelled as an exploratory observation.

This work does not relax the project's <=10% 100-step accuracy criterion or
the real-CFD total-drag/lift constraints. No surrogate-only PPO or physical
control benefit is implied by completing training.
