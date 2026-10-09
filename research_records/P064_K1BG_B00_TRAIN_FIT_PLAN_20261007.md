# K1/B/G b00 train-fit diagnostic (preparation only)

Status: source/CPU-test preparation only. GPU execution is not authorized.

This closes one missing diagnostic: terminal K1, B and G aerodynamic branches
are evaluated on the same train-only controlled b00 trajectory used by B/G.
The mechanically fixed origins are `0,100,...,700`; no development result was
used to choose them. At each origin, leads 1--5 are five independent
true-current-state one-step force predictions (`q[s+j-1]`, actual current/next
omega, target `F[s+j]`). They are not a five-step autoregressive rollout.

The implementation reuses the reviewed E100 official loader, K1 history input,
precision/resource enforcement and tensor digest machinery. It reads the b00
train-only view through the reviewed project `TandemRolloutDataset` adapter,
which is backed by the official PhysicsNeMo `DatasetBase` and `HDF5Reader`.
The adapter itself is project code, not an official PhysicsNeMo class. One
reviewed G consumer loads K1, B and G manifests. The diagnostic does not call
the flow FNO or an optimizer. Expected work is exactly 120 aerodynamic forwards:
8 origins x 5 leads x 3 models.

Every model must receive byte-identical normalized states, packed inputs,
targets, actions and clocks for each `(origin, lead)`. The result reports every
row and per-origin/per-lead/pooled MAE, RMSE and signed bias for all four forces,
rear Cl and total Cd. Model tensors must be unchanged.

Interpretation is bounded to training-trajectory instantaneous force fit. It is
not AR, development generalization, scientific admission, checkpoint selection
or evidence that errors decompose additively. If B/G do not improve K1 here,
training-objective/optimization/representation fit remains inadequate. If they
do improve here but not on the fixed development/retention panels, the result
supports a distribution/generalization gap descriptively, not causally.

Fixed resources: 12 GiB memory, no swap, CPU quota 1, TasksMax 64, 6 GiB CUDA
allocator, startup/runtime MemAvailable 50/22 GiB with 20 GiB reserve, 600 s
worker / 630 s outer / 20 s stop. No new training, CFD or data generation.
