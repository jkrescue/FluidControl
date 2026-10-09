# Saved H5 CPU-versus-GPU replay preparation

This engineering replay authorizes no execution yet.  It binds the completed
H5 result SHA `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`
and reuses its ten saved MPC current-field packets, CPU five-candidate H5
predictions, actual endpoint forces, original K1 checkpoint pair, normalization,
baseline and exact 62-sample raw causal history.

The sole engineering change is inference precision/device: one GPU.  The
official K1 identity is first loaded under its historical `high`/TF32-enabled
contract; only after that verified load does the reviewed engineering override
set float32 matmul precision `highest`, CUDA matmul TF32 false and cuDNN TF32
false.  Both before/effective states are recorded and the effective state is
checked before every replay.  For
each saved state, reconstruct history using only the original actual endpoints,
rerun all 5 candidates x 5 stages x 4 forces, then apply the unchanged H5
canonical ledger, costs, feasibility and tie-breaking.  Report every raw force
difference, cost/component differences, full ranks and selected actions.

The predeclared engineering consistency criterion is exactly 10/10 identical
selected indices and actions.  Passing it supports a separately reviewed GPU
transport optimization only; it is not surrogate accuracy, control benefit,
formal admission, PPO evidence or permission to alter horizon/weights.  Failure
is preserved and blocks claiming GPU substitution without further review.

The worker uses the already demonstrated official Curator/PhysicsNeMo runtime,
allocator fraction .06, 12 GiB/no-swap cgroup, host Available 50 GiB startup and
22 GiB runtime, and a 300-second outer deadline.  CUDA free/total bytes remain
observational because this is the reviewed UMA profile; they are not a separate
admission floor.  It creates a new exclusive output and never launches CFD, creates an
optimizer, saves a model, or modifies H5 evidence.
