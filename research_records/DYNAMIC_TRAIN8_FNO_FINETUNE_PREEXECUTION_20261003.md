# Dynamic train8 FNO fine-tuning: pre-execution status

Status: **not executed**. This runner is a reviewed, dry-run-first development
path. It does not authorize PPO or claim flow-control benefit.

The run preserves the byte-identical train20 normalization from the immutable
`tandem_cylinders_matched_start_full40_dev30_v1` release. It combines base
train20 at stride 20 with the independent train-only dynamic8 trajectories at
stride 2 through PhysicsNeMo `MultiDataset`. The frozen split is neither
mounted nor available.

An explicit completed epoch directory is required. Mutable `best` directories
are rejected; the matching model and optimizer checkpoint are copied into the
new run, SHA-256 checked, and made read-only. H20 uses allocator fraction 0.25
for six epochs; H50 uses 0.45 for four epochs. Both train with teacher forcing
zero and validate at H100. A one-train-batch/one-validation-batch probe must
pass before a full run is reviewed.

After training releases the GPU, the runner evaluates the unchanged
validation10 at H1/H10/H50/H100 and then the independent dynamic6
validation-only suite at the same horizons. The existing H100 persistence and
action-difference gates remain unchanged. Reported force MAE is in physical
coefficient units after inverse scaling, not normalized units. Failure of
either diagnostic keeps PPO blocked; any physical success claim still requires
paired long-window real OpenFOAM CFD.
