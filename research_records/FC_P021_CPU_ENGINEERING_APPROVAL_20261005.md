# FC-P021 causal force conditioning — CPU engineering approval

Root decision, 2026-10-05 UTC. This is implementation and CPU testing authority,
not a GPU experiment, candidate, training run, new metric or scientific acceptance.

## Evidence and hypothesis

P020 is independently complete and rejected for simultaneous local improvement;
its result SHA is a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042.
Preserve that result and the P018 formal rejection. Do not promote P020 or sweep
its loss coefficient. Investigate whether causal current four-force information,
in addition to the unchanged flow/action inputs, helps the official force FNO.
Persistence alone is not a solution: its matched H1 and free-AR waveform errors
are substantially worse than the existing FNO despite some better H1 statistics.

Design reviewed in full by Root and an independent reviewer:
CAUSAL_FORCE_CONDITIONING_DESIGN.md SHA
c2dd3bb59c261a5319fd260c84a382e00deaad812b4f9f19dd6d4c8c5ed63e9e.
The design's outstanding official source prerequisite is now resolved below.
Its original preparation status remains historical; this approval authorizes only
the bounded engineering subset explicitly listed here.

## Official component and initialization

Use the existing immutable official image
sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e.
Root inspected its official FNO2DEncoder without GPU access. File
/usr/local/lib/python3.12/dist-packages/physicsnemo/models/fno/fno.py has SHA
e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9.
It appends two coordinate planes after physical input channels. Therefore an
expanded official aerodynamic FNO has 10 physical inputs and 12 lifting inputs:
copy old physical weight columns 0:6 to 0:6, old coordinate columns 6:8 to 10:12,
and initialize new force columns 6:10 to zero. Copy every other tensor exactly.
Preserve both originally frozen lifting biases. Do not claim bitwise output
equivalence across differently shaped convolution kernels; measure it.

## Authorized work

1. Stage project-owned initialization/input adapters and CPU unit tests; no
   production replacement or checkpoint saving. Official FNO remains the model.
2. Build a pure causal recurrent objective with original J0 weighting. Sequential
   paired H1/AR batch2, H100, full chain-rule differentiation with ten-step
   checkpoint blocks; no detach or measured-force reset at block boundaries.
   A future matched control uses the same expanded model and zero force inputs.
3. Test real/complex toy fullgraph versus checkpoint gradients, a nonzero
   recurrent Jacobian across boundaries, future-truth mutation invariance of AR
   outputs, initial-force sensitivity, normalization/channel indexing and frozen
   tensor protection. Synthetic tensors are unit-test fixtures, not CFD evidence.
4. Resolve raw timestamp provenance before implementing a usable real-data
   conditioning adapter. Preliminary source inspection found float32-time
   interpolation can include the following solver sample in some HDF values.
   Do not excuse that dependency with a wider time tolerance. Use existing raw
   CFD values at the declared physical endpoint, with an explicit unique-row
   match and source hashes; retain old HDF targets/normalization/splits unchanged.
   Root must review the final timestamp audit before this adapter is approved.

Keep all work staged separately on Spark until review. Tests may run on Spark
CPU in the existing isolated environment or CPU-only pinned official container.
No GPU exposure, optimizer experiment, architecture search, heldout access, new
CFD simulation, PPO, production model replacement or admission change is allowed.

## Next authority required

### Timestamp prerequisite resolved for the six-window engineering scope

Root subsequently read the full timestamp report and its producing script.
Report SHA03a728d1f3d3de48c74207f25a0139460cecdf5be674c121ac0fdc9c792f141e,
JSON SHA72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2:
all606 nominal endpoints have a unique same-time raw four-force record. CPU
adapter implementation is therefore authorized only using these exact source
values, with source/content hashes and fixed identity/time validation. No claim
about all44 trajectories or heldout timestamps is authorized. Preserve raw and
curated data unchanged.

The old HDF-lag H1 persistence baseline is NOT strictly online-causal: 200 of505
base/train8 sampled endpoints contain a small next-solver-sample interpolation
contribution. Recompute a separately identified H1 reference from exact raw
current values against unchanged HDF next-step targets. The constant-initial AR
baseline is unaffected because all six initial values were exact. Keep both
receipts and clearly distinguish them; do not rewrite the older analysis.

After CPU implementation and independent review, separately approve one bounded
real-train-window forward/backward resource test, without optimizer or candidate.
Measure full recurrence gradient correctness, latency and peak memory before
approving any 16-update scientific comparison. Both MemAvailable and MemFree must
remain >=20 GiB. Do not silently truncate H100 gradients, alter precision or
relax acceptance criteria if the resource test fails.

Final success remains an accepted surrogate, compatible HydroGym/PPO training and
paired real-CFD closed-loop control, not this engineering work.
