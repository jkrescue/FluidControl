# FC-P003B — parallel dynamic matched-pair supervision

Lead approval recorded 2026-10-05 Asia/Shanghai, before treatment training.

## Why advance the former fallback now

FC-P001 rejected both frontloaded candidates; FC-P002 shows good zero-action
rollout but poor dynamic phase/sign responses. Independent CPU QC now proves
that eight existing train-only dynamic trajectories can be paired with zero
control from exactly the same five-field OpenFOAM restart. Official DataLoader
reading of all eight pairs passed. The second compute node is available while
FC-P003 trains its static-pair schedule control.

This decision explicitly supersedes the fallback-only waiting condition in
docs/FC-P003_DYNAMIC8_PAIR_CANDIDATE.md. FC-P003 remains unchanged and must finish.
Advance the independent data-content comparison in parallel to avoid serial
training latency, not to fill GPU utilization. No current FC-P003 outcome has
been observed or used to select this intervention.

## Hypothesis and comparison

Repeated dynamic action histories are more useful paired supervision for
controlled flow response than startup-ramp/constant-action pairs. Compare
FC-P003B against FC-P003, not directly against frontloaded lambda10 to claim a
pure data effect. The intended single factor is paired-supervision trajectory
content. Report that eight distinct pairs each occur twice, versus sixteen
distinct static pairs: the effective sample diversity differs and must not be
described as sixteen independent dynamic trajectories.

Keep Main-e2 parent model SHA
8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240,
official PhysicsNeMo2.2.2 FNO/image, regular train20+train8+train16 data and order,
train-only normalization, seed20261003, LR1e-5, H100, zero teacher forcing,
batch1, paired batch1, lambda10 and two epochs unchanged. No new architecture,
new CFD, validation/frozen training data, reward changes or control claim.

Use interleaved positions floor(i*(N-1)/15), i=0..15, N=1368. Consume precisely
two deterministic complete permutations of the eight dynamic pairs per epoch;
record canonical pair IDs and actual positions, proving each pair occurs twice.
Do not silently stop after eight updates. Evaluate train-only paired statistics
over the eight unique pairs; this diagnostic is not checkpoint selection or
admission. The regular validation selection formula remains unchanged.

## Bound inputs and preconditions

Dynamic manifest:
artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json
SHA b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c.
Retain the immutable prior manifests/QC evidence. Bind final adapter/probe code
and receipt hashes in the launch record after independent review. Normalization
SHA remains f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1.

Enforce exact original five-field restart hashes, HDF content hashes, equal
initial forces/masks/time grids and state0 rtol=0, atol=3e-7. These differences
are measured float32 curation differences, not permission to pair different
initial states. The adapter is project code composing official readers, not an
invented official API. Do not overwrite the static16 dataset or running source.

Before full training: final real-data CPU probe, fixed-count/permutation tests,
metadata compatibility, identical regular sampling check, finite-gradient GPU
probe, parent/data/source/config/image/approval SHA binding and independent
review. Use a separate immutable snapshot and new output directory. GPU probe
checkpoints must not initialize the formal treatment.

## Execution contract

Worker unit: fluid-control-fcp003b-dynamic-pairs-20261005.service.
Canonical Main output:
artifacts/tandem_fno_dynamic_paired_interleaved_lambda10_20261005.
Worker uses scoped temporary compute storage; return complete outputs and SHA
receipts to Main. Preserve all failures and do not overwrite earlier runs.
Guard physical unified MemAvailable>=20GiB continuously on the compute node.
Budget: one two-epoch treatment and the unchanged FC-P001 postevaluation suite.
Independent jobs must not compete on the same GPU during training.

Physics/Data owns the dynamic adapter, Worker launch and transfer. Surrogate
owns shared trainer selection/repetition and unchanged numerical training
logic; coordinate exclusive files. Evaluation independently verifies input,
sampling, protocol and result evidence and monitors the actual task. Lead owns
scientific decisions. Do not start full training before the preconditions pass.

## Decision and continued closed-loop objective

Use identical validation10/dynamic6/force-window evaluation, report per-phase,
per-action, per-horizon fields and forces with separate aggregation definitions.
Do not relax gates. If admission passes, bind the candidate to a newly trained
PPO policy and verify paired real-CFD feedback. If not, record the failure and
choose the next evidence-supported intervention. The final original physical
goal remains paired80D CFD/final60D: drag reduction>=2%, rear Cl-prime ratio
<=1.05, normalized mean lift bias<=0.10 with existing action limits.
