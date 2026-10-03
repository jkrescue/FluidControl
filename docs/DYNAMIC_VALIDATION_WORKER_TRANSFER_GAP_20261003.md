# Dynamic6 Worker execution feasibility decision

## Decision

Worker execution is **not authorized** in the current time window. The
reviewed Spark-local serial runner remains the only executable path for the
six-case time-varying validation panel. No case was staged and no OpenFOAM
solver was started as part of this assessment.

This is not a compute-capability limitation. The Worker can run the pinned
OpenFOAM image, and two 20-D/U cases could in principle run concurrently with
Spark GPU training. The blocker is evidence integrity: the dynamic6 path does
not yet have the full atomic transfer and independent Spark acceptance chain
already used by the full40 acquisition.

## Why a direct copy is unsafe

The existing full40 scheduler implements all of the following as one reviewed
contract:

1. SHA verification of every Worker-side runner dependency and the pinned
   OpenFOAM image before staging;
2. Spark generation followed by transfer into a unique Worker temporary
   directory, complete tree-hash comparison, and atomic rename;
3. a bounded scheduler that counts live solver processes, enforces resource
   floors, detects orphan/dirty locks, and starts no more than the authorized
   parallel count;
4. an exclusive Worker completion marker plus a per-file raw manifest;
5. transfer into a distinct Spark staging directory, full returned-tree SHA
   comparison, and refusal to overwrite a differing Spark skeleton;
6. independent Spark-side numerical/source/action QC followed by an exclusive
   `RAW_TRANSFER_VERIFIED` receipt and aggregate panel QC.

The current dynamic runner intentionally provides none of the Worker transfer
semantics. Merely allowing a second repository path or using `rsync` would
make it possible to accept a partial result, mix a stale skeleton with new
solver files, overwrite provenance, or lose the distinction between solver
completion and Spark acceptance.

## Minimum contract for a later Worker implementation

A future implementation must be independent of the running full40 scheduler
and must satisfy all of these gates before execution is enabled:

- bind the exact dynamic predeclaration SHA
  `0478c8532bd2ded504ccd5f89303001eb8359f69b3036f296e31428a085d1272`;
- bind the six case names, b01/b05 five-file source-state hashes, runner and QC
  script hashes, and pinned OpenFOAM digest;
- stage each Spark-generated skeleton through a unique Worker temporary path,
  verify its full tree hash, then atomically rename without overwrite;
- use an independent lock namespace, a hard maximum of two live cases, at
  least 40 GiB Worker `MemAvailable`, a disk floor, and explicit rejection of
  orphan/dirty states and unrelated CFD/Curator work;
- write an exclusive Worker completion marker and raw manifest containing all
  solver fields, force coefficients, logs, QC, config and provenance files;
- return into Spark `.staging`, verify every manifest entry before merging,
  and reject both missing files and conflicting existing files;
- rerun the dynamic panel audit on Spark and create an exclusive per-case
  transfer receipt; solver `End` alone must never count as acceptance;
- require all six receipts before aggregate QC and before any FNO dynamic
  validation report is produced.

Until that implementation and its failure-path tests are reviewed, dynamic6
must not run on a Worker temporary copy. The conservative cost remains the
Spark serial estimate of 36–48 minutes after current Curator/training work is
idle. This preserves scientific traceability and does not weaken any model or
PPO admission threshold.
