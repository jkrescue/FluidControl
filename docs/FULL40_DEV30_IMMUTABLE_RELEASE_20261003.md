# Immutable full40 development release

The dev30 release unblocks official PhysicsNeMo development after train20 and
validation10 are curated while deterministic frozen curation continues on the
Worker. It does not alter the existing full40 finalizer and does not weaken the
frozen-test policy.

## Release contract

`scripts/finalize_matched_start_full40_dev30.py` selects exactly the
predeclared train phases b00/b02/b04/b06 and validation phases b01/b05. It
validates all existing nine-case provenance and each new development case's
RAW receipt, worker manifest hashes, VTK_READY 801-frame manifest, HDF schema,
forces, action, coordinates, and source state. Frozen phases b03/b07 are read
only from the already frozen 40-case predeclaration; no frozen HDF path is
constructed or enumerated.

The publisher copies, rather than hard-links, all 30 HDF files into a temporary
release, computes normalization and state support from the exact train20 list,
writes split manifests with every HDF SHA-256, writes a declaration-only
frozen seal, makes the tree read-only, and atomically renames it to:

`data/curated/tandem_cylinders_matched_start_full40_dev30_v1`

The later full40 release must require byte-identical train/validation case
lists and HDF hashes and must not recompute a different training
normalization.

Dry-run readiness only:

```bash
python3 scripts/finalize_matched_start_full40_dev30.py
python3 scripts/plan_full40_dev30_fno_retrain.py
bash scripts/run_full40_dev30_fno_retrain_spark.sh --dry-run
```

No command above publishes data or trains a model. Publishing requires the
reviewed execution token after all development HDF files pass.

## Frozen-blind training container

The runner mounts only these host paths:

- `scripts`, `src`, and `conf`, read-only;
- the immutable dev30 release at `/workspace/devdata`, read-only;
- one new dedicated output directory at `/workspace/output`, writable;
- for H20 only, the reviewed one-step `best` directory, read-only.

The repository root, CFD cases, staging tree, and any frozen directory are not
mounted. The container has no network, drops capabilities, is read-only apart
from `/tmp` and its output, uses GPU0 only, and runs the existing official
PhysicsNeMo FNO trainers and model configuration. The memory guard requires at
least 20 GiB available before allocation. Any existing output path is refused;
a retry needs a new run ID.

## Reviewed systemd launch commands

These are examples for after both the release and preflight report READY. Do
not run them while the report is BLOCKED.

```bash
systemd-run --user --unit=fluid-control-dev30-onestep-r1 --collect \
  --working-directory=/workspace/fluid_control \
  --property=Type=exec --property=CPUQuota=800% --property=MemoryHigh=56G \
  --property=MemoryMax=64G --property=TasksMax=512 --property=OOMPolicy=stop \
  --property=Nice=5 \
  --setenv=FULL40_DEV30_TRAIN_APPROVAL_TOKEN=EXECUTE_REVIEWED_FULL40_DEV30_FNO_RETRAIN \
  bash scripts/run_full40_dev30_fno_retrain_spark.sh onestep --execute r1
```

After the reviewed one-step service exits successfully:

```bash
systemd-run --user --unit=fluid-control-dev30-h20-r1 --collect \
  --working-directory=/workspace/fluid_control \
  --property=Type=exec --property=CPUQuota=800% --property=MemoryHigh=56G \
  --property=MemoryMax=64G --property=TasksMax=512 --property=OOMPolicy=stop \
  --property=Nice=5 \
  --setenv=FULL40_DEV30_TRAIN_APPROVAL_TOKEN=EXECUTE_REVIEWED_FULL40_DEV30_FNO_RETRAIN \
  --setenv=FULL40_DEV30_ONESTEP_RUN_ID=r1 \
  bash scripts/run_full40_dev30_fno_retrain_spark.sh h20 --execute r1
```

Before either launch, record the READY preflight, image ID, service command,
dev30 manifest SHA, MemAvailable, and output nonexistence. The runner performs
the same checks again. Neither service authorizes frozen-test evaluation.
The development validation result is provisional: after the final full40
release is assembled, the same checkpoint must pass the formal full40 gate
against byte-identical train/validation HDF hashes before any PPO readiness
decision. The dev30 runner does not modify or bypass that formal gate.

Before that formal Gate or any PPO preflight, run the independent promotion
identity audit:

```bash
python3 scripts/verify_dev30_full40_promotion.py \
  --output artifacts/tandem_cylinders/dev30_full40_promotion_<run-id>.json
```

It requires identical train/validation names, phase sets, split-manifest
SHA-256 values, per-file declared and actual HDF SHA-256 values, and an
identical normalization file. The normalization key set, definitions, values,
and byte SHA must all match. A later full40 normalizer that introduces any new
field is deliberately blocked pending explicit scientific review rather than
being silently treated as equivalent. The audit reads frozen seal metadata but
does not open or enumerate frozen HDF files.
