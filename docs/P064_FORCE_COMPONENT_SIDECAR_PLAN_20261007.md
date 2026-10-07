# P064 train-only force-component sidecars — preparation plan

Status: preparation only; conversion, training, inference, CFD, and evaluation are not authorized.

## Fixed scope

Create one compact label-only HDF5 sidecar for each of the 45 existing training trajectories (20 base, 8 dynamic, 16 direct-PPO, one controlled b00; 20,493 frames). Each sidecar contains the original `time` and `force` bytes as `hdf_total`, plus the aligned physical OpenFOAM `raw_total`, `pressure`, and `viscous` coefficient vectors. `viscous` is parsed from the source dictionary; it is never synthesized as `hdf_total-pressure`.

The original field HDF5 files, total labels, train-only normalization, split, schedule, and data mix remain unchanged. No large HDF5 payload is copied or rehashed; their existing manifest hashes remain the provenance pins. The writer is small project `h5py` code, and every generated sidecar is checked at its first and last row with the official PhysicsNeMo `HDF5Reader`.

## Fixed alignment and evidence

HDF time is matched to the nearest existing `functionObjectProperties` time with absolute tolerance `2e-5`. Parsed `pressure+viscous=raw_total` must hold within `1e-10`. The complete read-only coverage pass found 45/45 cases and 20,493/20,493 frames, with no missing labels. It retains the observed HDF-total/raw-total discrepancy instead of declaring either label wrong; the older curation and controlled-b00 conversion used different recorded pipelines.

The conversion manifest records every selected property path/hash, per-case maximum alignment/component discrepancies, output sidecar hashes, and physical-coefficient population mean/std for pressure and viscous labels for base20, every family, and all training frames. Existing normalization statistics are not refit or overwritten.

## Proposed bounded execution (not yet authorized)

One unique transient systemd unit/output, CPU quota 100%, memory max 8 GiB, swap max 0, CUDA hidden, runtime max 120 s, startup `MemAvailable>=50 GiB`, and per-case runtime `MemAvailable>=22 GiB`. The pending spec and output must be absent before execution. Any mismatch fails closed; there is no automatic retry or case substitution.
