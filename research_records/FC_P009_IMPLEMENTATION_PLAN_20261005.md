# FC-P009 train-only free-AR force-row calibration implementation

This is an implementation plan and CPU-tested code path, not authorization to
run the GPU extraction, produce a candidate, start formal evaluation, or PPO.

The fixed parent is FC-P003C epoch 2 under its historical default-TF32/high
protocol.  The only proposed model change is replacement of the four existing
force rows and biases in the official FNO final affine layer.  State rows,
hidden layers, architecture, normalization, CFD data, and all scientific gates
remain unchanged.

The design matrix contains the hidden representation actually reached during
each train-only free-autoregressive H100 rollout.  It uses the exact regular
training window enumeration: base20 720, train8 408, and train16 240 windows.
Every one of the 1,368 windows and every one of its 100 relative steps has equal
weight (136,800 rows).  Those rows reuse only 19,648 unique CFD endpoints; they
are distinct model-state exposures, not independent physical samples.

The first approved stage is cache-only: one free-AR extraction followed by CPU
fits. It neither updates nor saves a candidate and therefore avoids the second
15--18 minute full native replay. A complete future candidate plus native replay
would cost about 35--40 minutes and requires separate approval.

The coefficient fit is fixed alpha=0 normalized-force least squares.  Four
physical source phases provide train-only held-phase diagnostics, but no alpha,
mixture, epoch, or model is selected.  The implementation reports parent,
and ideal-affine metrics. It also expands the immutable P008 H1 cache onto the
same `(trajectory, target endpoint)` 136,800-row multiset. The raw HDF endpoint
identity must agree; recomputing the P008 canonical float64-then-float32 label
must match that cache byte-for-byte. The DataPipe's float32 label is checked
against its own torch-FP32 formula, while the small difference between the two
normalization arithmetic paths is reported rather than used as a gate. Both
fits use the same P008 canonical label and fixed-alpha0 phase OOF.
This matched-weight H1 control separates hidden-feature source from the changed
endpoint weighting. For each held physical phase, both the free-AR-fit and
matched-H1-fit alpha-zero heads are scored on both held free-AR and held H1
features (a fixed 2-by-2 matrix), including relative H1/H10/H50/H100 summaries.
These cross-domain reports are diagnostic only and cannot select a parameter.

Before GPU model work, all three train roots must contain byte-identical fixed
normalization and action scale 0.75. For the official CPU loader's first H100
batch in every family, case/start/raw target endpoints must match the immutable
P008 H1 cache identity; the two label arithmetic contracts are checked as
described above. The
parent parameter-and-buffer tensor SHA is required to remain unchanged across
extraction. Validation, frozen data, PPO, new CFD, model mutation, and
checkpoint output are forbidden in this first stage.

The conservative future execution budget is batch 4, allocator fraction 0.15,
continuous unified MemAvailable floor 20 GiB, and a 20--25 minute guarded
window per full extraction pass.  GPU execution requires a separate immutable
launcher and Lead approval binding the final implementation SHA.
