# P030 isolated CPU implementation approval

Lead approval, 2026-10-06, after reading final plan SHA
`07febacc623df47923db15fc8e560770c7b34f3643aa61a133efd65b022dc0b8`
and independent design review SHA
`237d88ae7709b82bbe4a9a6d547f5447a0b05fcec17726f9963fb192f62555ff`.

Authorize isolated project diagnostic code and synthetic CPU tests implementing
the fixed 44-start0, two-flow, recorded-action H100 comparison. Preserve official
FNO/DataPipe interfaces, original normalization, masks, force timing and precision.
Report at-lead field/force errors separately from cumulative-prefix statistics.
Reuse canonical field-error aggregation and existing inference helpers; do not
change a running evaluator or duplicate existing model architecture.

Implementation owner: Surrogate agent (sota_methods). Independent code/testing
owner: recovery_evaluation_review. Stage under a unique project artifact directory
on Spark; use apply_patch for edits. Synthetic fixtures must remain explicitly
engineering data, outside real CFD datasets and scientific results. Hide CUDA
for tests; use existing isolated CPU environment, without installing packages.

No real HDF/model payload reads, GPU forwards, optimizer, checkpoint, additional
CFD, validation/frozen data, controller training or scientific admission is
authorized here. Prepare execution integration only after core CPU tests and
independent review. Actual data/model access and numerical execution require a
separate bound approval after the currently running P029 formal job terminates.
No acceptance threshold or final closed-loop objective changes.
