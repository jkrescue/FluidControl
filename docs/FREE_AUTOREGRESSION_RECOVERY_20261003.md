# Tandem-cylinder surrogate recovery and closed-loop plan

Scope: Re=100, L/D=5, two fixed tandem cylinders, rear-cylinder rotation only.
The physical objective remains >=2% mean system drag reduction with rear lift
fluctuation ratio <=1.05 and absolute mean rear lift / zero-control lift
fluctuation RMS <=0.10. A functioning software loop and physical improvement
are separate results, both reported explicitly.

## Evidence at restart (2026-10-03 14:30 UTC)

The qs1 five-epoch FNO finished; no training job remained active. Validation10
H100 pooled total-drag NRMSE was 372.1809 (required <=0.10). Dynamic6 validation
also failed: pooled H100 NRMSE 45660.7643 and strict common-start action-delta
MAE 2688.2259 (required <=0.023). These are dimensionless ratios, not percentages.
Neither result authorizes surrogate PPO or a claim of successful feedback.

Read-only code audit found no state/force off-by-one or normalization mismatch.
The previous five epochs ended with 0.1 truth mixing, never reaching pure free
autoregression. Training and checkpoint selection only covered H20. A bounded
omega-only data audit found action changes in about 0.6% of train transitions
versus 26.33% of dynamic6 transitions. These are plausible independent causes:
long-horizon exposure and dynamic-control data coverage, not a proved exhaustive
root-cause diagnosis.

## Controlled model experiments

Both experiments continue the identical qs1 epoch5 FNO SHA
`a66779c18e4c6c0724f903dd4e767eee643d0867180ad8c6cab95587353537ae`,
using unchanged official FNO architecture, dev30 train20/validation10 and
train-only normalization. Frozen test is not mounted.

| Setting | A | B |
| --- | --- | --- |
| Training rollout | 20 frames | 50 frames |
| Truth mixing | 0 throughout | 0 throughout |
| Epochs / seed / learning rate | 8 / 20261003 / 1e-5 | same |
| Batch | 4 | 4, with isolated probe first |
| Validation each epoch | free H100, stride100 | same |
| GPU allocator fraction | .25 | .45 |

Checkpoint selection uses H100 terminal field MAE + 0.1 times terminal force
MAE, not the shorter training horizon. This is development model selection,
not the physical acceptance test. Existing trainer behavior is preserved when
`validation_rollout_steps` is absent. Nonfinite training loss aborts before
an optimizer update. After training, the runner automatically runs the unchanged
validation10 H1/H10/H50/H100 diagnostic at stride25 and retains its SHA-bound
report. A probe output is never promoted to a production checkpoint.

## Official API provenance

Container image ID is
`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`;
installed `nvidia-physicsnemo==2.2.2` was verified by import/source inspection.
The model is `physicsnemo.models.fno.FNO`; data loading uses official
`physicsnemo.datapipes.DatasetBase`, `DataLoader`, and
`physicsnemo.datapipes.readers.hdf5.HDF5Reader`. The project supplies the
tandem-cylinder sample mapping and training objective through these interfaces.
It does not declare project adapters to be upstream NVIDIA APIs.

Official Curator `Source/Filter/Sink` extensions process real OpenFOAM data.
The installed package is `physicsnemo-curator==0.1.0`. Official references:
https://github.com/NVIDIA/physicsnemo
https://github.com/NVIDIA/physicsnemo-curator
https://docs.nvidia.com/physicsnemo/latest/user-guide/physicsnemo_for_pytorch.html

Installed model alternatives FNO, UNet, recurrent models and DPOT were verified
live. This recovery experiment changes training strategy rather than model
family so the effect can be attributed to autoregressive exposure.

## Data and feedback next steps

In parallel, predeclare and generate eight real dynamic-control CFD trajectories
from train phases b00/b02/b04/b06 using fixed schedules. Process them using the
official Curator into a new train-only profile. Keep b01/b05 dynamic6 for
development validation and b03/b07 frozen. Do not select training waveforms
based on validation outcomes. Use dynamic data if fixed-action free-rollout
training does not repair response to changing actions.

After a surrogate meets the existing long-horizon and action-difference checks,
train HydroGym PPO and test the resulting policy with actual CFD state feedback.
A CFD-corrected short-horizon software demonstration may be investigated
separately, but does not waive the existing model/physical acceptance criteria.
No model-assisted performance claim is made from an inaccurate surrogate.

## Resource and execution policy

All scientific execution remains remote. Spark is the authoritative store;
Worker data is temporary and checksum-verified before/after transfer. Every
CUDA task runs in the existing container with a 5-second host MemAvailable
guard of at least20GiB (DGX Spark unified memory). The two-batch A probe succeeded
with minimum MemAvailable98.5GiB. Training and full evaluation are sequential
on each GPU. New CFD allocation targets <=35GiB raw data; retain >=200GiB free
disk. No Docker/image cleanup or host package changes are required.
# Optional train-only dynamic-data continuation

The rollout trainer now accepts `data.additional_train_roots` and
`training.additional_train_stride`. The default remains the unchanged dev30
dataset. Extra roots are composed using the installed official
`physicsnemo.datapipes.MultiDataset`, not a replacement NVIDIA API.
Only the eight predeclared training-phase trajectories are accepted; each HDF5
SHA, the original normalization bytes, action scale, and split are checked.
Validation and frozen-test trajectories cannot enter this additional source.
The trainer records the manifest hashes and per-source window counts.

Seven focused composition tests and four existing rollout-configuration tests
passed in the pinned PhysicsNeMo container on CPU (2026-10-03). Test fixtures
exercise validation logic only and are not CFD training data. A real HDF5 /
official DataLoader integration check is required after Curator finishes,
before any dynamic-data training. This optional path is not enabled in the
currently running H20/H50 controlled comparisons.

Future runs also report pooled total-drag relative RMS error after each epoch.
The implementation first restores the original force-coefficient scale, sums
front and rear drag, then computes sqrt(sum(error^2)/sum(reference^2)).
Terminal-horizon and all-horizon values are reported separately; neither is an
average of per-case ratios. These diagnostics do not change the loss or existing
checkpoint-selection score. Four focused CPU tests passed in the pinned
PhysicsNeMo container. Already-running comparisons retain their loaded code.

## Interrupted H20 run and official synchronous loading

The H20 process stopped during epoch 6 on 2026-10-03 15:54 UTC. Its five
completed epoch checkpoints are retained. The installed TensorDict dependency
uses a process-global device-transfer recorder; concurrent HDF5Reader calls
raised `Can only mark one TensorDict at a time`. This was not an out-of-memory
failure (minimum host MemAvailable was 72.30 GiB).

The runner now uses the official DataLoader option `prefetch_factor=0`, which
selects synchronous iteration in the installed NVIDIA source. Disabling CUDA
streams alone does not disable threaded host prefetch. No vendor code or API
is replaced. Resume retains the failed log, writes a separate resume record
and log, and restores the completed model/optimizer/scheduler checkpoint.
The restart is not claimed to be bitwise identical to uninterrupted training;
the sampler/RNG sequence is not restored by this project trainer.

## Flow-field accuracy versus force accuracy

Future independent evaluations additionally report pooled relative L2 errors
for u, v and gauge pressure separately, plus the combined velocity-vector L2
error. They use physical-scale fields, exclude solid cells, and pool squared
errors/reference energy across samples and cases before division. A zero
reference energy produces null rather than a misleading zero error. These
are distinct from total-drag coefficient errors and do not replace the
existing control-readiness checks. Four focused tests passed. An actual
H50-epoch-1, validation10, start-zero-only H1 integration probe passed with
all ten outputs finite; that limited probe is not a full accuracy assessment.
The result is retained in `artifacts/field_relative_metric_probe_20261004`.

H20 completed all eight training epochs after recovery. Its first automatic
post-training evaluation hit a read-only visualization output path, not a
model-training error. The failed evaluation log was retained in
`validation10_failed_readonly_visuals_20261004`. The runner now explicitly
places plots inside the writable output mount and supports evaluation-only
recovery without repeating training. Actual H1/H10/H50 image outputs were
verified during the recovered full evaluation; its execution record includes
the runner/evaluator source hashes. No model, data, or evaluation window was
changed by this output-path fix.

## Common B-e5 parent: limited dynamic-action baseline

Before interpreting dynamic-data fine-tuning, the immutable B-e5 parent was
screened on the six dynamic validation cases at start=0 and H100 only. Its
model, normalization and validation-manifest SHA identities were verified.
This is not the full stride-1 protocol and does not qualify the model.
Pooled total-Cd NRMSE was 4.5601%, but velocity-vector relative L2 was 25.3869%
and the four zero-relative action comparisons had delta-Cd MAE 0.114236 with
only 2/4 correct effect signs. In particular, two genuinely drag-reducing
actions were predicted to increase endpoint drag. The separate u/v/p relative
L2 values were 20.14%/71.46%/77.58%. A seemingly small aggregate drag error is
therefore not evidence of reliable flow fields or useful control decisions.
Continue the already declared dynamic-data H50/H100 training; do not promote
this parent into surrogate-only PPO. Raw outputs and the limited-screen
summary are in `artifacts/fno_b5_dynamic6_start0_20261004`.
