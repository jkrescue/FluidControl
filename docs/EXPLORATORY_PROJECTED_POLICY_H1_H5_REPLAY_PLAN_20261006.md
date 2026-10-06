# Projected-policy H1–H5 saved-CFD replay plan

Status: CPU/source preparation is authorized. Real VTK conversion, model loading,
GPU inference and any new CFD execution are not authorized by this document.

## Question and fixed comparison

The sealed fixed-action H1–H5 confirmation quantified short-time K1 error, but it
did not cover the changing actions actually realized by the successful projected
32768-policy b00 trajectory. This retrospective diagnostic asks one narrower
question: with the frozen original K1 dual FNO, how accurately are the next one
through five endpoints predicted from saved real-CFD states while consuming the
recorded applied action endpoints?

Those future recorded command endpoints are available only retrospectively. An
online forecaster at the start state would not know the later policy commands;
therefore this measures conditional replay under the realized command sequence,
not an implementable prospective H5 forecast.

No controller, checkpoint, architecture, normalization, threshold or CFD state
is changed. The existing K1 H100 formal FAIL remains. This is not a prospective
shadow controller and does not establish causal action response, because after
the first endpoint the two saved branches occupy different physical states.

## Mechanical sample selection

- Source is the immutable completed FC-E058 b00 pair, result SHA
  `199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6`.
- Bind `progress.json` SHA
  `e6d2555b2907d5844dd27f7cd673403cf57fd086b182182bd0b90234626f69c7`
  and the independently reviewed terminal report.
- Use both `case_mpc` and `case_zero`.
- Predeclare global cycle starts `[0,100,200,300,400,500,600,700]`, corresponding
  to times `[148,158,168,178,188,198,208,218]` D/U. These are mechanical and
  metric-blind.
- For each start and branch, convert exactly the six saved fields at `j..j+5`.
  Total: `8 starts × 2 branches × 6 frames = 96` Curator samples and
  `8 × 2 × 5 = 80` evaluated endpoints. No origin51, sealed frozen-test case or
  new CFD state is used.

Before any model load, validate the predeclared 96 native field/time pins, their
selected U/p file presence, the expected 0.1 grid and the bound progress rows.
The completed-run receipt separately establishes 801 solver states per branch;
this sparse diagnostic does not rescan or claim to reconvert all 801. Saved U
boundary contents are not treated as scalar action evidence.

## State, action, force and time semantics

Reuse `scripts/persistent_curator_frame.py::sample_frame` exactly: official
Curator `VTKSource`, x=8..25/y=4..11/z=.05, finite joint mask, invalid values
zeroed, and the valid-ROI pressure mean subtracted in float64 before FP32 output.
Do not add another centering operation. Validate every returned `TimeValue`
against its selected directory using the established float32-aware tolerance.

For transition `j -> j+1`:

- `omega_j = 0` at global frame zero; otherwise it is
  `progress.rows[j-1].applied_omega`.
- `omega_{j+1} = progress.rows[j].applied_omega`.
- The zero branch uses zero for both action endpoints.
- Never use `requested_omega`, projected raw action, or an interval-held endpoint
  approximation. The real solver used a linear ramp between these endpoints.
- Cast each physical action to FP32 before division by `.75`, matching the
  canonical K1 input builder.

At each start only q_j is truth-conditioned. Recursively feed the predicted
state for leads 2..5 while advancing through the five recorded action endpoint
pairs. Frozen K1 aero reads the current predicted state and current/next action
and predicts the next four coefficients in the fixed order
frontCd/frontCl/rearCd/rearCl.

Truth state is the converted saved q_{j+h}. Truth force is the corresponding
`output_observation[64:68]` in progress. Field persistence holds q_j. Force
persistence separately holds the branch's measured start force: for MPC it is
the start row's `input_observation[64:68]`; for zero it is the shared initial
input only at j=0 and `rows[j-1].zero_observation[64:68]` at later starts.
Zero targets use `rows[j+h-1].zero_observation[64:68]`. Persistence force is
never reconstructed by pooling a truth field or borrowed from the MPC branch.

After conversion, pack each branch/start into one six-frame mini-HDF with the
canonical arrays `state`, `mask`, `omega`, `force`, and `time`. Read it through
the already reviewed official `HDF5Reader` adapter, then use the established K1
rollout packing. This deliberately avoids an NPZ-only custom inference path.
The mini-HDF is an evaluation transport artifact, not a new dataset or split.
Validate that its physical arrays, FP32 actions and four force rows reproduce
the bound converter packets/progress values before inference. All 96 masks and
x/y grids must equal one bound anchor; individual finiteness is insufficient.

## Immutable identities and outputs

Bind before payload access:

- K1 manifest `7adca21e...acc7` and its exact official flow/aero checkpoint pairs;
- original train normalization `f1b4607e...2bc1`;
- configuration `07e55fd...5d9`;
- `persistent_curator_frame.py` `6c1ae12c...7416`;
- the installed official `HDF5Reader` implementation file (not only its package
  version) and the project official-reader adapter;
- `online_current_frame.py` `2945d387...aa4` and history helper
  `2b5b37dc...64c`;
- FC-E058 result/progress/report and a precomputed read-only selected-source
  inventory containing relative path, size and SHA for every copied file.

Load the official K1 checkpoint under its recorded `high`/TF32 identity first,
then apply the reviewed inference-only `highest`/no-TF32 override and record the
effective flags. Train-only normalization remains unchanged.

Copy only the selected time directories plus required `constant/` and `system/`
metadata into a new exclusive output. Mount the original completed case read
only. Never mutate or overwrite FC-E058. Persist `selection.json` before
conversion and `converted_inventory.json` before inference.

Store exactly 16 per-start/branch NPZ artifacts containing the initial field,
all five truths and predictions, mask/grid/time, the six physical action samples,
and force truths/predictions so the aggregate can be recomputed independently.
Store per start/branch/lead field sufficient statistics, four force predictions
and truths, persistence errors, actions, times and identities. Aggregate pooled
physical SSE/reference sums before field relative-L2; report force MAE by
channel, total-Cd MAE and persistence comparisons for H1..H5, plus branch/start
tables. No new pass/fail threshold is introduced.

## Resources, stopping and interpretation

Conversion is CPU-only in the pinned official Curator runtime, one frame at a
guards 50/22 GiB and a 1200-second deadline. Inference uses the pinned official PhysicsNeMo runtime,
frozen/eval models, no gradients/optimizer, a 6 GiB allocator cap and the same
50/22 GiB physical guards with a 600-second deadline. Source and result deadlines are bounded and owned
containers/processes must be inspected and cleaned by exact identity.

Any missing source, SHA/time/mask mismatch, nonfinite value, incomplete 96-frame
inventory, action mismatch, model identity mismatch or resource breach fails
closed and preserves partial evidence. The output is a retrospective diagnostic
only. It cannot tune the opened frozen cases, relax H100 gates or substitute for
new phase/control validation.

## Decision use

Report the full numbers without a categorical accuracy gate. If realized-action
errors are materially larger than the sealed fixed-action panel, that supports
the distribution-gap hypothesis but does not identify whether state, action or
force-readout mismatch caused it. If comparable, these 16 starts do not support
a short realized-action error increase; they do not rule out unsampled
states/actions or longer accumulation. Cross-panel values use different data
and are not presented as a measured improvement.
Either outcome informs one separately reviewed next intervention; neither
authorizes training or CFD automatically.
