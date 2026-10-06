# E089 b02 controlled-field data preparation (not approved for conversion)

This is a prospective, train-only data-engineering plan. It does not authorize
conversion or training and does not read an active CFD output.

## CFD source and labels

The approved E089 acquisition writes a new exclusive source tree
`artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007` from
the fixed OpenFOAM restart at solver time 106. Its controlled branch is
`case_mpc`; its paired zero branch is `case_zero`. The solver advances 800
control intervals of 0.1 D/U from 106 to 186, retaining 801 physical `U` and
`p` states. The underlying solver uses the reviewed
`tandem_backward_dt005` restart, the pinned OpenFOAM image
`sha256:24205c...465fcb`, and the source restart/config hashes recorded in
`docs/P064_B_SYMMETRY_CANONICAL_B02_TRAIN_CFD_APPROVAL_20261007.json`.

The fixed physical case is a two-dimensional laminar tandem-cylinder flow.
The blockMesh domain is x=0..30, y=0..15 and one empty cell through z=0..0.1;
its 21 blocks contain 19,290 cells. The front and rear cylinders have diameter
D=1, centers (10,7.5) and (15,7.5), hence center spacing L/D=5. The inlet is
fixed U=(1,0,0), the outlet has U zero-gradient and p=0, the upper/lower
boundaries are slip for U, both cylinder surfaces are walls, and front/back are
empty. The Newtonian kinematic viscosity is nu=0.01, so U_infinity*D/nu=100.
The solver is OpenFOAM v2512 `pimpleFoam`, laminar, with solver deltaT=0.005.
Force coefficients use rho=1, U_infinity=1, lRef=1 and Aref=0.1 and are written
each solver step for both cylinders. The approved execution pins image ID
`sha256:24205c...465fcb`; it starts from the hashed time-106 restart rather
than regenerating the mesh or initial flow.

The front cylinder remains fixed. For each 0.1 control interval the driver
rewrites only the rear-cylinder wall motion using the previous and newly
filtered endpoint omega, giving a linear ramp over that interval. The paired
zero branch uses the same solver/restart and omega=0. Thus 800 controller
intervals correspond to 16,000 solver steps, while retained U/p endpoints are
106,106.1,...,186 (801 frames). These facts come from the immutable restart,
transport driver and approved E089 metadata, not from inspecting the active
CFD output.

The controlled action is produced by the frozen E082 symmetry-canonical PPO:
one canonical policy call, sign restoration, then the existing physical
amplitude/rate filter once. The future `progress.json` is expected to contain
800 rows with `start_time`, `end_time`, `input_observation`,
`output_observation`, canonical and physical requested-action audit fields, and
the executed `applied_omega`. Conversion uses only the executed action and
actual endpoint four-force values: frame 0 receives the initial observation
force and omega 0; frame j>0 receives row j-1 `output_observation[64:68]` and
`applied_omega`. Requested action is not substituted for applied action.

No hash or completion claim is made before the E089 source is terminal and
independently reviewed. The converter must then bind the actual CFD result,
progress, all 1602 selected `case_mpc/<time>/{U,p}` files, and required
`constant/system` metadata, and verify the source again after conversion.

## Minimal reuse of the reviewed b00 path

The thin b02 adapter is derived from canonical
`scripts/convert_b00_controlled_train.py` (SHA
`f96c882a90e7ecaf4a2f8a5fc327764909ab42b4e6bbb11cde8e99205488b075`).
It retains the reviewed exporter/lifecycle source SHA `304fece8...cad68`,
Curator sampler SHA `6c1ae12c...67416`, official-reader adapter SHA
`68870966...ab1`, and installed PhysicsNeMo `HDF5Reader` source SHA
`cafa65d6...aa0`. Only source identity, the 106-to-186 time grid, output
identity, and reuse count change. Unlike b00, there are no receipt-bound b02
packets, so all 801 frames are newly sampled in 17 batches (16x48 + 33). No
b00 packet is reused.

The output schema remains `state[801,3,128,256]`,
`mask[801,1,128,256]`, `omega[801,1]`, `force[801,4]`, `time[801,1]`, plus
static x/y. It is reread exactly through official `HDF5Reader`. Runtime stays
`.venv-curator-py312`: torch 2.14.1, numpy 2.5.3, pyvista 0.49.0,
physicsnemo-curator 0.1.0, nvidia-physicsnemo 2.2.2.

## Train-only view and split discipline

After a successful independently reviewed conversion, create one exclusive
view with only `train/b02_canonical_ppo_train.h5`, a manifest bound to the
conversion receipt/HDF SHA, and a byte-identical copy of
`data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json`
(SHA `f1b4607e...2bc1`). `normalization_refit` remains false. The view must
contain no `validation`, `test`, or `frozen_test` directory and must not link
b01 or b03 development trajectories.

Validation uses project `TandemRolloutDataset`, backed by official
`physicsnemo.datapipes.readers.HDF5Reader`, with split=train, H100, stride 1,
and all four force channels. An 801-frame HDF must expose 701 starts 0..700;
public `dataset[0]` and `dataset[700]` checks cover the first and last windows.
The existing b00 view was made by a post-terminal audit whose reusable source
was not retained canonically, so b02 needs a tiny reviewed view/audit adapter;
this is a reproducibility repair, not a new conversion pipeline.

## Resource and future command skeleton

Keep the proven conversion envelope: CPU1, 12 GiB MemoryMax, swap 0,
TasksMax64, CUDA hidden, MemAvailable >=50 GiB at startup and >=22 GiB during
the run, 10 GiB output planning budget, 20 GiB free-disk preflight, 3600 s
hard deadline and control-group cleanup. The b00 run took 1240 s with 753 new
frames; 801 new frames should be planned at roughly 22--25 minutes, while the
one-hour cap remains authoritative.

Future command skeleton (not executable until all placeholders are real and a
separate approval is issued):

```text
systemd-run --user --unit=<exclusive-b02-conversion-unit> --remain-after-exit \
  --property=Type=exec --property=WorkingDirectory=<repo> \
  --property=MemoryMax=12G --property=MemorySwapMax=0 \
  --property=CPUQuota=100% --property=TasksMax=64 \
  --property=RuntimeMaxSec=3600 --property=TimeoutStopSec=40 \
  --property=KillMode=control-group --property=OOMPolicy=stop \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=NVIDIA_VISIBLE_DEVICES=void \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  <repo>/.venv-curator-py312/bin/python -u <immutable-b02-converter> \
  --spec <approved-final-spec> --spec-sha256 <real-final-sha> --execute
```
