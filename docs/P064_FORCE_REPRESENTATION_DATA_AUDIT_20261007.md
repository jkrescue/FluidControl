# P064 force-representation data audit — 2026-10-07

## Scope

This is a read-only source, schema, and file-inventory audit. It did not run a
recoverability estimator, model inference, training, CFD, or read the live E112
checkpoint. Directory counts below are metadata observations, not a scan or
scientific analysis of every saved field payload.

## Current model input and force readout

The project DataPipe opens exactly `state`, `mask`, `omega`, `force`, and
`time` (`src/fluid_control/tandem_datapipe.py:63-70`). For K1 it constructs six
explicit physical-input channels (`tandem_datapipe.py:77-95`):

1. normalized streamwise velocity `u`;
2. normalized transverse velocity `v`;
3. normalized gauge pressure `p`;
4. the fluid mask;
5. the spatially repeated current angular velocity;
6. the spatially repeated next angular-velocity command.

The three state channels are normalized with train-only statistics and are then
multiplied by the mask. The action is divided by the fixed action scale. The
history adapter independently confirms that no target state or force is an
input and that the official FNO appends its two coordinate features internally
(`scripts/p026_state_history.py:96-111`). Those coordinate features therefore
are not part of the six project channels.

The pinned normalization file identifies the physical state channels as
`u`, `v`, and `gauge_pressure`. Gauge pressure has mean
`3.132473762581896e-15` and standard deviation
`0.11832027697710167`. The normalization is fixed from the 20-case training
split; it is not recomputed here. Relevant immutable files are:

- `data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json`
  (`f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`);
- `.../splits/train.json`
  (`1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89`).

The aerodynamic readout is not a surface-force integral. The FNO produces
spatial output channels, and project code obtains the force prediction by the
mask-weighted spatial mean of channels after the first three
(`scripts/train_tandem_fno.py:52-57`). Training supervises the resulting four
integrated front/rear Cd/Cl values, not a wall-pressure or wall-shear map.

## ROI, grid, and cylinder mask

The release manifest
`data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json`
(`5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2`)
fixes a `256 x 128` grid over `x=[8,25]`, `y=[4,11]`, with 801 frames and
800 adjacent pairs per trajectory.

A metadata-only inspection of
`train/matched_start_acquisition_train_b00_m0375.h5` found `dx` approximately
`0.0666667` and `dy` approximately `0.0551181`. The binary mask contains 428
solid cells and is unchanged between the first and last saved frames. Each
cylinder accounts for 214 masked cells. Their grid-cell bounding boxes are:

- front: `x=9.5333338..10.4666662`, `y=7.031496..7.968504`;
- rear: `x=14.5333338..15.4666662`, `y=7.031496..7.968504`.

These agree with diameter approximately one and centers `(10,7.5)` and
`(15,7.5)`. The HDF5 therefore contains coarse volume samples and a binary
solid mask; it does not contain wall-face normals, face areas, or sampled wall
tractions.

## Retained OpenFOAM force evidence

All 20 cases named by the fixed training split still have corresponding raw
directories under `cfd/tandem_cylinders/cases/`. A directory-only inventory
found 802 saved `uniform/functionObjects/functionObjectProperties` files per
case (initial state plus saved endpoints), one `forceFront` coefficient file,
and one `forceRear` coefficient file. One concrete example is:

- `cfd/tandem_cylinders/cases/matched_start_acquisition_train_b00_m075/`;
- `.../postProcessing/forceFront/148/coefficient.dat`;
- `.../148.1/uniform/functionObjects/functionObjectProperties`.

The per-time `functionObjectProperties` dictionaries retain integrated
`CdPressure`, `CdViscous`, `ClPressure`, and `ClViscous` values separately for
the front and rear cylinders. The HDF5 files retain only the four total
front/rear Cd/Cl labels. The raw case also retains `U`, `p`, and
`constant/polyMesh`, but this audit found no saved `wallShearStress` field or
per-wall-face pressure/shear distribution. The `coefficient.dat` table contains
total coefficients and the OpenFOAM front/rear-half decomposition, not the
pressure/viscous split; that split is available in the per-time dictionaries.

The count of 802 dictionaries for every training case was obtained from path
inventory only. This audit opened one representative dictionary and headers;
it did not parse every dictionary or field payload, and it does not assert that
all time-alignment values have already been scientifically verified.

## Bounded feasibility conclusion

The retained data are sufficient in principle for a CPU-only empirical test of
whether the saved coarse `(u,v,p,mask,omega)` state contains enough information
to recover the aligned integrated total force and its pressure/viscous
components. Such a test would require a preregistered estimator, mechanical
time alignment, and the existing train/development separation. No such test
was run in this audit, so no recoverability result is claimed.

The HDF5 data are not sufficient for an exact surface-traction reconstruction:
they omit wall samples, normals, face areas, and wall shear. Interpolating
near-wall coarse cells onto a cylinder and applying quadrature would be a new
approximate estimator whose error must be measured against the retained
OpenFOAM integrated components. Conversely, recomputing forces from the raw
OpenFOAM mesh and fields would validate OpenFOAM post-processing, not establish
that the `128 x 256` model ROI is information-sufficient.

Accordingly, these facts justify a bounded recoverability diagnostic without
new CFD, but they do not establish that the current representation is the cause
of the remaining prediction error and do not justify adding channels or
changing the architecture by inspection alone.
