# Dynamic6 raw-force/action time alignment

Read-only Lead diagnostic, 2026-10-05 Asia/Shanghai. No training data changed,
no GPU work, no frozen HDF access. The six development-validation cases are
b01/b05 × minus/zero/plus, 201 frames each.

`scripts/audit_dynamic6_time_alignment.py` verifies each HDF SHA against the
existing dynamic6 manifest, then reads only time, force and omega arrays. It
independently reconstructs front/rear Cd/Cl from raw OpenFOAM coefficient.dat
columns (time, Cd, Cl) and the exact restart row from the source baseline;
omega comes from each case's recorded piecewise-linear action table.

Result: `artifacts/dynamic6_raw_time_alignment_v2_20261005.json` records every
raw/config/HDF/source SHA and per-case differences. At the stored float32 time
values, all four force channels and omega match the reconstruction exactly
for all six cases. The time arrays equal the float32 representation of the
declared 0.1-D/U grid.

Against the ideal float64 grid, maximum rear-Cl difference is
1.2472271919250488e-5; maximum omega difference is 6.109476089477539e-6.
These are effects of reconstructing at the ideal versus stored timestamps,
not a whole-frame temporal shift. They are much smaller than the observed
controlled H1 rear-Cl prediction MAEs of approximately 0.16–0.19.

This check does not establish solver convergence, physical validity of the
mesh, fidelity of every interpolated field pixel, or correctness of a neural
model. It specifically provides no evidence of force/action time misalignment
in this panel. Current FC-P003/P003B training and all acceptance criteria stay
unchanged. Earlier v1 output is retained; v2 adds the stored-time comparison.

Reproduce using the existing isolated CPU curation environment:

```bash
.venv-curator-py312/bin/python scripts/audit_dynamic6_time_alignment.py \
  --repo /workspace/fluid_control \
  --output artifacts/<new-exclusive-output>.json
```

The command refuses to overwrite an existing output. This is a diagnostic,
not a new model-admission gate or a reason to change time precision mid-run.
