# Fixed B/G canonical reproduction profiles

The safe entry remains `scripts/reproduce_canonical_closed_loop.py`. Default is
the existing B canonical b01 controller. G is an explicitly selected exploratory
profile with one independently verified physical condition, not a replacement
for B and not prediction-model admission. G's original selection FAIL remains.

From the original Spark repository, these commands only check metadata, hashes,
available resources and driver schema; they do not deserialize a policy or run CFD:

```sh
.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py
.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py --profile g
```

Both must print `PREFLIGHT_PASS_NOT_RUNNING`. The historical output shown is
evidence, not an instruction to overwrite it. G additionally binds its
[independent physical review](P064_G_SYMMETRY_CANONICAL_B01_CFD_TERMINAL_REVIEW_20261007.md)
and raw result SHA. The G comparison achieved the original six-window physical
criteria, but primary drag reduction3.95135% was slightly below B4.00907%; this
does not establish G superiority or statistically independent generalization.

Actual reproduction still requires a separately approved new JSON, exact SHA,
unique unit/output, and explicit `--execute`. All scientific/resource fields
must match the selected fixed base approval. An old one-run approval is not
reusable authority. Wrong policy/driver/profile, missing or modified G review,
changed resource limits, existing output or existing unit fail closed. No new
CFD run is authorized by this document or by adding the G profile.

Preparation validation: 16 CPU tests passed (9 original B launcher regressions
plus7 profile tests), including default B, unknown profile, mixed policy/profile,
missing/changed G evidence and requirement for new authority. Actual Spark
read-only B and G preflights both passed. Running CFD sources were not modified.
