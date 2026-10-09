# Physical y-reflection helper CPU review

ACCEPT for isolated helper preparation, not a scientific training result or GPU execution approval. The actual training runner remains WIP and requires a separate integration review.

Reviewed source: `src/fluid_control/p064_y_reflection.py` SHA `3e33164042d71a1ebce76fc493111fb095c9cbc77f2ac890da7d36daaf93b1ea`, byte-identical to the accepted isolated helper. Proposal `docs/P064_Y_REFLECTION_EQUIVARIANCE_AUGMENTATION_PROPOSAL_20261007.md` SHA `0f99d251cee12deb6b4dbf3a99a5c62527725db963d53d7a8f40dfafae1f5640` distinguishes exact physical involution from finite-precision normalization roundtrips.

Independent isolated tests:13 passed in2.70s, no skips, source test SHA `19703f4ee3fa96d2e0489c02e857392ac01943aadbfcb5ef957f67060cbaa05b`. Canonical test `tests/test_p064_y_reflection.py` changes only helper/accumulation import-path fallback, SHA `103bd59360a34066b8cf5c594b362dfdcb4180ee2e9a66b607ca03f8527e31ea`; canonical rerun13 passed in2.82s, no skips. Original isolated files remain unchanged for runner binding.

Command (Spark repository cwd):

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/USER/.local/bin/uv run --offline --python .venv-curator-py312/bin/python --with pytest python -m pytest -q -p no:cacheprovider tests/test_p064_y_reflection.py
```

Verified: physical state/force parity and exact pure-map involution; nonzero-mean denormalize/reflect/renormalize and bounded roundtrip; mask zeros, actions, history and asymmetric-mask rejection; synthetic official-Reader fixture with independent NumPy expected state/target/force; one actual b00 train window through the project DataPipe/official Reader; original eight-window accumulator with non-collinear branch gradients, explicit preclip norm/reference update; preservation of accumulated gradients, hook cleanup after exceptions, graph/large-tensor rejection; actual disabled seam calls only the original branch.

Small Linear fixtures execute CPU forward/backward and optimizer steps. No scientific FNO inference/optimization, candidate load/save, GPU or CFD was performed. Independent frozen-flow construction is presently a callback contract illustrated by the toy branch fixture, not proof that a future scientific runner wires it correctly. That runner must call the same frozen flow independently from each branch's q0/actions, preserve original diagnostics and32-update/256-original/512-transformed budget, and use half gradients before one original divide-by-eight/clip/step. Existing B and all prediction failures remain unchanged.
