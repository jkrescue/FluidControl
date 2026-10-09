# FC-P026 HydroGym explicit-history runtime CPU integration

Date: 2026-10-06

Status: Root reviewed the exact canonical production SHAs and independently reran the canonical CPU suites; the worktree is ready for final commit. This is not PPO readiness or scientific admission.

## Scope

The reviewed change touches exactly six production files:

- `src/fluid_control/tandem_hydrogym.py`
- `src/fluid_control/full40_canonical_hydrogym.py`
- `src/fluid_control/dual_control_contract.py`
- `scripts/audit_candidate_ppo_readiness.py`
- `scripts/train_full40_hydrogym_ppo_canonical.py`
- `scripts/run_candidate_full40_canonical_ppo.py`

It adds an explicit, stateless-at-network-boundary P026 K1/K4 history path, per-environment cloned history buffers, candidate-bound snapshot restoration, finite prospective-time and physical-force checks, and approval/terminal-proof binding. Legacy single-model and direct-CFD behavior remain separate. MPC integration remains pending.

## Canonical CPU checks

Fake-HydroGym runtime tests and readiness/legacy tests ran in separate Python processes so cached fake modules could not contaminate an actual-core claim:

- `tests/test_p026_hydrogym_runtime.py`: 15 passed.
- `tests/test_p026_ppo_readiness.py`, `tests/test_candidate_ppo_readiness.py`, and `tests/test_candidate_canonical_ppo_launcher.py`: 60 passed.
- Scoped Ruff (with the existing `E731` compatibility ignore), `py_compile`, and `git diff --check`: passed.

Root independently reran the same two isolated test processes: 15 passed in 1.32 seconds and 60 passed in 1.21 seconds.

The canonical test normalization changed only repository-root/module labels and removed one staged-directory-layout assertion. It did not remove a scientific or runtime behavior assertion.

Exact staged-to-canonical test sources are:

- `/tmp/p026_hydrogym_runtime_stage/impl/tests/test_p026_hydrogym_runtime_staged.py` → `tests/test_p026_hydrogym_runtime.py`: canonical `ROOT`, canonical module labels/comments, and removal of the staged-directory-only MPC/direct-file-absence assertion.
- `/tmp/p026_hydrogym_runtime_stage/impl/tests/test_p026_ppo_readiness_staged.py` → `tests/test_p026_ppo_readiness.py`: docstring and import-module labels only.
- `/tmp/p026_hydrogym_runtime_stage/impl/tests/test_p026_hydrogym_actual_core_staged.py` → `tests/test_p026_hydrogym_actual_core.py`: the three source-location assertions now target the canonical repository rather than the `/tmp` stage.

The staged originals remain outside the repository and were not copied into the commit scope.

## Actual HydroGym-core lifecycle evidence

Retained unit: `p026-hydrogym-actual-core-canonical-20261006.service`

- terminal state: `active (exited)`, `ExecMainStatus=0`, `Result=success`
- resource limits: `MemoryMax=1073741824`, `MemorySwapMax=0`, `CPUQuota=200%`, `TasksMax=64`
- startup memory: `MemFree=23460928 kB`, `MemAvailable=111939400 kB`
- result: one test passed in 0.022 seconds
- source list SHA before/after: `d185a68f1a8a944eed03bbf550a6be57e7ee205c00e50a938eb85a5ce1076131`
- log SHA: `2788255a41662d60310e063573eaeb07a9bca673c8aa2c90a39453c7b07a5aaf`
- immutable evidence directory: `artifacts/fcp026_hydrogym_actual_core_canonical_cpu_20261006/`
- evidence manifest SHA256: `b0ff32a762b13736da9f22a5f2ebeb4b476fc7ac6dc01e5576a94bd29a49674d`

The test imports the pinned real HydroGym `PDEBase`/`FlowEnv` implementation and the canonical project modules, but uses a tiny mock K4 network plus synthetic temporary HDF/raw-force data. It does not use an official model, real CFD, candidate HDF, GPU, PPO, or a policy. Therefore it proves only bounded lifecycle/API compatibility.

## Remaining conditions

No P026 PPO run is authorized by this work. A real terminal candidate, reviewed formal evidence, exact runtime identity, existing PPO image source/API verification, and the unchanged closed-loop acceptance process remain required. The running K1 training and all numerical/scientific gates are unchanged.
