# FC-P030 execution-integration independent review

Status: ACCEPT for bounded execution preparation; this review does not authorize execution or scientific admission. Reviewed 2026-10-06. No model, HDF payload, GPU, Docker or CFD execution was performed by this reviewer.

## Final source identities

| Source | SHA256 |
|---|---|
| diagnose_p030_train_horizon.py | 6b2173b9ca2c9df2c82331eda3e59ca595640491cd2c38cec5066cbca0b7782e |
| run_p030_train_horizon.py | b5e07d162267f709950e58adf02b9e0eb8a68433fc31203319bfa5bc6ded5a3b |
| prepare_p030_train_horizon_spec.py | f8df1e7a2473e4ff7646c8d6ca4cc50fad2a1c165a62f3f98f5cd8cc0000354e |
| materialize_p030_train_horizon_source.py | 2ff0e1422471af77cf5a8835cc1b833a8c999680e266fbe3ec567c08febb1016 |

The actual v2 immutable source manifest is `a14bd2c00ff5c905ccd6193843736a6ccc2806cd2c2d7a51827635cb09fdb6e3`, under `artifacts/fcp030_train_horizon_source_20261006_v2_immutable`. Independently rehashed all 17 members, checked read-only permissions, no symlinks and no extra files beyond the source manifest. The original v1 manifest remains `e4c6c3eb6f31604e55bc17371403cd02fef95c2b438917c61e07a43dbdf4b262`, with its 17 members unchanged.

Pending v2 spec SHA is `991d16b6ad71b406acd34780618621e30d561e12b2e8e6c96fde6ab08380e786`. It remains NOT_APPROVED, execution_authorized=false and lead review=false. Its source/config/provenance/audit/reload small-file hashes were independently verified; this was not a new model/archive/HDF audit.

## Reviewed numerical and operational contracts

- Exact original train families and 1368 H100 sampler members: base 720, train8 408, train16 240. Exactly one start0 origin per each of 44 trajectories; stored actions retain scale 0.75. No validation/frozen split is mounted.
- Official reader supplies all 101 timestamps. Targets are frames 1..100 and actions 0..100. The accepted core enforces the full time grid, mask and normalized state contract. Both arms use the same exact frozen K1 aerodynamic tensors; force is predicted from current state before the flow residual advances it. No future truth enters autoregressive inputs. H1 force identity invokes the accepted core check.
- V2 atomically writes selection.json before the first rollout forward, binds the exact spec SHA, and includes the selection file hash in the final result. The launcher checks that actual file/hash. The small v1-to-v2 diff does not alter scientific arithmetic.
- Source hashes are checked before project imports. Candidate roles/config/normalization remain loader-validated; outputs are exclusive and diagnostic-only. No optimizer or model save is introduced.
- Launcher retains exact-CID create/inspect/start/terminal/cleanup, read-only narrow source/train/candidate mounts, official image, GPU0, 12 GiB container/no extra swap, allocator 0.06, startup MemFree 30 GiB/MemAvailable 50 GiB, and continuous physical/CUDA 20 GiB floors. The three cleanup/recovery helper ASTs match the reviewed H10 launcher. No stale process identifier is treated as container liveness.

## CPU evidence and runtime budget

Independent final command from `/tmp/p030-integration-stage-sota`:

```sh
P030_REVIEWED_LAUNCHER=/workspace/fluid_control/scripts/run_p028_h10_comparison.py CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider tests
```

Result: **23 passed in 0.56 s**. Root separately reported 23 passed in 0.60 s. Tests are synthetic software fixtures, not scientific measurements.

Existing actual P029 H10 Docker lifetime was 67.369 s for 1760 flow forwards and 2640 aerodynamic state evaluations. P030's theoretical 17600 model calls are four times that total; a conservative 6.7-times walltime extrapolation is about 451 s. Thus 900 s is a defensible bounded attempt, not a measured P030 runtime or memory guarantee. H100 processing remains no-gradient and trajectory-streamed; actual resource guards must remain active. No physical mean tolerance, surrogate prediction threshold or formal admission rule changes are authorized. P030 cannot confer H100/PPO/real-CFD admission.
