# FC-P030 v3 recovery preparation review

Independent verdict: ACCEPT for the narrow interface repair and bounded recovery preparation. No GPU rerun was performed or authorized by this reviewer. The first attempt remains an operational failure with no recoverable scientific result; see `FC_P030_TERMINAL_REVIEW_20261006.md`.

## Exact reviewed versions

- Driver: `a874ad0860f95361861a7f594e3f7044dc8d3c2a0112db320eae83e68051d9e8`.
- Launcher: `d0bb73b2e722d78bb6ac31b85b3e65891b2d0addc2f8eb4ad8f5e04dbb6d9c78`.
- Driver test: `b1efef065daab0bfd66eb540b0cebd50c5eafed9a038aa49f32cacd2ea6a303e`.
- Actual v3 source manifest: `73ac42127e0ace741c675cb7a5a53a6339171565462d0771c11665124d0f08e6`, under `artifacts/fcp030_train_horizon_source_20261006_v3_immutable`.
- Pending v3 spec: `91d0af5324b8cf571a25b4cf9e94649cd9e4cf40bedd1187e22ca9441441907c`; execution_authorized remains false.

Independently rehashed all 17 v3 source members and checked member read-only permissions/no symlinks. The v2-to-v3 production diff changes only aggregation plumbing and evidence persistence: pass the validated metadata `rows` sequence rather than a tuple-keyed dictionary; atomically save `raw_records.json` before aggregation; bind and verify that file's SHA in final result/launcher. The accepted numerical core, model/data identities, 44 fixed starts, H100 arithmetic and resource guards are unchanged. Prior immutable sources and failed output are not overwritten.

## Regression evidence

Independent command from `/tmp/p030-integration-stage-sota`:

```sh
P030_CORE_SOURCE=/workspace/fluid_control/scripts/p030_train_horizon_core.py P030_REVIEWED_LAUNCHER=/workspace/fluid_control/scripts/run_p028_h10_comparison.py CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider tests
```

Result: **25 passed in 0.88 s**. A synthetic 44-pair fixture now invokes the actual canonical core aggregation. A separate production-AST assertion checks that `execute` passes `rows` as the third argument and persists raw records before the aggregation call; this would reject the failed v2 caller. These are CPU software tests, not physical/model accuracy evidence.

No further concrete blocker found in this bounded diff. A separately approved new execution is still needed to obtain numerical results. Completion will remain diagnostic-only, never H100/PPO/CFD admission; no gate, physical tolerance, optimizer or model architecture change is part of this repair.
