# FC-P031 objective and no-update probe implementation review

Verdict: ACCEPT for source preparation and a separately approved no-update resource probe. This is not GPU execution approval, full-training approval or scientific admission. Staging: `/tmp/p031-impl-sota-20261006`.

## Reviewed bytes

| File | SHA256 |
|---|---|
| scripts/p029_control_aware_flow_objective.py | e34721dbf4e6d921394800edcc2df451884a43ce9f3d2d36bc5ec3ded7851b23 |
| scripts/train_p031_control_aware_flow_h25.py | 24076d0ff282fe5c0780c8fe3ea528fc2ef2fbfa2ffe91120d87ca2b7cded89d |
| scripts/run_p031_control_aware_flow.py | 928de1536ccff031633785ffd6eaff0367ee7057af5f0eafbf36e5be50065058 |
| tests/test_p031_control_aware_flow_objective.py | 048d0361c739a3110b5bcb8fadf5fc3ba71b5070b6e1ed57b42be6eb594fa5c8 |
| tests/test_train_p031_control_aware_flow_h25.py | fc4419a8aece32b18c7129ab9f35c8494fdba1966ab27ebde19b448dab76d852 |
| tests/test_run_p031_control_aware_flow.py | 482f991e4dfbb2d88517171e2607af28b7d5981dd5ca2bb223282378bdfa17fc |

Independent command from canonical repository cwd:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider /tmp/p031-impl-sota-20261006/tests
```

Final result: **35 passed in 0.67 s**. These are synthetic CPU fixtures only. No real HDF, checkpoint/model load, Docker or GPU work was executed by the reviewer.

## Actual caller and numerical review

Read the complete objective, runner and launcher, not only their test fixtures. Default/explicit H10 preserve canonical P029 result and gradient tests. Only exact integer10 or25 is accepted. In H25 the frozen eval aerodynamic model predicts next force from current recurrent state and current/next actions before the flow update; later force terms preserve their input-gradient path through flow. Initial force has no flow dependency and terminal q25 receives field supervision only. No detach, architecture change, checkpointing, scale recomputation or future truth input was introduced.

The actual resource branch selects original `train[816]`, calls the objective with `backward=True, horizon=25`, masks the unused output rows using the reviewed helper, checks finite flow gradients and proves unchanged parent tensors. It creates no optimizer and never calls terminal save. Both loaded parent tensor identities are compared against the SHA-bound P029 scales receipt; the two denominators are fixed exactly. The official DataPipe/original1368 order and exact44 train hash contract remain reused. Startup precision is configured before role-loader validation.

The launcher binds the reviewed P028 lifecycle bytes, exact resource-probe singleton-order SHA, P031 statuses, train-only aliases, exclusive payload, and proof mounts. The inherited lifecycle retains exact container ownership and cleanup. Probe timeout900s, container12GiB/no extra swap, startup30/50GiB and continuous physical/CUDA20GiB remain unchanged. A review finding was resolved: low-level runner formerly admitted allocator fractions up to0.45; it now rejects anything except0.06, matching plan and launcher, with a negative fixture.

## Remaining execution boundaries

The full-train branch and explicit P031 save metadata are preparation, not authorization. Before training, the exact producer/dual-loader/candidate-auditor/official-reload/formal and diagnostic profile chain still requires its separate compatibility review, plus actual H25 probe evidence and optimizer/transient memory margin. The no-update probe cannot establish training peak memory; the retained optimizer estimate is a lower bound, not a guarantee. Source freezing and final mode-specific approval must bind these reviewed bytes and actual fixed inputs before any GPU execution.
