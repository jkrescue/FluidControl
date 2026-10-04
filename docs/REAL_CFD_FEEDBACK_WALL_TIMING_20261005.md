# Real-CFD feedback wall-time evidence

The control interval `0.1 D/U` is nondimensional CFD time. It is not a 100 ms
wall-clock deadline and no existing result establishes real-time execution.

## What the completed CFD-only runs measure

The immutable journals are:

- `artifacts/direct_cfd/directppo2048_b00_eval80_v1/worker_env{0,1}.jsonl`
- `artifacts/direct_cfd/directppo2048_b01_eval80_v1/worker_env{0,1}.jsonl`

Each branch contains 800 clean physical control steps. `solver_wall_seconds`
times only the `docker exec ... pimpleFoam` segment. Consecutive step-event
timestamps provide aggregate completion-to-completion throughput, including
policy/socket coordination, file operations, observation extraction and
scheduling, but do not separate those components. The two paired branches run
concurrently, so their solver times must not be summed.

| run/branch | solver median (s) | solver p95 (s) | step-event delta median (s) | 800-step span (s) |
|---|---:|---:|---:|---:|
| b00/env0 | 1.924 | 2.227 | 2.262 | 1734.29 |
| b00/env1 | 1.953 | 2.251 | 2.265 | 1733.48 |
| b01/env0 | 1.558 | 2.059 | 1.795 | 1379.41 |
| b01/env1 | 1.557 | 2.080 | 1.771 | 1379.41 |

This supports historical paired throughput of about 0.46 control-step pairs/s
for b00 and 0.58 pairs/s for b01. The roughly 0.24--0.34 s difference between
median event spacing and median solver time is an inseparable residual, not a
measured policy-inference latency. These are CFD-only PPO measurements and do
not measure the future PhysicsNeMo-assisted controller.

## Timing added for the next eligible real-CFD evaluation

Execution remains subject to candidate and policy admission. Only the timing
instrumentation is approved here; no candidate PPO or CFD run is authorized.

The canonical FNO-PPO OpenFOAM runner now records monotonic durations for
policy inference, action/boundary configuration, the paired CFD solve and each
branch, observation extraction, progress evidence writing and the complete
control step. It records run total time and mean/median/p95/max summaries.
UTC timestamps are for event location only; duration uses the monotonic clock.
The paired-CFD duration includes process launch, concurrent waits and solver-log
health checks. Each per-branch duration starts immediately before its `Popen`
call and ends when the corresponding concurrent `wait` returns; it therefore
includes process launch and scheduling and is not pure solver compute time.
While a step's progress-file write is itself being measured, its live row is
explicitly `PENDING_PROGRESS_WRITE`. After the loop, a final atomic progress
snapshot makes every row `COMPLETE`; the finalization write is not recursively
counted as part of the last control step. A pending row's `completed_steps`
still means the physical CFD step completed; only its timing fields await
sealing. `result.json` also contains only complete timing rows.

`--wall-deadline-seconds` is optional and unset by default. If explicitly set,
the runner counts misses without changing the action, solver, control interval
or physical acceptance criteria. No extra CFD case is run for timing. Old runs
are not backfilled with timing components that were never recorded.
