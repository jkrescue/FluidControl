# FC-P025 independent terminal review — 2026-10-05

## Verdict

Operationally complete; the preregistered local-support test is **false**. The fixed symmetric statistical loss did not repair all simultaneous requirements. Close this isolated-input-block/statistical-loss branch as declared; do not sweep the coefficient or repeat this experiment. No candidate, surrogate admission, held-out evaluation, PPO or real-CFD control benefit is established.

## Execution and provenance

- Unit: `fluid-control-fcp025-isolated-statistics-20261005.service`.
- Invocation: `124d6521045a41cd9dcf5f35edff6712`.
- Observed terminal state: `ActiveState=active`, `SubState=exited`, `MainPID=0`, `Result=success`, `ExecMainCode=1`, `ExecMainStatus=0`.
- Output: `artifacts/fcp025_isolated_statistics_20261005/result.json`.
- Result SHA256: `7648df661439f530984504538bfaef3d1a602c1b960fd914fcba2f016fc4e74c`.
- Execution approval SHA256: `6fb2f810e395fdc477339352d677616ad782b7edca6535f282710e2ae2573700`.
- Protocol SHA256: `7f6cc9c66df5a23ecb0b56a6a6bf9da5ffc1ec494acba9e586397f113f580598`.
- Probe SHA256: `3411fccebaaa61301250c64ca4220a3c8900598ada227aab274bae770e3128bf`.
- Launcher SHA256: `98d4fa9d23d065f3af1f954e94dbf8458b0dd7b68344950c98344effc9cdd127`.
- Historical control: P023 HIGH, `artifacts/fcp023_input_block_20261005/result.json`, SHA256 `adfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb`.
- Elapsed harness time: 446.33250793 seconds.

Independent read-only checks rehashed approval sources, dependencies and the copied launcher, matched the protocol, and verified exact P023 HIGH initial raw rows/aggregates and runtime precision. This was an audit of the existing execution, not a new model run or data rewrite.

## Numerical and identity checks

The experiment used one fresh AdamW optimizer at learning rate `1e-5` for the sole 96-coefficient causal input block; original parent weights remained frozen. Six raw window gradients were averaged before clipping for each of 16 updates. Recorded optimizer state had one `[24,4,1,1]` parameter and 768 moment bytes. The old-base tensor SHA remained `c42ad84bf3fae9cd8c2b94335884bb98b14cafd48a9a3e2d190a38d5db9d3b2d` throughout recorded updates.

Verified counts: 96 window backwards, 600 frozen-flow calls, 9,600 training forward calls, 9,600 checkpoint-recomputation calls, 2,400 endpoint-panel calls and 1,200 ablation calls. Endpoint and ablation evaluations cover 36 windows in total.

All 96 recorded training totals reproduce `J0 + (5/16) * sum(four normalized tail mean/RMS squared errors)`. All six panel aggregates, comparison checks and repeat assessment were independently recomputed. Saved values are finite. Repeated raw panel rows are identical, and terminal zero-input ablation raw rows reproduce the initial rows exactly. Repeat agreement is observed numerical evidence, not a rigorous uncertainty bound.

Final block norm was 0.00154606213, with maximum absolute coefficient 0.000165686346. Final restoration is enforced by the reviewed runtime code before successful result writing; no saved terminal model exists for an independent reload check.

## Scientific outcome

Objectives use the same six-window macro aggregation; statistics use the same five nonzero windows and tail indices `[38:100]`. All comparisons retain the original protocol and thresholds.

| Metric | H1 terminal | AR terminal |
|---|---:|---:|
| Original objective | 0.00352265657663 | 0.00883103037389 |
| Mean-error squared | 0.000350339219824 | 0.000766511321306 |
| RMS-amplitude-error squared | 0.000950502330864 | 0.00688820540971 |
| Absolute RMS-amplitude error | 0.0259631942580 | 0.0744449898579 |
| Centered residual MSE | 0.00223951120061 | 0.0132076150555 |

H1 mean-error squared worsened by 0.1473296% from initial and exceeded P023 HIGH by `2.6796683326e-7`. H1 original objective also failed to improve over HIGH. AR mean-error squared improved by 0.1028832% from initial but exceeded HIGH by `7.3818633591e-7`. These are explicit failures of the simultaneous local rule.

RMS-amplitude-error squared improved against both initial and HIGH. Against initial, H1/AR centered residual MSE improved by 0.2474242%/0.3204728%; absolute RMS-amplitude errors improved by 0.3331635%/0.0876959%. These component improvements do not override the failed requirements. Squared RMS error and absolute RMS error are distinct quantities.

The result rejects this particular fixed statistical-loss intervention under the declared local comparison. It does not prove a universal capacity limit, unavoidable tradeoff, or a cause of the earlier full-surrogate failures. Physical mean-lift limits and surrogate prediction-error criteria remain distinct and unchanged.

## Resource completeness

- Host watcher: 225 samples; minimum MemFree 28.2702827454 GiB and MemAvailable 108.1546821594 GiB.
- Internal checks: 23,719; minimum MemFree 28.2385482788 GiB and MemAvailable 108.1208076477 GiB.
- GPU guard: 226 samples, exit code 0; minimum reported CUDA free memory 28.2387313843 GiB.

Both host memory floors remained above 20 GiB. Successful execution is separate from the unsupported scientific outcome. Any subsequent representation or optimization experiment requires a new scoped plan and approval; this review authorizes no further execution.
