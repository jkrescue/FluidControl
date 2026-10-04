# FC-P003C first-position gradient debug approval

Lead approval, 2026-10-05 Asia/Shanghai. Scope: one Worker GPU diagnostic execution, not training or scientific admission.

The original diagnostic stopped at position 0 with `regular total gradient decomposition differs`. It recorded no numerical residual. The failure is preserved in `artifacts/fcp003c_train_gradient_diagnostic_20261005_failed_v1/`; no optimizer or model save occurred. Main FC-P003C training remains unchanged.

Approved change: the observation-only patch reviewed at commit `7c92e02` prints actual residual/reference norms and the original tolerance before failing. Run only the first original insertion position with `--max-positions 1`. A single-position result must be labelled DEBUG_ONLY, not a completed 16-position diagnostic. This label-only correction is allowed; no further numerical change is authorized.

Execution contract:

- Node: Worker `WORKER_HOST`; independent new output and unit, preserve the original failure.
- Same Main-e2 parent, official image, resolved configuration, train-only data, normalization and sampler as approval `de06124`.
- Preserve FP32 computation, objective definitions and the original relative tolerance `2e-5`; no optimizer, parameter update, checkpoint or validation/frozen access.
- Record and verify the new diagnostic, launcher and source snapshot SHA values before execution. Only the reviewed observation change, one-position limit, debug label and mechanical Worker paths may differ.
- Retain at least 20 GiB physical unified MemAvailable and allocator fraction at most 0.35. Hard runtime cap: 5 minutes; clean up only this execution's container.
- Execute once. Do not invoke the full 16-position completion validator or silently retry. Return raw logs, exit status and measured numerical residuals to Main with SHA verification.
- Any later tolerance, precision, algorithm or training change requires a separate evidence-based decision. This diagnostic does not authorize PPO or change physical acceptance criteria.

Owner: Surrogate agent. Lead will inspect the returned numerical evidence before choosing a subsequent action.
