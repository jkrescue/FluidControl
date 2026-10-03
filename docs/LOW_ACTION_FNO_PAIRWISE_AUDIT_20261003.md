# Low-action FNO pairwise audit — 2026-10-03

## Formal record

- Artifact: `artifacts/tandem_cylinders/low_action_phase94_fno_pairwise_h100_20261003.json`
- SHA-256: `c75d5801ddfb800f36b39f368d2e3e1036a9c2e72689ed9d9f73f9fd48823c2c`
- Scope: validation-only H100 endpoint diagnostic for the paired `-0.75` and
  `+0.75` open-loop trajectories. It is not a zero-control comparison, model
  selection result, or CFD closed-loop success claim.

## Result

Only `start=0` is a strict common-initial-state action comparison. Its true
endpoint difference, defined as `Cd_total(-0.75) - Cd_total(+0.75)`, is
`0.504740`.

| Model | Predicted difference | Absolute difference error | Sign correct |
|---|---:|---:|:---:|
| v3 H20 parent | 0.411791 | **0.09295** | yes |
| v4 H20 candidate | 0.393383 | **0.11136** | yes |

Both models therefore select the correct sign for this one pair, but one
correct pair is not an accuracy rate or evidence of robust control. Across all
29 matched elapsed-time starts, both models have `20/29 = 68.97%` sign
agreement. Starts greater than zero belong to trajectories already separated
by their earlier actions, so this all-start statistic is descriptive and is
not a counterfactual action-ranking accuracy.

The H100 value at `start=0` is an instantaneous endpoint near `t=104`. It is
not the physical acceptance statistic, which is mean Cd over `[114,174]`.
For scale only, 2% of the phase-94 zero-control mean Cd is approximately
`0.046` (`0.02 × 2.30171`). The v3 and v4 endpoint-difference errors are about
`2.02×` and `2.42×` that numerical scale. Because endpoint error and long-window
mean benefit are different observables, this comparison cannot prove either
model is reliable for the 2% control objective.

## Verification and display

- `tests/test_low_action_pairwise_fno.py`: 5 directed tests passed.
- `tests/test_live_research_dashboard_latest_evidence.py`: 7 directed tests
  passed.
- Python compilation, JavaScript syntax, and `git diff --check` passed.
- The live dashboard service is active and shows H100 pooled NRMSE, model MAE
  versus persistence MAE, and the strict `start=0` errors. It labels these as
  validation diagnostics rather than control success. The page refreshes every
  5 seconds; use `Cmd+R` if the browser has cached the older layout.
