# Gate-B terminal-drag metric integrity audit

Entry point: `scripts/audit_gate_b_metric_integrity.py`.

This independent audit corrects an ambiguity in the existing Gate-B reports
without modifying or rerunning the evaluator. The evaluator stores, for every
case at horizon 100, the terminal total-drag RMSE, physical target RMS and
number of evaluated segments. The audit reconstructs the pooled metric as

```text
pooled NRMSE = sqrt(sum_case(n_case * RMSE_case^2)
                    / sum_case(n_case * target_RMS_case^2))
```

It also retains the evaluator's equal-case macro NRMSE and the worst-case
NRMSE. The fixed Gate-B ceiling remains 10%; this script has no option to
change or relax it. The pooled result, rather than the macro average of ratios,
is the strict aggregate accuracy decision.

## Provenance contract

Every invocation must explicitly provide the expected split, evaluation data
profile, normalization profile, case count, checkpoint SHA-256 and
normalization SHA-256. The audit then requires:

- `split` is exactly the predeclared `validation` or `test` split;
- report cases exactly equal the sorted HDF5 filenames in that split;
- every HDF5 file declares the same split, stores `force` as `(frames,4)`,
  and has the exact root attribute order
  `front_cd, front_cl, rear_cd, rear_cl`;
- manifest profile and split count match the declarations;
- report evaluation/normalization roots match the supplied roots;
- four force channels are ordered
  `front_cd, front_cl, rear_cd, rear_cl`, with indices `0,1,2,3`;
- report, checkpoint metadata and normalization manifest agree on action scale;
- checkpoint metadata agrees on force channels/indices and the 6-input,
  7-output model contract;
- checkpoint filename epoch, report epoch, checkpoint directory and SHA-256
  agree;
- normalization is train-only, has finite positive scales, and its SHA-256
  matches the predeclared digest;
- every case is stable, has at least one segment, and internally satisfies
  `NRMSE = RMSE / target_RMS`;
- summary macro NRMSE and segment count reproduce the per-case records.

The command refuses to overwrite an existing output path.

This supports v3 H20/rear-drag validation and frozen-test reports, plus v4
validation reports whose evaluation and normalization profiles may differ for
the v3 parent comparison.

## Scientific boundary

The reconstructed metric is still the error of the **instantaneous terminal
force at step 100**. The existing evaluator does not retain the other 99 force
predictions in that rollout. Passing this integrity audit therefore does not
establish accurate mean `Cd_total` over the 100-step window, action ranking,
or closed-loop CFD drag reduction. A separate sequence/window-mean metric is
required before making those claims.

## v3 H20/rear-drag reference audit (2026-10-03)

The strict audit was run read-only against the existing epoch-9 PhysicsNeMo
checkpoint and reports. All three provenance contracts passed, but all three
pooled terminal-force metrics remain above the unchanged 10% ceiling:

| Evaluation | Cases / segments | Existing macro NRMSE | Pooled NRMSE | Worst case |
| --- | ---: | ---: | ---: | ---: |
| v3 validation | 4 / 116 | 13.414% | 15.047% | 23.997% |
| v3 frozen test | 5 / 145 | 15.540% | 16.172% | 24.653% |
| independent phase test | 1 / 6 | 10.812% | 10.812% | 10.812% |

The initial machine-readable outputs were generated before the per-HDF5
attribute check was added. They are retained unchanged for provenance. The
current schema is recorded in new `_v2` outputs:

- `artifacts/tandem_cylinders/gate_b_metric_integrity_v3_h20_rear_drag_validation_v2_20261003.json`
- `artifacts/tandem_cylinders/gate_b_metric_integrity_v3_h20_rear_drag_test_v2_20261003.json`
- `artifacts/tandem_cylinders/gate_b_metric_integrity_v3_h20_rear_drag_phase_v2_20261003.json`

The phase macro and pooled values coincide because that report contains only
one case. These corrected values strengthen the existing failure conclusion;
they do not reopen or reinterpret the frozen split.
