# P026 formal caller software review

## Reviewed identity

Staging root: `/tmp/fcp026_formal_callers_stage_sota`.

| File | SHA-256 |
| --- | --- |
| scripts/evaluate_tandem_fno.py | daef4a3a6eb1b9190f5cde728581b0c1b4b9656f56e865cf61a53deb1f37de88 |
| scripts/diagnose_fno_force_window.py | eee1f59fa269a22556048f3fde8fc7ecbc31bb913571e49075dead92618a8576 |
| tests/test_p026_formal_history_callers.py | 98d19bda017c55b20a14b02cae7553c9abe961474801c246044c3dd4a9dc93fb |
| tests/test_p026_actual_formal_mains.py | 3dbb3729b4456b672430b67e6933b3de5568dc0ef6436893132472774aa10fb8 |

## Evidence and limits

No blocking numerical-wiring issue found within the tested scope. Explicit history profiles route K1/K4 separately from legacy calls; flow predictions remain the autoregressive state source, and aerodynamic field outputs do not replace that flow state. Original metric calculations remain in the actual caller paths.

The existing six helper tests cover K1 rollout equivalence, K4 history routing, force-window reset/padding, explicit profile rejection and unchanged dual output combination. Supplemental independent tests execute the actual argument parsers and both `main()` functions using synthetic HDF files and mocked official model loading/distributed initialization/configuration. They do not exercise official checkpoint loading or real validation data.

The final supplemental file passed all six CPU tests in 1.70 seconds with CUDA hidden, bytecode/cache writes disabled and the test process started from staging. Actual evaluation H1/H10/H50/H100 case and summary metrics match legacy K1 exactly. Actual force-window K1 case/pair results also match exactly. In actual K4 force-window evaluation, poisoning every observed state after frame zero leaves predicted forces/window statistics unchanged while truth-field error diagnostics change. Additional independent-start tests at starts 0, 2 and 5 poison every frame strictly after each respective start and verify exact equality of recurrent raw outputs and predicted states. This closes the earlier helper test's weaker global poisoning coverage.

This is CPU software-fixture evidence, not scientific validation, a trained-candidate acceptance, PPO authorization or a real-CFD result. Full official candidate reload, signed provenance and independently approved original numerical evaluation remain separate requirements.

## Synthetic plot spill and recovery

An initial supplemental evaluation test omitted the visualization CLI override and created 60 synthetic PNGs under the evaluator's default `artifacts/tandem_fno/rollout_visualizations` directory. The reviewer stopped only that exact pytest process. All 60 file birth timestamps were within the test interval beginning 2026-10-05 19:22 UTC; Root independently confirmed they were newly created and had no registered dashboard references. They were not scientific images.

Root moved all 60 files, without deletion, to `/home/USER/workspace/fluid_control/engineering_quarantine/p026_fixture_plots_20261005_1922`, outside the repository and dashboard. The final fixture explicitly disables visualization, supplies a temporary visualization path, and changes cwd to its exclusive pytest temporary directory before executing either caller. Report outputs were explicitly temporary throughout. The corrected final test was rerun successfully. No production source, scientific dataset, model or scientific execution was changed by the test.
