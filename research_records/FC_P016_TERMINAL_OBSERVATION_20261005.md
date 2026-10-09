# FC-P016 terminal observation — not admission

Observed 2026-10-05 12:24 UTC. The exact service invocation
`a95370a65c2b47e0b0e2926261937e33` is retained active/exited, MainPID=0,
Result=success, ExecMainCode=1, ExecMainStatus=0. The internal GPU guard reports
exit_code=0. Independent full terminal audit is still in progress.

Result SHA256: `f760d2e79a9fc797501f6481f74808cb0ffe790149fdb67340ac9abeb3454248`.
Result: `artifacts/fcp016_fixed_panel_fit_20261005/result.json`.
Recorded optimizer updates=32, window exposures=192. No candidate saved.

## Recorded terminal32 versus initial0

| Measurement | H1 relative change | Autoregressive relative change |
|---|---:|---:|
| Six-window normalized objective | +0.175% | -3.803% |
| Five nonzero-window tail62 bias squared | +420.873% | +208.108% |
| Five nonzero-window tail62 centered residual MSE | -14.939% | -10.884% |
| Five nonzero-window tail62 absolute Cl-prime RMS error | -13.186% | -4.176% |

The preregistered simultaneous-improvement interpretation is unsupported.
Waveform/amplitude improvements do not compensate for the increased mean bias.
These are local training-panel measurements, not generalization or CFD control
results. The original admission criteria have not changed.

Next action: inspect the same-batch objective/gradient/update mechanics and the
recorded optimization trajectory before proposing further full-data training.
No automatic PPO, no extended training and no selection of intermediate panels.

## Monitoring

Dashboard update `2c4f3cc` is deployed. Root reran 23 focused tests; the implementation
agent reports 62 regression tests. The original private localhost API was queried
after restart and reports verified terminal P016, independent audit pending, and
the pinned P015 formal rejection (joint 1/6). This API check does not establish
rendered-browser visual correctness or scientific acceptance.
