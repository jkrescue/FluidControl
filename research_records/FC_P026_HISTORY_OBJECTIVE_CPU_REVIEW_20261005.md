# P026 history force objective: CPU engineering review

Status: implementation and independent CPU review passed; no GPU execution or
scientific admission. This review records completed tests, not training results.

Source `p026_history_objective.py` SHA256:
`4d27fb53e05df73ba94d84bf42ba8205d78ebe6f91de832a68659870ea7d77c0`.
Tests `test_p026_history_objective.py` SHA256:
`907e25bd4a8ead87eee1f65a62e22c84f999471718e8ca53f66d198367344277`.
Original P013 objective is imported only after verifying SHA256
`f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7`.

Root read the entire source/tests and ran the combined suite on Spark with CUDA
hidden: 20 passed in0.78s. Independent reviewer read the implementation and ran
the same seven new plus thirteen existing adapter tests: 20 passed in0.75s.

## Covered invariants

- K1 matches the pinned original P013 objective for batch1 and batch2: all
  captured force predictions, aggregate/channel losses and parameter gradients
  are compared with zero tolerance in the synthetic convolution fixtures.
- K4 input order is chronological, including the trajectory-start padded prefix.
  Final current state is99 and the selected next action is100. Final force target
  index99 contributes the expected loss.
- Changing true H1 states and target forces does not alter AR predictions.
  Aerodynamic field outputs are discarded even if a test supplies invalid fields;
  the independent frozen flow remains the only state predictor.
- All supplied state/action histories must be frozen. With no aerodynamic
  feedback between steps, ten weighted chunk backwards are mathematically the
  same force objective as a full-window backward. The K4 toy-model test compares
  gradients at rtol1e-5/atol1e-7, loss absolute difference below1e-6.
- All288 additional lifting-column gradients in the K4 toy fixture are finite
  and nonzero. Nonfinite input data and trainable histories are rejected.

## Scope and next action

The tests use CPU fixtures, not a production FNO update, resource measurement or
scientific accuracy evaluation. They do not independently prove all future
callers assemble their inputs correctly. Separate P009 flow/P018 aerodynamic
parent loading, full-size gradients/memory, official save/reload and history-aware
formal/online reset callers still require integration and checks.

Prepare the one-window K1/K4 no-update resource harness under the reviewed plan;
GPU execution requires separate Lead review of final source and resource guards.
No optimizer, checkpoint, metric threshold or CFD control result changes here.
