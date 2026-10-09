# D015 H1 error decision tree

This is a conditional interpretation plan for the approved, no-gradient D015 diagnostic. It is not a new gate, a causal conclusion, or authorization for training, PPO, or frozen-test access. All comparisons retain the four physical force channels, action-minus-zero force deltas, and the existing flow-field accuracy objective.

## Read the three panels in this order

1. **Train paired, targets 1–100.** If absolute and delta H1 errors are already large on the exact endpoints receiving extra paired supervision, the primary evidence is training-fit failure. Do not call the later validation error a generalization failure yet. If delta error is large but action and zero absolute errors share a similar bias, the unresolved quantity is action sensitivity; if delta is small but both absolute errors are large, the common state/zero-flow prediction is the more likely limitation.
2. **Train late, targets 100–200.** Target 100 is the explicit overlap check. If the paired prefix is accurate but errors rise consistently after target 100 within the same trajectories, this supports a within-trajectory/time-window coverage problem. It does not imply that targets 101–200 were unseen: they were present in regular H100 training, but lacked the extra paired term.
3. **Validation late, targets 100–200.** Only if both train panels are accurate while the same metrics degrade on b01/b05 may the evidence be called an operating-condition generalization gap (phase/state/action-history jointly), not merely a time-window gap. Similar errors on train and validation instead support model/optimization mismatch common to both splits.

Channel interpretation remains explicit: an isolated rear-lift failure is different from a common four-force failure. Improvements in force or delta metrics do not establish an acceptable surrogate if the unchanged u/v/p field diagnostics remain inaccurate. No panel changes the existing admission thresholds.

## At most two conditional next experiments

1. **Train-fit calibration, only if the paired panel itself fails.** Run one bounded train-only replay diagnostic from the same fixed parent, using the existing dynamic8 pairs and true-state objective, with a predeclared small update budget and no validation-based selection. Change only paired-update exposure; keep model, normalization, data, optimizer family, force weights, gates, and architecture fixed. If paired-panel H1 errors do not materially and consistently decrease across phases/actions/channels, stop this branch rather than increasing lambda or network size. If they decrease while flow errors worsen, also stop: the force-only trade is not acceptable.
2. **Coverage intervention, only if train fit passes.** If only train-late degrades, move the fixed paired supervision window from targets 1–100 to 101–200 using the existing train8 CFD labels and keep the number of paired updates unchanged. If both train panels are accurate but validation-late degrades, instead acquire a separately predeclared train-only paired set at new restart phases/action histories; do not choose waveforms from b01/b05 errors. In either case, rerun the unchanged complete field/force/window protocol. Failure to improve the predicted panel without degrading u/v/p is the stop condition.

The two coverage alternatives are mutually exclusive branches of one experiment slot. D015 results must be read before selecting a branch, and no result may be described as causal proof by itself.
