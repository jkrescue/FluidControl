# Control-force error localization: real-state one-step versus free rollout

Lead read-only diagnostic, 2026-10-05 Asia/Shanghai. No new inference, training,
data generation or threshold change. This is development evidence, not a frozen
test or proof of a particular architectural cause.

Candidate: FC-E006 lambda0, model SHA
3a222827c9f0b3d39d72cec035c20c7d35411ecfc119415109c9528ea7876de7.
Source: artifacts/tandem_fno_paired_stats_lambda0_20261004/posteval_fc_p001/dynamic6/segments.json.
Source SHA: c7c1af13da3775ab8c88b2a4256e941abaf525f87f62e71deadbd87b453a0c64.

## Matched physical endpoints

For each H100 segment beginning at index s, compare the H1 segment of the same
case beginning at s+99. Both predict the same physical target frame s+100. H1
receives the true CFD state at s+99; H100 receives its own recursively predicted
state. Exactly 101 endpoints per case, 606 total; the maximum difference between
their recorded target total-drag coefficients is exactly zero.

Compute each column as the arithmetic mean of the corresponding per-segment
rear_cl_mae over those 101 endpoints. These are instantaneous coefficient MAEs,
NOT Cl-prime RMS window errors and NOT normalized percentage errors.

| Case | True-state H1 rear Cl MAE | H100 rear Cl MAE |
|---|---:|---:|
| b01 minus | 0.190859270 | 0.236086139 |
| b01 plus | 0.159689075 | 0.124202035 |
| b01 zero | 0.008699855 | 0.039290220 |
| b05 minus | 0.161588188 | 0.132526178 |
| b05 plus | 0.190879490 | 0.242659049 |
| b05 zero | 0.007318613 | 0.043599125 |

## Interpretation and next action

Large controlled-force error already exists with a true-state one-step input.
Therefore autoregressive accumulation cannot be the sole explanation. H100 is
not uniformly worse at these matched targets; compensation of errors is also
possible. The table alone cannot identify whether force-output learning, state
coverage, conditioning, normalization or incomplete state representation causes
the discrepancy, and does not prove an architectural change is necessary.

Keep FC-P003 and FC-P003B unchanged. Dynamic paired supervision directly tests
one data/objective-side explanation; after their common evaluation, repeat this
same-endpoint diagnostic for each candidate before deciding whether to target
one-step force learning or rollout stability next. Do not select a candidate
using the most favorable endpoint or bypass the full force-window gate.

Reproduction algorithm: build a lookup keyed by (case,horizon,start); for every
entry with horizon=100, select lookup[(case,1,start+99)], verify equal recorded
target_total_drag, then group both entries' rear_cl_mae by case. Do not compare
unmatched H1 all200 windows against H100 all101 windows to infer accumulation.
