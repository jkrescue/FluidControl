# FC-P002 action-history coverage audit

This train/development-only audit reads only the `omega` and `time` arrays. It does not read flow fields or frozen-test data.

Definitions are recorded in the JSON artifact. A change is `abs(delta omega)>1e-8`; change fraction is the fraction of adjacent intervals that change. Constant dwell is a run without such a change, measured as sample count times median `dt`. For sign crossings, zero-valued samples are removed before adjacent nonzero signs are compared. Every source manifest and every case's `omega` and `time` content have SHA-256 identities in the artifact.

The observed scalar ranges do not support a simple amplitude/rate out-of-distribution explanation. Train20 covers `|omega|<=0.75` and `|delta omega|<=0.1`, but is almost entirely constant after a short ramp. Train8 and train16 contain genuine changing actions; train8 includes dwell times up to 3.8 D/U, while dynamic6 action branches have a 3.1 D/U longest dwell. Train16 changes nearly every 0.1 D/U and has greater total variation than dynamic6.

This only excludes an obvious scalar amplitude or rate-bound mismatch. It does not prove the validation trajectories are jointly in distribution: validation phases are b01/b05, whereas training phases are b00/b02/b04/b06; the matched-pair statistic supervision uses constant actions and only four distinct initial states. FC-P001 results must therefore be grouped by phase, action type and horizon before selecting the next single-factor hypothesis. No new CFD or training is authorized by this audit.
