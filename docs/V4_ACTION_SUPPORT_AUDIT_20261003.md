# v4 real-CFD action-support audit

Date: 2026-10-03 UTC. Source: actual HDF5 `omega` arrays in
`data/curated/tandem_cylinders_control_gap_v4`, not planned action tables or
synthetic samples. Manifest SHA-256:
`903dbf80e0c4b831698ece27775f3fdb65a204ad122cc7e28fef8320bcd5de24`.

| Split | CFD trajectories | Frames | Fraction `|omega| <= 0.1` | Fraction `|omega| <= 1` | Fraction `|omega| > 3` | Median `|omega|` |
|---|---:|---:|---:|---:|---:|---:|
| Train | 28 | 22,428 | 7.70% | 27.86% | 27.20% | 1.960 |
| Validation | 4 | 3,204 | 6.05% | 23.38% | 31.99% | 2.103 |
| Frozen test | 5 | 4,005 | 8.76% | 22.60% | 37.80% | 2.286 |

These frame counts are **not independent experiments**: each trajectory is a
time-correlated OpenFOAM sequence. The two new signed-pulse training
trajectories each spend 41.1% of frames at `|omega| < 0.05` and reach at most
`|omega| = 1.25`; each of the original 26 train trajectories reaches at least
about `|omega| = 3.44`, and most reach 5. This is targeted real-CFD coverage
of low-amplitude signed actions, not a broad closed-loop-policy distribution.

The dataset can support a first action-conditioned FNO experiment, but is not
yet sufficient to assert accurate prediction for arbitrary feedback policies:
there are only two new low-amplitude schedules, one restart phase, no measured
policy trajectory in the training addition, and only four validation
trajectories. The v4 frozen five-case test is unchanged from v3 and must not be
used to tune this data profile. Validation action ranking at a distinct initial
phase and the fixed 100-step error criterion determine whether more CFD labels
are needed. If the model selects wrong actions or fails the 10% error gate,
acquire new real CFD near the policy's actual action histories before PPO or a
physical drag-reduction claim.

Audit computation: enumerate sorted `*.h5` in each split, read only each
`omega[:,0]` array with `h5py`, concatenate within split, and calculate the
listed empirical frame fractions and median. No state or force arrays were
loaded or modified during this audit.
