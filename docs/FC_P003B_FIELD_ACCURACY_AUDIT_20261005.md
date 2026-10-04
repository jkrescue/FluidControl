# FC-P003B field accuracy: separate from force admission

Lead read-only audit, 2026-10-05 Asia/Shanghai. No new inference, data,
threshold, or training change is introduced by this note.

The evaluator accumulates physical-unit squared field errors at each segment's
terminal endpoint over the existing fluid mask and ROI, then pools the error
and reference squared sums across cases. These are not errors averaged over
all intermediate rollout times. The joint velocity denominator combines u and
v; streamwise mean flow can make it much less sensitive to transverse-flow
errors than the v-only ratio. Pressure uses the stored dataset convention;
no new gauge correction is applied in this comparison.

## Dynamic6: fixed rolling starts, observed actions

| Horizon | Endpoint samples | P003 velocity L2 | P003B velocity L2 | P003B u L2 | P003B v L2 | P003B p L2 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 1200 | 0.4722% | 0.4722% | 0.3426% | 1.4936% | 1.3817% |
| H10 | 1146 | 3.3211% | 3.3246% | 2.4738% | 10.2415% | 10.1266% |
| H50 | 906 | 6.6864% | 6.7008% | 5.2975% | 19.1255% | 21.9721% |
| H100 | 606 | 8.7282% | 8.7648% | 7.1992% | 23.4929% | 27.2588% |

Both candidates report zero nonfinite/failed segments for these panels.
Finite predictions are not accurate predictions. P003 H100 v/p errors were
23.3609%/27.5998%, respectively: the dynamic-pair change did not materially
repair long-horizon fields, either. H100 is 10 D/U for the dataset's 0.1 D/U
sampling; it is not the final 80 D/U physical-control evaluation.

## Interpretation for the next experiment

The substantial rotating-action H1 rear-Cl errors coexist with much smaller
H1 field errors. This supports separately investigating the learned force
mapping, but is not proof of a unique root cause. FC-P003C's approved paired
force intervention does not by itself promise to solve long-rollout v/p
errors. Its unchanged full evaluator must continue reporting these field
metrics alongside force/action/window errors. Do not describe endpoint Cd
accuracy, a low joint velocity ratio, or a force-only improvement as uniformly
high-accuracy flow prediction.

If a subsequent force intervention succeeds but field errors remain large,
record the remaining field-prediction limitation and assess its control impact;
do not silently narrow the user's goal to force prediction alone. This audit
does not authorize architectural changes or relax existing admission criteria.

## Evidence

- P003: `artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005/posteval_fc_p003/dynamic6/evaluation.json`, SHA-256 `68371e0702c23e16ac3d25e1b5f2fefafd368adc4626ca08fea7fed1e43f7dfd`.
- P003B: `artifacts/tandem_fno_dynamic_paired_interleaved_lambda10_20261005/posteval_fc_p003b/dynamic6/evaluation.json`, SHA-256 `4b6e901c9b791b1da8782d9aea24560b6e06eabaa6f1df7aaec1eaf6e4253669`.
- Formula inspected in `scripts/evaluate_tandem_fno.py`: terminal `target_indices = start_indices + horizon`, masked physical field error, and `relative_field_metrics(pooled_field_sums)`.

This is retrospective documentation of already completed evaluations, not a
new independent experiment or preregistration.
