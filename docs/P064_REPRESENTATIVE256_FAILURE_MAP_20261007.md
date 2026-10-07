# Representative256 failure map — saved-array diagnosis

2026-10-07. Read-only CPU arithmetic; no FNO import/forward, optimizer, new CFD, validation or frozen-test access. B remains default; candidate retention remains FAIL. This is descriptive, not a new gate or causal attribution.

## Evidence and formulas

- Training result: `artifacts/p064_representative256_training_20261007/result.json`, SHA `50617c1c49cebcdc2198fb1cc427f7a7289dc1912846d3f882b0a257025b6d6a`.
- Matched-precision six result: `artifacts/p064_fit256_fixed_six_20261007/result.json`, SHA `57add3a45f4cedcde94fbc242337e64cd6d54d51156e4d2be27c29cb8c309289`.
- Arithmetic source: `scripts/analyze_rep256_failure_map.py`, SHA `f335369ee8ebb154a87bea8fa5fa9cb285885ef2ada812eea7dc63f8e10fe1d8`.
- Full compact CSV: `artifacts/p064_representative256_failure_map_20261007/metrics.csv`, SHA `96e8784c587176c2d46a2b543de849d9ea2b1cd569bffd9b7c9e6e70c5ca8824`; 1,010 rows. JSON summary alongside it.

For selected points, error = saved physical prediction minus the bound HDF force[target_index]. Only 256 four-force labels were read; each original target SHA was verified, no state fields read. For six arrays, physical error = (saved normalized prediction − target) × saved force_std. MAE is mean absolute error; total-Cd error sums front/rear Cd errors before absolute value. `delta_mae` = candidate − B (negative is improvement). `six_ar_minus_h1` is the difference of the two MAEs, **not** MAE of trajectory increments. All inference underlying these two results used highest/no-TF32.

Original 256 sample weights and one duplicate remain; 109/45/38/64 points in base20/train8/train16/controlled-b00. Action groups use exact saved values: both zero, changing, or constant nonzero (23/134/99 points). No tolerance or new action labels inferred. Six arrays have no saved actions: per-point action classification is unknown there. Lead groups are origin-relative, not independent forecast experiments. CSV includes every available selected lead and four 25-lead bins; six includes each case and four 25-step segments. Metadata-only same-case/target-index overlap is flagged separately, not proof of identical model input tensors.

Action-count clarification: the earlier preparation shorthand “132 changing / 25 zero” is reproducible on the **same saved panel** as two different, nonexclusive definitions: `not np.isclose(omega_current, omega_next)` gives 132; `omega_next == 0` gives 25. The present mutually exclusive exact groups instead give 134 changing / 23 both-zero / 99 constant-nonzero. The two extra exact changes are consumed positions 65 and 185 (`dynamic_train8_b02_prbs`, target indices 153 and 73): `0.3749992251396179 → 0.375`, which default NumPy tolerance treats as constant. The two extra next-zero records are positions 8 and 136 (`b00_projected_ppo_train`, targets 97 and 436): `−0.025368154048919678 → 0` and `0.06898051500320435 → 0`; they belong to changing, not both-zero. Current-zero alone counts 26, not 25. The original pinned panel summary has no action-count fields, so these reproduce the shorthand numerically rather than establish its historical code provenance. No panel identity, data, CSV definition or gate changed.

Closed's retained preparation message further specifies the earlier changing threshold as `abs(omega_next − omega_current) > 1e−6`, also giving 132 (the two differences above are about `7.7486e−7`). Thus default `np.isclose` is a reproducing comparison, not the claimed original implementation. The earlier zero-action definition was not retained; next-zero=25 is the directly reproducible interpretation, not proven historical provenance.

## Where the training improvement came from

| Selected family | N | rear-Cl MAE B → candidate | total-Cd MAE B → candidate |
|---|---:|---:|---:|
| controlled-b00 | 64 | .18012310 → .05589407 | .04575244 → .01378297 |
| base20 | 109 | .03959399 → .03187419 | .01035020 → .00910466 |
| train8 | 45 | .03954324 → .04069088 | .01894890 → .02011989 |
| train16 | 38 | .03494258 → .02519723 | .01222530 → .01367649 |

The b00 quarter contributes .03105726 of the global .03558905 rear-Cl MAE reduction (87.3%), and .00799237 of .00810145 total-Cd reduction (98.7%). These are signed weighted MAE-change contributions, not gradient/cause estimates. Its initial errors were also much larger.

All four force-channel MAEs worsen on the **selected train8 points themselves**. Front-Cd and front-Cl worsen in every original family, while both improve on b00. Overall front-Cl MAE rises .00944933 → .01125044. Thus failure is not exclusively generalization away from selected points; there is observable within-panel source/channel tradeoff.

Changing-action points improve all four channels; constant-nonzero points worsen both front channels; zero points worsen front-Cl while improving the other three. These groups confound family/state coverage and do not isolate action causality. Each lead-quarter has 64 points: rear-Cl and total-Cd improve in all four, while front-Cl worsens in all four. No single late-lead-only defect explains the aggregate result.

## Six-window transfer and rollout

Physical rear-Cl MAE (H1 / continuous AR100):

| Original six case | H1 B → candidate | AR B → candidate |
|---|---:|---:|
| base20 b00 zero | .034972 → .019290 | .032720 → .030531 |
| train8 b00 PRBS | .044951 → .048171 | .105341 → .122677 |
| train8 b02 PRBS | .042728 → .035358 | .096503 → .135108 |
| train8 b04 PRBS | .047964 → .049421 | .073783 → .071543 |
| train8 b06 PRBS | .055311 → .041193 | .068805 → .071337 |
| train16 PPO b00 | .043238 → .045807 | .046638 → .053041 |

Six H1 front-Cd and front-Cl MAEs worsen in **all six cases**. Rear-Cl H1 improves in three cases but AR improves in only two. Particularly b02 H1 improves while AR deteriorates strongly: the AR−H1 MAE gap grows about .04598. Aggregate original balanced H1/AR objectives remain +15.8169%/+14.2703% worse. This does not prove frozen-flow error is the cause; both trajectories already share the same frozen flow.

Only 1/2/3/5/2/1 of the 100 endpoints per six case overlap selected target indices. Six provides wider train-time trajectories, not an independent development test. Improvements are therefore neither universal nor confined strictly to exact selected points: e.g. the zero six window rear-Cl improves broadly, whereas selected train8 points already regress. Sparse fitting and source/channel tradeoff are compatible explanations, not identified causes.

## Decision within the remaining time

Do **not** launch another blind training run from this map. Existing PROJECT_STATE/EXPERIMENTS record H1-only F, longer absolute64, AR-reset G, pressure auxiliary H, reflection, late-state I, temporal-increment auxiliary, loss-scale gradient diagnosis, and fixed40/representative256 fitting. Repeating H1-only, extra steps, ad hoc force weights, auxiliary pressure or temporal loss would not be a new falsifiable hypothesis.

The strongest new observation is concentration of gains on already-hard b00 versus old-family/front-force regression. It does not yet distinguish gradient interference, sampling sparsity, parameter drift or unequal source difficulty. A new family weight or layer-freeze choice would be selected post hoc from these outcomes; a sub-hour train plus the necessary matched-precision retention evaluation would not identify that cause. No adequately justified nonduplicate corrective experiment is recommended within this remaining window. Keep the failed candidate and this map as evidence; preserve the working B CFD delivery, all original gates, and the scientific deadline.
