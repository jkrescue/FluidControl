# FC-P015 first evaluation stage: provisional, not admission

Observed 2026-10-05 11:38 UTC. Formal service remains active under invocation
`c41e61fcaa5f49e3be9e0092c1e19bc3`; dynamic6 and force-window results are pending.

Independent review verified all four validation10 artifact hashes and their
lineage/precision/approval bindings. Step receipt SHA:
`f311354be60641563f46a3c9cab1bae7f4d470671270459ba1bc076d5b2d4185`.

Identical protocol: validation10, observed actions, stride25, batch4, same real
CFD data, normalization, starts and horizons. H denotes forecast steps, not
training epochs. One step is 0.1 D/U.

| H | Rear Cl MAE P009 | P013 | P015 |
|---|---:|---:|---:|
| 1 | .0201247 | .0799222 | .0413482 |
| 10 | .0279903 | .0875178 | .0415973 |
| 50 | .0284020 | .0945361 | .0319326 |
| 100 | .0387410 | .1037097 | .0369867 |

P015 improves all four horizons relative to P013, but only H100 relative to
P009. Flow relative-L2 and state MAE are unchanged, consistent with the frozen
flow model; this is not improved flow forecasting.

| H | Pooled total Cd NRMSE P009 | P013 | P015 |
|---|---:|---:|---:|
| 1 | .00545055 | .00752533 | .00986159 |
| 10 | .00665640 | .00994570 | .01130156 |
| 50 | .00702768 | .00842937 | .00809546 |
| 100 | .00558671 | .00963676 | .01048992 |

Do not mix pooled and macro aggregation. H100 macro total-Cd NRMSE is
.00518737 / .00931429 / .01044542 for P009/P013/P015. Strict start0 action-minus-zero
Cd MAE is .01920627 / .01554142 / .02022350. P015 start0 signs8/8 and ordering20/20
pass the endpoint component, not the combined development criteria.

Decision: continue the preapproved unchanged complete evaluation. No PPO,
frozen-test access, checkpoint selection or threshold adjustment. Report actual
dynamic trajectory and force-window results before deciding the next experiment.
