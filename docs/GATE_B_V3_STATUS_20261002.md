# Gate-B v3: independent-model replication and long-horizon stability

Status at 2026-10-02 15:35 UTC: the two 30-epoch PhysicsNeMo 2.2.2 FNO
models finished training on the same train/validation split with independent
random seeds. The frozen five-case CFD test evaluation has produced the
observed-action panel; the full action-counterfactual and phase audits are
still running. **Neither model has passed Gate B.**

| Model | Seed | 1-step total-drag NRMSE | 10-step | 50-step | 100-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary one-step FNO | 20261002 | 2.919% | 5.744% | 18.954% | 28.773% |
| Worker one-step FNO | 20261005 | 2.028% | 7.920% | 34.524% | 83.948% |

The observed-action panels use five real OpenFOAM histories, including the
fresh, untouched `expanded_test_05`. They are not control-benefit results.
The model with the smaller one-step error has much larger 100-step error:
checkpoint selection by one-step validation alone does not establish
autoregressive stability. Qualitative CFD-versus-FNO wake plots show the
long-horizon prediction developing high-frequency field artifacts; those
plots are diagnostic, not a substitute for the force-error gate.

The original Gate-B limit remains 100-step total-drag NRMSE at or below 10%
on the frozen test set, plus the predeclared stability, action-sensitivity,
and independent shedding-phase checks. No threshold or test split is changed
after seeing these results. The dataset has a documented first-frame legacy
force-label offset in 32 inherited trajectories; see
`docs/GATE_B_AUGMENTATION_20261002.md` for the exact provenance and count.

## Next predeclared experiment

Both seeds are independently fine-tuned using the same 10-step,
10-epoch rollout-loss recipe on the v3 **training** split, with checkpoint
selection on the v3 **validation** split. The worker is running seed 20261005
and keeps no test data. The primary seed 20261002 job is scheduled to start
only after the two one-step test audits finish, so its GPU capacity can be
used without disrupting the frozen audits. The unchanged Gate-B suite then
runs on the primary Spark for each model. All results remain on the primary;
the worker is a temporary compute node.

If these replications still miss Gate B, the next step is to diagnose
long-horizon error and train/validation coverage. CEM, PPO/HydroGym and
claims of drag reduction remain gated. A future policy must additionally be
checked by phase-matched paired OpenFOAM closed-loop runs, not by surrogate
reward alone.
