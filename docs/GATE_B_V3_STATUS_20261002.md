# Gate-B v3: independent-model replication and long-horizon stability

Status at 2026-10-02 16:05 UTC: the two 30-epoch PhysicsNeMo 2.2.2 FNO
models finished training on the same train/validation split with independent
random seeds. Their complete frozen five-case CFD, action-counterfactual and
independent-phase audits have finished. **Both failed Gate B.**

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

| Full Gate-B audit | Primary seed 20261002 | Worker seed 20261005 |
| --- | ---: | ---: |
| Five-case 100-step total-drag NRMSE | 28.77% | 83.95% |
| Independent-phase 100-step NRMSE | 16.49% | 141.68% |
| Fresh test 05 alone, 100-step NRMSE | 29.87% | 93.29% |
| Legacy four-case NRMSE, separately | 28.50% | 81.61% |
| Failed audit checks | 2 (both NRMSE) | 5 (NRMSE and persistence) |

The primary seed passed the finite-rollout, persistence and action-sensitivity
checks but missed both 10% accuracy limits. The worker seed also failed to
beat persistence at the 100-step observed-action horizon and at the 50/100
independent-phase horizons. This is genuine seed sensitivity in long-horizon
behavior, not an approval to choose a seed by frozen-test performance.

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
only after the two one-step test audits finish; it started at 15:45 UTC.
The unchanged Gate-B suite then
runs on the primary Spark for each model. All results remain on the primary;
the worker is a temporary compute node.

If these replications still miss Gate B, the next step is to diagnose
long-horizon error and train/validation coverage. CEM, PPO/HydroGym and
claims of drag reduction remain gated. A future policy must additionally be
checked by phase-matched paired OpenFOAM closed-loop runs, not by surrogate
reward alone.

## Interim rollout update, 2026-10-02 16:30 UTC

The worker seed-20261005 ten-step/ten-epoch fine-tune completed with a
validation-selected checkpoint (epoch 10, ten-step selection score 0.020547).
Its **observed-action preliminary** frozen five-case 100-step total-drag
NRMSE is 67.143%, versus 83.948% for its one-step parent. The five per-case
100-step errors range from 55.597% to 79.226%, while one-step errors range
from 0.927% to 3.470%. This pattern indicates a broad long-horizon rollout
problem, not a single exceptional test action. The zero/sign-flip/shuffle,
independent-phase, legacy-four and full Gate-B audits are still running; the
67.143% figure is **not** a completed gate decision or a control benefit.

The primary seed-20261002 ten-step replication is training. The worker has
also started a predeclared 20-step/ten-epoch rollout ablation initialized
from its ten-step best checkpoint. Its model and all final evaluations will
be copied back to the primary Spark; no canonical data are stored on the
worker. A separate validation-split 100-step diagnostic is running to test
whether the failure is generic autoregressive drift. Active-learning CFD
acquisition remains conditional: this preliminary all-case drift does not
yet justify generating additional OpenFOAM trajectories.

The independent **validation**-split diagnostic then completed with
100-step total-drag NRMSE 50.950% (1/10/50-step: 1.406% / 4.977% /
22.688%). Its deterioration across rollout length parallels the frozen
test trend, so the present evidence supports autoregressive instability
rather than a test-only coverage gap. The active-learning branch is on hold
while the predeclared longer-rollout ablation runs; no extra CFD labels were
generated, and this validation result does not relax Gate B.

The worker ten-step model's **complete** frozen Gate-B audit subsequently
returned `GATE_B_NEEDS_MULTISTEP_RETRAINING`. Its five-case 100-step NRMSE
is 67.143% and independent-phase NRMSE is 94.889%, both versus the locked
10% maximum. It also failed the 100-step observed-action persistence check
and the independent-phase 50/100-step persistence checks (five failed
checks total). The test-only preliminary result above has therefore been
superseded by a full failed audit. This does not select or reject the
still-running primary-seed model; the seeds remain separate replications.

On the validation split at 100 steps, rear-cylinder drag MAE is 1.0318,
versus front-cylinder drag MAE 0.0143; the combined drag MAE is 1.0211.
The error is concentrated in the controlled rear wake/force channel, while
the front-cylinder contribution remains comparatively stable. The model's
validation total-drag MAE only narrowly beats persistence (1.0211 versus
1.0685), so an apparent gain in short-horizon force accuracy cannot yet
support reliable control optimization.

The worker ten-step finalizer has now completed all transfers, checksums,
frozen evaluations and the separate legacy-four comparison. The legacy-four
100-step NRMSE is 64.122%; fresh untouched test 05 alone is 79.226%.
Neither the new fifth case nor a changed test composition explains the
failure. The formal five-case result remains 67.143% and failed.

## Primary ten-step replication and rear-drag loss ablation

The independent primary seed-20261002 ten-step fine-tune completed ten
epochs. Its observed-action frozen five-case 100-step **preliminary** NRMSE
is 21.699%, improved from its one-step parent's 28.773% but still above
the locked 10% maximum; the complete action/phase Gate-B suite continues.
Its separate validation-split 100-step NRMSE is 18.927% (1/10/50-step:
2.350% / 3.870% / 11.460%). Thus this seed has substantially better
long-horizon behavior than worker seed-20261005, but still has a real
validation as well as test deficit.

Using the validation-only rear-force diagnosis, a controlled loss ablation
was prepared on the **same official PhysicsNeMo FNO architecture**: normalized
force-channel loss weights `[1, 1, 4, 1]` for front Cd/Cl and rear Cd/Cl.
The frozen test data and Gate-B thresholds are unchanged. Three unit tests,
Hydra composition and an isolated one-batch GPU smoke passed in the pinned
2.2.2 container. A ten-epoch run initialized from the primary ten-step best
checkpoint has started under its own artifact directory; its eventual
result is pending, not a control-benefit claim.

The worker 20-step ablation's epoch-5 checkpoint was evaluated **only on
validation** while epoch 6–10 training continues. Its 100-step total-drag
NRMSE is 54.725% (1/10/50-step: 1.468% / 5.280% / 24.268%), versus
50.950% for the worker's final ten-step model on the same validation split.
Thus the midway longer-rollout checkpoint has not yet improved the
governing horizon. The first diagnostic container attempt failed before
inference because the worker UID has no container passwd entry; rerunning
with the official runner's `USER`/`LOGNAME` environment succeeded.
Both logs and the result were copied checksum-identically to
`artifacts/distributed_runs/gateb_aug_v3_h20_epoch5_validation_20261002/`
on the primary Spark. No worker test data were used.
