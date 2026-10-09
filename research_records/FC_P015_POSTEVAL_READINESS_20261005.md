# FC-P015 post-training evaluation readiness

2026-10-05. Engineering readiness only; no completed P015 candidate or scientific
admission is claimed. The active training source and launch copy were not changed.

## Reviewed transition

1. `finalize_fcp015_training.py` checks the exact retained training invocation,
   its successful exit, externally pinned approval and live observation hashes,
   and the complete independent candidate auditor. Running training returns
   without a completion receipt. A failed/missing/different service is rejected.
2. The P015 post-evaluation profile freezes the original numerical commit
   `7216214b545fbbd50b2fb5ed866f231039b06b18` and the same five reviewed dual-runtime
   overlays. P015 receives its own experiment/status/output identity and exact
   171 updates, 1368 windows, accumulation size 8 checks. Legacy P013 remains
   distinct. Auditor imports are included in the frozen dependency closure.
3. `verify_fcp015_dual_reload.py` actually loads both saved official FNO models on
   CPU and compares their tensor hashes with the terminal result. It performs no
   forward, optimizer step, save, or CUDA initialization. The receipt binds the
   model/state files, manifest, configuration, training result and runtime source.
   Its required image field is a requirement, not observation: Lead must record
   the actual Docker image and command separately at execution.
4. A separate formal execution approval must bind the actual reload receipt and
   candidate completion. The runner also independently checks the training
   service's retained successful terminal state. No automatic PPO is permitted.
5. Run unchanged validation10 H1/10/50/100 stride25 batch4; dynamic6 at the same
   horizons stride1 batch8; six force windows; and the original combined gate.
   A component PASS cannot substitute for the combined result.

## Checks completed before terminal training

Root ran 76 CPU tests covering P015 profile, reload verifier, candidate auditor
and finalizer. An initial Root command used an incorrect test filename and ran
zero tests; the corrected command produced the 76-test result. Independent
evaluation review additionally exercised legacy profiles and isolated frozen
imports. These are software checks, not observed model reload or accuracy.

The actual finalizer invocation against the live training service returned
`FC_P015_TRAINING_RUNNING_NO_COMPLETION_RECEIPT`, as required. No completion or
reload result has been manufactured while training is active.

## Next decision

After training terminates, perform the checks above and record the measured
formal result. If rejected, preserve the negative evidence and propose the next
testable intervention. If accepted, train a compatible new controller and then
perform same-start real OpenFOAM feedback validation under the existing drag,
lift and action constraints. Neither outcome changes the final project goal.
