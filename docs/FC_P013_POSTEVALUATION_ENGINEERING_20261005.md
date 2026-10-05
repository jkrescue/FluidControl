# FC-P013 post-evaluation engineering status

This is implementation evidence, not a completed model evaluation or admission.

## Verified before the terminal exists

- Running container observed directly through Docker and systemd. The observation
  is explicitly retrospective execution evidence, not a fabricated prior approval.
  `running_execution_evidence.json` SHA:
  `a106555efbe2e6b6ab1046a94ed05f58d903785b223716cdd7075a73bc432f63`.
  Exact train-only bind sources, read-only settings, image and trainer were captured.
- The terminal auditor checks actual saved tensor digests, unchanged official
  frozen lifting biases, finite aerodynamic tensors, actual AdamW state for all
  28 trained parameter tensors at step 1368, and successful memory-guard exit.
  It reconstructs window identities from the 44 real HDF trajectory shapes and
  verifies the original global sampler-order SHA; it also records HDF hashes.
  The auditor has NOT run against a terminal because training is still active.
- Separate fixed-six diagnostics retain the original P011 numerical function for
  physical H1/free-AR errors, field errors, and tail62 mean/RMS. No selection.
- Dual evaluators now require `--dual-training-config` for the manifest-bound
  training configuration. They continue composing the unchanged evaluation
  `--config`. Previously, comparing the evaluation leaf YAML bytes directly with
  the training resolved-YAML SHA would reject a valid run. Architecture is still
  checked against the dual manifest, normalization stays exactly bound, and both
  config files must be captured in the formal source receipt. No metric or
  sampling change is made by this interface repair.
- 30 relevant CPU tests pass, including negative optimizer-step, frozen-tensor,
  missing-record, nonfinite-metric, guard-failure and config-routing tests.
- Pinned official image CPU imports both evaluators and confirms the original
  evaluation config and the training resolved config specify the same FNO model
  architecture. This is software evidence, not synthetic CFD or model accuracy.

## Remaining work before formal execution

### Additional implementation evidence (06:33 UTC)

The shared runner/validator now has a separate P013 profile and a thin wrapper.
It preserves the original numerical tree and records four explicit adapters:
dual loader, P009 checkpoint-identity helper, field evaluator and force-window
evaluator. The P009 identity helper is required because the older numerical tree
does not accept its `expected_kind` argument; this is checkpoint validation, not
a changed metric. All auditor helper dependencies and the bound training YAML
are included in the frozen-source recipe. Dual flow/model/state identities are
required in step and completion receipts and checked against actual reports.

47 related CPU tests pass, and the actual P013 runner `--dry-run` completes
without GPU evaluation or a terminal model. One old runner text assertion was
already failing before this change (1 failed / 10 passed); it assumed literal
epoch-zero paths rather than the existing multi-profile variables. Its updated
assertions retain the original epoch-zero identity/protocol check through the
current shared arguments. This is not a changed scientific threshold.

The new HDF-order auditor was also exercised on the genuine completed P011A
record sequence: all 44 current real HDF files reproduce 1368 global windows and
order SHA `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`.
This validates its data-index reconstruction; it does not claim that the running
P013 experiment has completed. Additional independent review was requested but
the agent again failed with model-service capacity errors. Root review and CPU
tests do not replace the required independent scientific evaluation.

Items 3–6 below are now implemented and CPU-tested, but still need independent
review and frozen-snapshot execution preflight. No formal GPU run is approved.

1. Independently review/recompute terminal artifacts when available. Produce the
   completion receipt only from completed guard + actual persisted model evidence.
2. Execute fixed-six read-only diagnostics with the exact terminal manifest and
   result SHA. Do not choose another checkpoint using these results.
3. Extend the shared P008/P009/P011 formal runner with a P013 identity profile and
   a thin wrapper. Keep original numerical source commit
   `7216214b545fbbd50b2fb5ed866f231039b06b18`; overlay only the reviewed dual loader
   and evaluator adapters, recording every resulting SHA. Keep the original
   numerical gate scripts and complete config tree unchanged.
4. Bind BOTH checkpoint pairs plus the dual manifest in lineage, step receipts
   and final receipt. The aerodynamic pair is the primary legacy metric identity,
   not a claim that the system contains a single model. Use a consistent container
   checkpoint path matching the dual manifest's aerodynamic directory.
5. Include all candidate-auditor helper dependencies in the immutable snapshot;
   do not silently import current mutable worktree code.
6. Extend the shared validator to verify dual identities while reusing unchanged
   scientific recomputation. Test tampered flow/force/manifest/config identities,
   and retain existing single-model regression tests.
7. Only then authorize the unchanged validation10 H1/10/50/100 stride25 batch4,
   dynamic6 H1/10/50/100 stride1 batch8, force-window6 and development gate suite.
   Frozen tests and PPO remain excluded at this stage.

Formal execution is NOT authorized by this engineering note. The final project
still requires an accepted surrogate, a compatible controller and same-start
real-CFD closed-loop success under the original drag/lift/action constraints.
