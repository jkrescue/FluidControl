# FC-P013 post-evaluation engineering status

This is implementation evidence, not a completed model evaluation or admission.

## Formal dual-checkpoint path preflight — 2026-10-05

Read-only independent review found that the original dev30 diagnostic accepted only `/workspace/checkpoint`, whereas the P013 evaluator correctly records `/workspace/dual/aerodynamic`. The diagnostic now admits the latter only under the explicit `fcp013_independent_force_dual_fno` kind, after CPU validation of the actual dual manifest, aerodynamic role/epoch and all five model/manifest report hashes. Unknown aliases and legacy-kind attempts remain rejected. The runner and posteval validator now select this explicit kind; the diagnostic is an additional declared immutable-source overlay. Original pooled, paired-action and threshold computations are unchanged. Root reran50 CPU tests, including legacy numerical diagnostics and posteval profile/chain tests; all passed after updating the test fixture to include the required fifth overlay. No evaluation or GPU job was run for this patch.

Candidate control plumbing and optional portable compatibility-receipt access paths were committed/pushed as `4481de7`, with108 CPU tests passing. That commit is not execution readiness: the separate canonical raw endpoint-report path adapter remains under independent implementation. Promotion verification must be regenerated as a new receipt under the actual `/workspace` runtime, preserving old host receipts. Original formal reports, gate values and acceptance thresholds must not be edited.

Training r2 and its already frozen terminal auditor remain unchanged. The next formal-evaluation snapshot must include the reviewed diagnostic overlay and bind actual terminal artifacts before GPU execution; earlier four-overlay snapshots remain historical and must not be selected for the upcoming formal suite.

## Real-CFD admission identity checks — 2026-10-05

Independent control implementation adds dual-specific checks to the existing real-CFD runner. Readiness and completed PPO audit must carry identical complete dual identities, paired primary model and field-normalization hashes, plus a SHA-verified actual SB3 VecNormalize artifact. Its flags must be norm_obs=false/norm_reward=false, observation shape69, action shape1 with bounds±0.75, and raw observations must remain unchanged. Explicit host path/SHA and dual identity enter the predeclared lineage; the runner revalidates it under the execution lock before staging CFD. Existing legacy single-model checks and numerical control/reward/action rules are unchanged.

Root reviewed the diff and reran56 CPU tests spanning new dual admission, existing real-CFD admission, canonical dual routing and dual contract; all passed. The implementation agent separately ran33 tests. These are software fixtures, not a policy trained or CFD controlled. The candidate-to-policy/export owner is still implementing P013 plumbing. The inference loop still applies PPO directly to real-CFD physical observations; no online FNO inference was added or claimed.

## Independent control fix and finalizer replacement — 2026-10-05

The canonical PPO trainer now refuses the single-model route when P013/dual identity appears in supplied reports/readiness or actual selected checkpoint metadata. This prevents omission of all dual flags from silently treating the aerodynamic FNO as the flow model. The actual official pinned PhysicsNeMo `load_checkpoint` signature was inspected and supports `metadata_dict`; loaded metadata is rechecked before environment/policy creation. The implementation agent passed30 CPU tests; Root reviewed the diff and reran52 across dual binding, canonical dry-run, candidate launcher, control contract and actual HydroGym stepper fixtures. Legacy single-model behavior remains covered. No PPO was launched.

After the independently reviewed data-hash repair, the current authoritative immutable post-evaluation chain is `artifacts/p013_posteval_chain_2e04cbe0cc18_immutable`, receipt SHA `f9810e70ca55d09d30f67d7112cc6300b5deaa16a1f6c8bb470e15a829775500`. Stage-only verification passed. Root stopped only the old waiting finalizer and started `fluid-control-fcp013-finalizer-r2-datahash-20261005.service`, invocation `e2ac5526abd149bdb38352d185236d65`, using this new audit source and the same immutable finalizer program. Training service and scientific protocol were untouched. The old finalizer source must not be used for completion. Downstream real-CFD dual/normalization enforcement is assigned next and remains pending.

## Independent review and data-byte enforcement — 2026-10-05

A fresh independent review identified a real P2 provenance weakness: terminal audit previously recorded actual HDF hashes without comparing pinned manifest hashes. Root repaired this for base20 (SHA-pinned train split with stem-keyed HDF hashes) and train8/train16 (filename-keyed manifest hashes), with exact file-set checks before shape/order reconstruction. The diagnostic launcher now rechecks all 44 bytes and current per-directory HDF file sets against the terminal audit before GPU work. Running trainer/data were not changed.

An actual read-only hash pass verified all 20+8+16 training HDFs against their pre-existing pinned release maps: all44 matched. Root's focused tests passed23; independent reviewer reran the extended CPU set with34 passes and confirmed the original gap closed. Reviewer additionally requested the post-audit extra-file check, which Root added. This is provenance verification, not scientific admission. The waiting finalizer must be rebound to a newly frozen audited source before training terminates; the old source must not generate final acceptance evidence.

Independent control review also identified optional-dual omission, downstream normalization/dual-binding enforcement, candidate export plumbing and the distinction between surrogate-trained PPO online feedback versus FNO inference inside each online decision. Omission prevention is assigned to a separate agent; the remaining interface work is pending, not implemented. Do not relabel the existing CFD-only policy or online-PPO path as full dual-FNO online execution.

## Worker runtime core preflight — 2026-10-05

Worker has the exact pinned PPO dependency image `sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c`. The clean official HydroGym core (`hydrogym`, LICENSE, pyproject.toml) was archived from commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`; archive SHA matched on Main and Worker: `a8c7dc52953b603843ccf2173a98450f886665eccd7cb3b100f2a9e5fb87a59f`. Core tracked source bytes were 598428, not a bulk image/dataset transfer. Main retains `artifacts/worker_hydrogym_source_preflight_20261005/hydrogym_source.tar`; Worker scratch mount is `/home/USER/workspace/fluid_control_runtime_preflight_20261005/hydrogym`.

A read-only, no-network, no-GPU, 2 GiB/2 CPU container imported HydroGym FlowEnv/PDEBase/TransientSolver, official PhysicsNeMo 2.2.2 and SB3 2.7.1 successfully (`WORKER_CORE_IMPORT_PREFLIGHT_PASS`). This verifies core Python runtime imports only, not Firedrake/Nek/Maia backends, project adapter execution, CUDA inference, PPO or real CFD. No scientific data or result storage moved to Worker. Main training advanced to step360 during this check; it remained active.

## Recovery reproducibility observation — 2026-10-05

Read-only comparison of the first attempt and r2 run logs matched all 34 common logged window identities through update 272. The six logged losses at steps 8..48 were exactly equal. At step56 totals were 0.010853966698050499 (first attempt) versus 0.010854589752852917 (r2); at64 they were 0.018482627347111702 versus 0.018506601452827454. Maximum absolute total-loss difference over these 34 logged updates was 0.01536891981959343. This demonstrates matching logged window order but not bitwise numerical reproducibility. The fixed seed and original default-TF32 precision contract are unchanged; the trainer does not request deterministic algorithms. The cause of divergence has not been isolated. Do not attribute it conclusively to TF32, the later memory fault, or a model-quality change. Do not choose a run by these losses. The interrupted attempt produced no eligible terminal model; r2 still requires the entire original evaluation and later repeated-seed evidence for broader claims.

## Fixed-six diagnostic launcher prepared — 2026-10-05

`scripts/run_fcp013_fixed_diagnostics_spark.sh` is prepared, not yet executed on a candidate. It checks the frozen diagnostic source, unchanged P011 numerical trainer and resolved config, successful r2 completion/audit and actual artifact hashes. It refuses an active training container and requires >=50 GiB MemAvailable and >=30 GiB MemFree before starting. Only the existing train-only mounts, read-only parent and terminal dual candidate are exposed; GPU allocator fraction is 0.15, available-memory floor 20 GiB and timeout 1200 seconds. One fresh output directory is required. The six windows, parent/terminal comparison and metrics are unchanged. Official pinned-image CPU import preflight passed using these exact source mounts; no model forward, diagnostic result, optimizer or PPO was run.

This launcher remains a prepared stage; Root must execute from a frozen copy after successful training, and must verify the guard and diagnostic result before any next-stage approval. Independent review was requested again; availability is not assumed.

## Terminal audit automation approval — 2026-10-05

Lead approves one CPU-only finalizer waiting on the exact r2 systemd invocation `d0138175399a40f487493919a674e1a5`. It must see inactive/dead/success with normal exit code zero, then run the complete frozen candidate auditor, verify state remains terminal and atomically persist an audit plus training-completion receipt. The finalizer cannot launch evaluation, PPO, change models, or restart training. An active unit with Result=success is explicitly not completion. Unknown/missing units and observation timeout fail without restarting. Existing receipts must match exactly and are never overwritten.

Execution budget: <=14600 seconds waiting, CPUQuota=100%, MemoryMax=4 GiB, immutable finalizer and audit source, no GPU. This does not change the experiment or scientific criteria. It removes the manual handoff after training; original fixed-six diagnostics and separately approved formal evaluation remain required. Terminal receipt labels scientific_admission=false and diagnostics/evaluation pending.

## Recovery generation update — 2026-10-05 07:12 UTC

The first training attempt suffered a verified CUDA copy stall and was preserved, not accepted. See `FC_P013_RECOVERY_APPROVAL_20261005.md`. The identical experiment is running in the separate r2 output, observed at 88/1368 updates. No terminal candidate or formal scientific metrics exist yet.

- Live observation for r2: `c6da08418d3e9c6287d50534fc669f560ed8d5d2a7f4237c90892e91655d1e6d`; actual recovery launcher: `9a589802227a67035f3f0c421fb7ccfa1b955180b3cb0b3750e22a4ca00b4c4b`. Original immutable trainer, parent and training approval are unchanged.
- The candidate auditor now accepts only the two explicitly identified attempt roots and their distinct evidence hashes. For r2 it also requires the exact recovery approval and launcher, plus valid physical-free/available-memory and progress-watch samples. Original 1368 persisted optimizer-step, tensor, HDF and scientific protocol checks remain.
- The post-evaluation default points at r2. Source `23a4ec020ed62ba2f406283335a5d0f8dfbd002b` was frozen into `artifacts/p013_posteval_chain_23a4ec020ed6_immutable`; receipt SHA `a426f82bf1877ed1a930e763a62e6f2739b91185356fdf19bd435735fa7507ef`. Stage-only validation passed, preserving original numerical source and four reviewed dual-loading overlays. No evaluation GPU or PPO was launched.
- 18 CPU tests passed across recovery-watch, candidate-audit and formal-profile tests. They establish software checks, not flow or control accuracy.
- Dashboard `e11756c` now reads the r2 systemd invocation; 7 focused tests passed and live local API verified attempt=2, running=true and fresh step32. Existing Chrome tab was reloaded. Historical field images remain historical.
- Worker image transfer completed with matching pinned SHA and no dataset transfer. Do not repeat bulk image transfer concurrently with Main training. The export script remains untracked pending a resource-aware redesign; it is not promoted as a safe training-time tool.

Next: finish r2; independently audit actual terminal files and create a truthful completion receipt; run the six fixed train diagnostics; separately authorize the unchanged formal suite. PPO and real-CFD integration remain gated by scientific accuracy, not these engineering checks.

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

### Canonical PPO dual loading integrated (06:55 UTC)

The canonical PPO entry point now accepts the optional dual manifest, training
configuration and complete post-evaluation receipt with expected hashes. Its
original canonical preflight must already be READY; a partial dual argument set,
failed canonical gate, smoke-mode attempt, single-calibration bypass or absent
new identity VecNormalize output remains BLOCKED. Immediately before execution
the full binding is recomputed and compared with preflight. Official dual loading
then supplies the existing HydroGym stepper without changing PPO, reward,
observations, action constraints or environment transitions. The saved policy
audit includes both checkpoint pairs and the dual receipt identity. The default
single-model path is preserved.

40 related CPU tests pass. The pinned runtime image also imports the new entry
point, PhysicsNeMo, SB3 and HydroGym when used with the existing official-source
mount `.tools/hydrogym` at clean commit
`4ab9854dea3d84e38a59c25e0f5835a00cf8225f`. An initial isolated import omitted this
required source mount and correctly failed with `ModuleNotFoundError`; the retry
used the established launcher mount/PYTHONPATH rather than installing a different
package. No PPO or model inference ran in either import check.

The Worker lacked this exact runtime image. A bounded main-to-Worker image-only
transfer is active under `fluid-control-worker-hydrogym-runtime-transfer-20261005`.
It preserves image ID `sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c`.
No datasets or scientific results are being transferred. Source mounting and
resource readiness must still be checked before using Worker for PPO. The current
PPO guard deliberately checks both CUDA-free and unified MemAvailable; cache-heavy
Spark nodes may fail its conservative startup requirement. This check has not
been bypassed or changed, and no PPO execution is approved by these code changes.

### Supplemental controller identity contract (06:46 UTC)

`src/fluid_control/dual_control_contract.py` now provides a separate, tested
identity check for future canonical PPO integration. It requires the exact dual
manifest, both checkpoint pairs, bound training config/normalization, approved
complete post-evaluation receipt and all required artifact hashes. Each of the
three step receipts must refer to the same complete dual system. The original
development auditor is SHA-pinned (`ca6da0af…bc412`) and recomputes its unchanged
gate; a FAIL cannot pass the helper. Existing canonical endpoint, causal-window
and dynamic-action checks remain separately required.

The module is not yet called by the canonical PPO launcher. No policy or PPO
process has been created. Three software-only tests cover full-file integrity,
wrong/legacy dual identities, original-gate recomputation failure and mismatched
step identities. Their JSON/tensor fixtures are not physical data or results.
Independent agent review was retried and again failed at model-service capacity;
it remains unavailable, not a completed review. Training continues independently.

### Frozen-source and HydroGym interface preflight (06:40 UTC)

Source commit `5eef995ee397d99fbc1ca61b9fce3a7fb79ace26` adds a `--stage-only`
path, which creates and validates the exact immutable evaluation snapshot without
running GPU evaluation. Snapshot directory:
`artifacts/p013_posteval_chain_5eef995ee397_immutable`; receipt SHA:
`db603f5e98f26cbc8d0e4817344a0b1f6ada52ca8fac18f1bba81b181f10a8a6`.
It contains the original numerical tree, explicit reviewed adapter paths, all
candidate-auditor dependencies and the bound training configuration.

The pinned official PhysicsNeMo image imported both evaluators from this frozen
snapshot and loaded the genuine immutable P009 flow parent on CPU through the
official loader and strict P009 identity helper. No forward pass, optimizer,
candidate save, CFD generation, validation inference or PPO ran in this probe.

The new shared validator also recomputed/revalidated all four existing P011A
stages (validation10, dynamic6, force_window, complete) against their original
immutable reports and source tree. All four reuse checks passed; the existing
scientific admission FAIL is unchanged.

Two software-only tests now call the actual `TandemFNOStepper.step` with the dual
adapter. They verify separate flow-delta/four-force routing, physical force
de-normalization, shared inputs, action normalization/rate limiting and rejection
of nonfinite forces before state update. Small synthetic tensors are explicitly
test fixtures, not CFD training data or scientific evidence. The tests do not
construct a full canonical 69D environment, train PPO, or establish control gain.

The existing canonical PPO launcher still loads one checkpoint. Before any dual
PPO execution, its preflight and loading/result identities must be extended to
the accepted dual manifest and both checkpoint pairs while retaining every
endpoint/window/dynamic gate, the 69D observation and causal force prehistory.
The underlying HydroGym stepper requires no alternate update equation. This
interface finding does not authorize PPO before surrogate admission.

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
