# P026 K4 terminal training review — integrity only

Independent review verified saved JSON and raw file bytes, without reloading
models, rereading HDF data, rerunning the auditor, or launching evaluation.
The subsequent official CPU dual reload is verified below; formal evaluation
is not established by this review. No surrogate admission or PPO authorization follows.

## Terminal and provenance

Training unit `fluid-control-fcp026-history-k4-20261006.service`, invocation
`eee5a6fbad40411cac2f05e00520b079`, is retained active/exited, success,
ExecMainCode1/status0, MainPID0. Terminal-audit unit
`fluid-control-fcp026-k4-terminal-audit-20261006.service`, invocation
`45cb135ff6664985b36ac5ab649f1851`, has the same successful terminal properties.
Saved runtime container79a39768f1398bc5ff9d1085f027a4b1802978120107ca5f9008ffdd300a75e3
binds official imageb40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e.

Audit `artifacts/fcp026_history_training_k4_20261005/candidate_audit.json` SHA:
`423ad58a3d441d26f174174bc68824a59ccd453b2e49f0577888530a81083b0b`.
Status is `FC_P026_K4_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION`.
All seven candidate-file hashes and six execution-evidence hashes were
independently recomputed and match the audit. Auditorfc15ad1d…fe937ab is the
reviewed exact-unit compatibility revision in the independently checked
416-file runtime (manifest2417a44d…2fbb889). Original numerical gates are unchanged.

## Reload input map

Paths below are relative to the training root, retaining the literal candidate/
prefix. A candidate-relative consumer must strip only that prefix.

| File | SHA-256 |
|---|---|
| candidate/aerodynamic/FNO.0.1.mdlus | 01275829b8cbb894eb940c4dafa0636913359f01eed164a5ab39ffba1819dc9c |
| candidate/aerodynamic/checkpoint.0.1.pt | b9e0bbc3529e8cfe715f89da16edb5a90e13d47ab43258dd23f62e26493cb172 |
| candidate/dual_model_manifest.json | 9d1fb9bd61e125451c5c8be802d447077928174e047364b1ef34ba64e2ae1b5e |
| candidate/flow/FNO.0.0.mdlus | dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31 |
| candidate/flow/checkpoint.0.0.pt | 4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e |
| candidate/result.json | 7c71e0148e3d7f3ce446a25e84bdcfd1a116f20b17d56f76d174f80aa078c169 |
| candidate/training_protocol.json | 72b3638c4f0fbad687bcc4b216365e8c68935611778f7be562386abaa1db7a3d |

## Saved training evidence and diagnostics

The result has171 sequential updates, eight windows per update,1368 consumed
windows. All result float leaves are finite. Protocol/order/tensor digest
fields agree with the audit; orderSHA177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f.
Frozen-flow digest89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb;
terminal aerodynamic digestd78d4cc11949d7fab6ce9371ed574fac66466651e3078562b280b7b900218e9e.
These are consistency checks of the recorded tensor audit, not a second model
deserialization. Audit records28 changed aerodynamic tensors; role adapter
fresh reload remains false. Trainer records its own official fresh reload true.
Initial H1/AR replay maximum differences are both0.

All four fixed-panel aggregates (0/456/912/1368 windows) were independently
recomputed from saved rows within the reviewed four-ULP aggregation allowance.
Initial→terminal six-window objective: H1 0.003522910670→0.003492104637;
AR 0.008851685920→0.008829930414. Five-nonzero-window macro statistics change:

| Domain | Bias MSE | Centered residual MSE | Absolute RMS amplitude error |
|---|---:|---:|---:|
| H1 | −0.788831% | −0.529794% | −0.596194% |
| AR | −0.704218% | −0.234289% | −0.113670% |

These small descriptive changes are training-window diagnostics, not held-out
performance, a significance claim, or admission. No formal K1/K4 comparison
is established here.

## Resources and limitations

Independent JSON minima agree with the audit: internal155572 samples,
MemFree20.803394 GiB/MemAvailable105.568184 GiB; host2626 samples,
MemFree21.006046 GiB/MemAvailable105.671360 GiB. Guard reports2721 samples,
exit0, minimum CUDA-free21.071632 GiB/MemAvailable105.725796 GiB.
Twelve separately approved scoped cache-advice passes are preserved in the
maintenance ledger; they are operational interventions, not training changes.
Official CPU dual reload requires its own actual execution receipt before
separate formal approval. Training integrity does not establish scientific success.

## Subsequent actual official CPU dual reload

Retained container `c84f3e5f678bed7f07412a878b6313a4bc970e9f39a6c40e50182b2e950df24e`
(`fcp026-k4-official-cpu-reload-20261006`) independently inspected: official
b40d image, exited0/noOOM, start2026-10-05T23:33:39.710901294Z,
finish23:33:51.851677674Z. Isolation is runc, CPU2, memory8GiB/swaplimit8GiB,
read-only root/networknone/user1000:1000, no GPU device requests/devices,
CUDA_VISIBLE_DEVICES empty/NVIDIA_VISIBLE_DEVICES void. Source, training root
and config mounts are read-only; only the exclusive reload output is writable.
Actual timeout300 command binds K4, audit423ad58a…083b0b, runtime manifest
2417a44d…2fbb889 and the frozen official verifier.

Receipt `official_cpu_reload_20261006/dual_reload_receipt.json` SHA:
`491d6e4e8868edd0c0a222ceb1e1ed5cc1c5b2c3a8b88f0f4a053895aed1a729`.
Its status is `FC_P026_K4_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION`.
All416 source-map entries match the independently verified freeze; all seven
candidate hashes, audit/result/protocol/manifest bindings and both tensor
digests agree with the terminal audit. Official FNO/checkpoint source pins
match the reviewed e64eb9be…83c71a9/0d26a622…2f78e. Receipt confirms CPU
dual reload, without forward, optimizer creation, model saving or GPU use.
Scientific admission and PPO authorization remain false. No reload was rerun
by this independent review; formal evaluation requires separate approval.
