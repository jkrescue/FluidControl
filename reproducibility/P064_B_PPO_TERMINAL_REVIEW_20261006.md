# P064 B-bound PPO independent terminal review — 2026-10-06

Verdict: **engineering completion verified; not scientific admission or real-CFD success**.

Actual unit `fluid-control-p064-b-ppo-32768-20261006.service`, invocation `f613395cbf1140549dc60e7b046e0f6b`, was independently observed with MainPID=0, SubState=exited, Result=success and ExecMainStatus=0. Approval SHA256: `ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe`.

## Independent saved-evidence checks

- All 32,768 transition rows checked in exact four-environment order; 69-dimensional observations and terminal observations, five-step truncations, deterministic six-reset cycle per phase, frame0/frame62 identities, action magnitude/rate limits, and measured/predicted reward-history counts agree with the approved protocol. There are 6,552 complete H5 episodes; each phase reports reset counts `[274,273,273,273,273,273]`.
- All 512 optimizer-hook records have consecutive indices and the expected 512-step rollout clock; 256 PPO epochs are recorded. All 65 JSON logger rows and transition numbers are finite. Saved six reward-component means independently recompute from the transition rows.
- All 74 source and 192 runtime file hashes were rechecked. Saved source specification equals the actual approval. All six result-bound output files were independently hashed. B manifest identity is `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`.
- Reviewed executed producer constructs a fresh PPO policy, excludes frozen FNO parameters from its optimizer, verifies finite policy parameters, compares FNO tensor digests before/after, and saves only the final policy. Saved policy tensor digests differ: initial `6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf`, terminal `a444e0af793c0a913abf09264d6338103b269fc859dbb7586139f6942114006a`. Producer reports unchanged FNO tensor digest `b108afe056be173a3f11d5d8df9eef54dee40ca9d528ac834725be5d71e73928`. This independent review verifies saved evidence and source, not a second model deserialization or forward run.
- Precision identity is historical high/TF32 at official load, then highest/no-TF32 during training. Supervisor returncode=0 and error=null. All 1,177 saved memory observations remain above the 22 GiB runtime threshold; minimum MemAvailable is 120,218,787,840 bytes. Cgroup memory limit is 12 GiB and swap limit zero. Producer wall time is 586.946 seconds.

## Exact terminal artifacts

Output: `artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload/`.

| Artifact | SHA256 |
|---|---|
| result.json | `3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3` |
| ppo_final.zip | `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e` |
| vecnormalize.pkl | `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad` |
| transitions.jsonl | `9ecc1df8c98b31cf82a8797080879fd1e52282ba8d2157b23d561ab0911ce88c` |
| source_spec.json | `ebb4e9823657a0864e0c754d60eca20a6ba98f5a8c857a87d021bab1dce06aac` |
| progress.json | `2d0aacece01f38c0e6e57eb4eb92d0915536ae4b63ce192aed9c90892d1ad25b` |
| reset_packets.json | `2b1369d8bfe6e7270e97593cb41433cb1d65357e775c94af24333d663060b2d3` |
| ../supervisor_result.json | `f7e220d8c7d6a52a0b533300511a5f2e254cf16405fab3cb81ac43251d4afe80` |

The initial independent audit command used obsolete source-map key names and stopped with KeyError before numerical checks; correcting the read-only checker to actual `source_files`/`runtime_sources` completed successfully. This was not a training failure or retry.

This verifies genuine fresh PPO optimization against frozen B on the approved train-only reset panel. It does not establish policy improvement in CFD, formal surrogate acceptance, H100 repair, or replacement of the previously successful policy. Any subsequent paired CFD trial must bind this exact final policy/VecNormalize and requires its own execution authorization. No model, GPU, training or CFD was rerun for this review.
