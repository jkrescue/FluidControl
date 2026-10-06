# P064 B seed20261007 PPO independent terminal review

2026-10-06 UTC. **Engineering completion independently verified; not CFD success or surrogate admission.**

Actual unit `fluid-control-p064-b-seed20261007-ppo-32768-20261006.service`, invocation `38013204a3c94b7fb14da04b78b5fb83`, was independently observed MainPID0, Result=success, ExecMainStatus0. Approval `docs/P064_B_SEED20261007_PPO_APPROVAL_20261006.json` SHA256 `2a8b0bca02154ed5bf0e7b35035e01f0695c1763b183063251ee619db4fc6604`. No model or CFD was rerun during this review.

## Independent saved-evidence checks

- Executed source specification equals the saved approval. Protocol seed is20261007; all other scientific protocol values retain the prior B experiment. All74 source and192 runtime hashes rechecked; all six result-bound artifact hashes match.
- All32768 finite transition rows follow exact four-environment order, H5 truncation, deterministic six-reset sequence per phase, frame0/frame62 identity,69-observation metadata, action amplitude/rate bounds, CFD-time bookkeeping and actual/predicted62-sample reward-history counts. There are6552 completed episodes and each phase reset counts `[274,273,273,273,273,273]`.
- All512 optimizer hooks are consecutive with eight hooks per512-step rollout;256 PPO epochs. All65 JSON progress rows finite. The six reward-component means independently recompute from transitions, maximum rounding difference `2.4147350785597155e-15`.
- Frozen B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`; FNO tensor digest remains `b108afe056be173a3f11d5d8df9eef54dee40ca9d528ac834725be5d71e73928`. Reviewed producer performs unchanged-FNO checks and finite policy checks; independent review checks saved evidence/source, not a second deserialization or forward.
- Fresh seed-specific initial policy tensor digest `8ccb4e5cd01328b0c2a34194b20eb235a47c8c2a74ddaa4573eb39443fb46855` differs from prior seed; final `5021182483ac57c7260f9af3d8d16b8d78a51786e9e72160971d484a1b02f4c9` differs from its initial digest. Only final policy saved; no reward-based checkpoint selection.
- Historical load precision then effective highest/no-TF32 preserved. Supervisor returncode0/error null.1170 memory observations: minimum MemAvailable `119953592320` bytes, above22GiB; actual cgroup12GiB/swap0. Producer wall time `583.4456957019866` seconds.

## Exact final artifacts

Root `artifacts/p064_b_seed20261007_diverse_h5_32768_ppo_20261006/payload/`:

| File | SHA256 |
|---|---|
| result.json | `47dc970756bd608c8a1c3f4744c5b44a1ee87f3524737ea42c4f0fbdf9f82afd` |
| ppo_final.zip | `578ab9561d104976b16af427c8ee8c89f964ce50b4b3dbf9010e24c987ffb4ce` |
| vecnormalize.pkl | `ac756f59cb1743678654cdee01bf57d3a15680cd4ab51583a63c7b07b4006c9c` |
| transitions.jsonl | `12b5f6c2f8e329b6c881fe84ea4190c29e39ae4d2fe3780715dbd5b580dbe2bc` |
| source_spec.json | `ff8d97530b2019a40c48bb68fb27a448cf888d6cfb61ed83858f1fb8a6a23374` |
| progress.json | `674b5701092e80ff3729bfec6d703788f0056749153d3428014c352a9c9d03b2` |
| reset_packets.json | `2b1369d8bfe6e7270e97593cb41433cb1d65357e775c94af24333d663060b2d3` |

The next predeclared comparison is this exact final policy in the same b00 paired800-cycle CFD protocol, only after separate Lead execution approval. Do not select by surrogate reward or scan seeds. Existing trained/initial controls are reused without reruns; two seeds do not establish a robustness distribution. Original physical thresholds and surrogate prediction FAIL remain unchanged.
