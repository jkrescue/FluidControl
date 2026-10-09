# FC-E082 — symmetry-canonical PPO independent terminal review

2026-10-06 UTC (2026-10-07 local). **Engineering completion accepted; physical outcome unknown.**

Actual unit `fluid-control-p064-b-symmetry-canonical-ppo-20261007.service`, invocation `029124675a0d459e82455c59f831ba3b`, independently observed MainPID0/Result success/ExecMainStatus0. Approval `docs/P064_B_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json` SHA256 `1a49d9a72a8dabcb1159b2d04ebcf19f039cbdb70f87ba1e1c4d714d898368c7`.

## Independent saved-evidence checks

- Saved executed specification equals approval. All75 source and192 runtime file hashes, and all six output-artifact hashes checked. Same frozen B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`; FNO tensor digest `b108afe056be173a3f11d5d8df9eef54dee40ca9d528ac834725be5d71e73928` unchanged according to reviewed producer checks and saved evidence. No second model deserialization or forward performed.
- All32768 finite transitions checked in exact four-environment order: H5 episode/truncation clock,24-start deterministic reset order, frame0/frame62 identities,69-dimensional observation metadata, action magnitude/rate,62-sample measured/predicted history accounting.6552 episodes completed; each phase reset counts `[274,273,273,273,273,273]`.
- New coordinate checks on all rows: applied and next orientation in ±1; logged next orientation consistent; physical requested action exactly equals canonical action received times **current** orientation and base requested_omega. Non-reset consecutive steps preserve previous-next/current orientation and the single physical action filter.546 within-transition orientation changes;0 reflection-fixed logged next observations. Pivot/margin/fixed metadata are valid. This does not claim independent recomputation of orientation from every original physical observation, which is not saved in these transition rows. First-reset orientation is supported by reviewed wrapper/reset implementation, not independently reconstructed here.
- All512 optimizer hooks consecutive at eight per512-step rollout,256 PPO epochs.65 finite JSON logger rows. Six reward-component means independently recomputed, maximum rounding difference3.4416913763379853e-15. Initial policy tensor `8ccb4e5cd01328b0c2a34194b20eb235a47c8c2a74ddaa4573eb39443fb46855` equals the preceding seed20261007 experiment; final `398c5ae0d4e87fcab88cf59723214ec0628e75d4856e28c38a579e177e924680` differs.
- Effective highest/no-TF32 recorded, original official checkpoint loading identity preserved. Supervisor returncode0/error null,1178 memory observations minimum MemAvailable119839916032 bytes, actual12GiB/swap0; wall587.3528002190287 seconds. No CFD or optimizer rerun for this audit.

## Final artifact bindings

Root: `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/`.

| File | SHA256 |
|---|---|
| result.json | `44ef56114e17db88077d5f3e7920a2a9031023be74e31d359bd8c66be7441d1e` |
| ppo_final.zip | `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e` |
| vecnormalize.pkl | `1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145` |
| transitions.jsonl | `a32852f161304eab93c3caf1c283b53ed2097e8d23f566a90525e725ae9bd520` |
| source_spec.json | `1701b7d0df4fc298d2ccd9b7ec659b781a7ecd713d03db3bde4742e370fd8d4b` |
| progress.json | `5d2b2a330548c66406647ffb7db06ba8fa40ad140c992e13dddd9d9418601b2f` |
| reset_packets.json | `2b1369d8bfe6e7270e97593cb41433cb1d65357e775c94af24333d663060b2d3` |

The single intended change is the policy-independent coordinate wrapper; B,seed20261007,32768/H5/24-reset/reward and PPO hyperparameters remain fixed. This demonstrates actual finite PPO optimization and consistent recorded action mapping, not physical success, formal surrogate admission, or a proven remedy. E080 negative replication and old successful policy remain unchanged. Any fixed b00 paired CFD evaluation needs separate Lead approval and must use these exact final artifacts and matching coordinate deployment, without reward-based selection or numerical threshold changes.
