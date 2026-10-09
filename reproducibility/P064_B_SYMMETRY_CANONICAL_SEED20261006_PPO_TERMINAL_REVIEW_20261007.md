# FC-E084 — fixed seed20261006 canonical PPO terminal review

Independent saved-evidence review, 2026-10-07. Unit `fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-20261007.service`, invocation `e236b09e33564b0bb4e6aad5c46eff60`, is MainPID=0, SubState=exited, Result=success, ExecMainStatus=0. This review did not load a policy/model, rerun inference, train, or execute CFD.

## Engineering verdict

ACCEPT for binding the completed final policy to a separately approved fixed CFD evaluation; not a physical or surrogate admission.

- All75 bound source and192 runtime file hashes, six result artifact hashes, final persisted approval/source_spec equality and result protocol were checked. The frozen B manifest remains `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`.
- Exactly32768 finite transition rows,256 PPO epochs and512 ordered optimizer hooks, eight hooks at each512-environment-step boundary.6552 completed H5 episodes; every phase reset count is `[274,273,273,273,273,273]`, matching the fixed24-reset panel. Reward six-component means independently recompute within3.969047313034935e-15.
- All recorded canonical actions multiplied by their current orientations equal the physical requested actions. Consecutive non-reset orientation chains, physical single-filter actions, next-orientation flags/pivots/margins and fixed-point rules pass.546 orientation changes and0 fixed observations. Original full physical observations are not retained in these training rows, so this is not an independent recomputation of all initial/reset observation orientations; that limitation is unchanged from E082.
- K1 history/current-frame and reward measured/predicted counts follow the existing H5/62-history contract in all rows. FNO tensor digest `b108afe056be173a3f11d5d8df9eef54dee40ca9d528ac834725be5d71e73928` is unchanged according to the executed tensor check. Precision effective is highest/no CUDA-TF32/no cuDNN-TF32.
- Initial policy tensor digest `6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf` matches the original seed20261006 initialization; final digest `986bf9ef03c1ae4d576d06759aecd5b4067ba3bf6095a9b7a33d044a9d4fac68` differs. Final-only policy, no reward/checkpoint selection.
- Actual wall time600.6985666719847s.1205 memory samples; minimum actual MemAvailable119469228032 bytes. Exact memory.max12884901888/memory.swap.max0; supervisor returncode0/errornull and result SHA agreement. No CFD executed and scientific_admission=false.

## Exact final bindings

Approval `docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_APPROVAL_20261007.json`: `d62cccbcf1d896ed812cca7cac27e8e5df9d1fd7623a84ae1b5478494cabdd39`.

Output prefix `artifacts/p064_b_symmetry_canonical_seed20261006_h5_32768_ppo_20261007/payload/`:

| File | SHA256 |
|---|---|
|result.json|cb1a2f0931fcf70c68802287bdb5e5f894553c0eab271a030a54ff7f2f598336|
|ppo_final.zip|edd616c507798e355776c420974f53c256d78af04170dab77fe63beb1af0baa5|
|vecnormalize.pkl|3035aad7e6ff5dd0a789a3a56bc6029b137221ffc0297cc223d6325825cf9018|
|transitions.jsonl|f01fe5c3752d315a7a63b5076310c9c3edb9443d61f35086e0f238b012f694b4|
|source_spec.json|f8b1f05360f8d47962d01ddc8d27c1bd50d7e736a08602a6c4a7928250b09b55|
|progress.json|995a542e62e74a1ce4dae14f409b22cf91e46989924532fb3b2d38306d8299b6|
|reset_packets.json|2b1369d8bfe6e7270e97593cb41433cb1d65357e775c94af24333d663060b2d3|

The sole intended scientific change from E082 is the predeclared seed20261007→20261006. B model, canonical adapter a55b, H5,24 real resets, reward,32768 budget and PPO hyperparameters remain fixed. E083 physical success is not assigned to this new policy before its own CFD; E080 negative result, early-window failures and full FNO prediction FAIL remain preserved. No automatic CFD launch is authorized by this report.
