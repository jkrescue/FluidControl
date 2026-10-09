# b00 long-controlled trajectory augmentation — preparation only

No conversion, model loading, training, or CFD execution is authorized by this plan. Keep the official existing FNO architecture and original scientific criteria. Target the demonstrated **one-step force readout error on real controlled states**, not relabel H100 failure as a pass.

## Actual saved inventory (read-only file metadata)

Source root: `artifacts/exploratory_projected_32768_ppo_long_cfd_20261006/`.
`case_mpc` is the historical directory name for **projected PPO**, not an MPC controller. Exactly 801 expected times `148 + 0.1*i`, i=0..800, have both regular non-symlink U/p files: 1,602 files, 680,971,694 bytes, no missing expected files. Paired `case_zero` also has all 801 U/p pairs, 680,359,658 bytes. This inventory checks existence/size, not a fresh full-content hash or field-quality scan.

Existing conversion `artifacts/projected_policy_h1_h5_conversion_20261006_r2/result.json`, SHA `a22c3aa67509e9b3a342071398ae85da2ce4e07c74a3cbbd87e2a493c2b248bf`, binds 16 official-reader-verified mini-HDF files, 96 frames across two branches. **Only 48 controlled frames** (starts0,100,…700, each six frames) can be reused for this one-trajectory addition; the other48 are zero-branch data and are not silently added. Thus753 controlled frames remain to convert. All16 HDF currently total40,977,408 bytes; linear storage planning for801 similar frames is ~342MB, not a compression guarantee. Packets currently33,457,302 bytes total. Avoid copying the complete case; copy only selected U/p plus necessary constant/system metadata into an exclusive conversion workspace. Full801 U/p copy is ~0.681GB; budget several GB for transient VTU/packets rather than duplicating full historical solver outputs.

## Minimal data change and reuse

Reuse the reviewed conversion source and unchanged Curator sampler, then the existing official HDF5Reader. No new resampler/API. Reuse48 packets only after checking their original-source, sampler, grid/mask, and runtime bindings against the new full-trajectory manifest. Bind remaining source U/p hashes before conversion; preserve original case read-only. Produce one801-frame train HDF with original physical u/v/p, common x/y/mask, actual sampled times, actual four-force coefficients and recorded applied omega. Keep VTK float32 time rounding explicit; action input is current omega_i and next omega_i+1 under the recorded linear ramp. Force target is endpoint i+1. Reuse train-only normalization f1b460… unchanged; do not refit it on this addition.

All b00 frames belong to **train as one trajectory**, with no random-frame test split. The eight opened replay origins become development diagnostics and cannot remain independent tests. b01/b03 trajectories already opened under the same policy are development validation only; do not select training examples or checkpoints by their errors. Future prospective physical confirmation may use predeclared b07, which has not run this controller; its fixed-action H5 data were already opened, so it is a new controller rollout, not an entirely unseen phase.

## Proposed single-factor experiment (requires separate approval)

Compare two fresh, identical initializations from frozen K1: A original train44; B original train44 plus the one b00 controlled trajectory. Same official architecture, force/flow role trainability, optimizer, seed, loss, precision, sampler rules and **same fixed number of optimizer updates/total windows** in both arms. Predeclare a trajectory-balanced sampling schedule before training so B substitutes a fixed fraction of windows rather than gaining extra optimization. Do not compare B's extra training only against untouched K1 and call that a data-only effect. The exact budget and trainability/loss must be copied from a chosen existing reviewed training protocol before execution; this plan does not invent them or silently combine a new H1 loss with data augmentation.

Primary diagnostic: real-input H1 four-force errors and totalCd (sum signed component errors before absolute value), plus persistence, on all predeclared development origins; H5 free-AR and field errors secondary. Report base train44 retention separately. H1 is evaluated before any predicted field feeds back, separating readout error from rollout accumulation. If controlled H1 remains poor after added coverage, coverage alone is not an adequate explanation. Improvement on b00 training data alone is not generalization, and b01/b03 correlation is not statistical independence. Preserve all old failures, including H100.

## Next executable step

Prepare a bounded source-preserving801-frame converter by extending the already reviewed fixed-selection converter, with fixed manifest, exact lifecycle cleanup, memory/time caps and official-reader roundtrip tests. Execute only after source/resource approval. No new CFD data generation is needed for this first augmentation; no new GPU inference or training is performed by this inventory/plan.
