# FC-E084 canonical-coordinate PPO seed replication launch

- Scientific factor: fixed PPO seed `20261006` instead of `20261007`.
- Frozen surrogate: P064-B; FNO weights are not trained by this run.
- Protocol: `32768` PPO timesteps, H5, 24 deterministic reset starts, 69 observations, unchanged reward and SB3 hyperparameters, final policy only.
- Training approval: `docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_APPROVAL_20261007.json` (`d62cccbcf1d896ed812cca7cac27e8e5df9d1fd7623a84ae1b5478494cabdd39`).
- Runner: `b75b37161ec51e086f6839a55c33cd1e86bc6c953d53bea54ebf3b446013dd8c`.
- Supervisor: `516ca485b99175d3a7c3d97a6eb24691bb5638de511d78a8e84453cea9750107`.
- Shared canonical adapter: `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`.
- Unit: `fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-20261007.service`.
- Invocation: `e236b09e33564b0bb4e6aad5c46eff60`.
- Initial MainPID: `3857457`.
- Output: `artifacts/p064_b_symmetry_canonical_seed20261006_h5_32768_ppo_20261007/payload`.
- Resource envelope: 12 GiB memory, no swap, one CPU quota, 256 tasks, 1950 s runtime, 20 s stop grace; inner physical-memory guards require 50 GiB at startup and 22 GiB at runtime while preserving a 20 GiB reserve.
- Final serialized preflight: 75 source files, 192 runtime sources, and 19 actual import origins passed; pre-launch `MemAvailable=120412484 kB`, GPU compute process list empty, and output absent.
- First observed real progress: 2804 complete transition records and 3072/32768 timesteps, six rollout iterations, `train/n_updates=20`, about 55 fps; unit active with swap usage zero.

This launch authorizes only this single PPO training run. It is not a seed scan, does not select a checkpoint by reward, and does not authorize CFD, retry, or replacement of the previously validated policy. Terminal scientific and artifact claims remain pending independent review.
