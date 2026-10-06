# FC-E059: independent b01 projected-policy terminal review

Status: actual paired CFD completed; original primary physical criteria met in this second observed phase. This is not statistical independence, general robustness, or K1 H100 surrogate admission.

The reviewed R2 unit `fluid-control-exploratory-projected-32768-ppo-b01-long-cfd-r2-20261006.service`, invocation `9ef43959e065431490bd4725fa8fb7fe`, ended with MainPID=0, SubState=exited and ExecMainStatus=0. Its approval SHA256 is `790bb12fae2f5df729efda98ef59e5d99f75521d220cb9b39e8caf04808e4c59`; immutable driver SHA256 is `8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741`.

Result: `artifacts/exploratory_projected_32768_ppo_b01_long_cfd_20261006/result.json`, SHA256 `961e1bc3ccb7a9f9dae4b54e9f8233c906507c794cff9a497d806391e0fc5c37`.

The first invocation `dfa8ba1412e34522a1ed1385e28b2df7` failed at the CLI guard because `--execute` was omitted. It performed no model/CFD execution. R2 was separately authorized with the same source/spec and the required flag; the first failure remains preserved.

## Independent saved-evidence checks

No model inference or HDF reopening was performed for this review. All 3,200 saved raw force-file hashes were verified. All six predeclared windows were independently reconstructed from coefficient files, using Cd column 1 and Cl column 4, with explicit open-left/inclusive endpoint handling. Counts, total drag means, rear lift means, centered RMS and absolute peaks agreed with saved results within 1e-12. All 800 recorded requests satisfy `0.5*(pi(o)-pi(Ro))`, followed by exactly the existing amplitude/rate filter.

| Window | Samples/branch | Drag reduction | Rear-Cl centered RMS / paired zero | Absolute rear-Cl mean / paired-zero RMS |
|---|---:|---:|---:|---:|
| (130,142.4] | 2,480 | 3.613462% | 0.877994216 | 0.065927366 |
| (130,136.2] | 1,240 | 4.220682% | 0.923966665 | 0.127807798 |
| (136.2,142.4] | 1,240 | 3.006181% | 0.824856052 | 0.004051053 |
| Primary (150,210] | 12,000 | 3.923637% | 0.815785507 | 0.027300222 |
| Companion [150,210] | 12,001 | 3.923540% | 0.815787044 | 0.027212543 |
| Full (130,210] | 16,000 | 3.815315% | 0.825013365 | 0.007191961 |

Primary total Cd was 2.210053425769 versus zero 2.300309214919. Rear-Cl mean was -0.032009424597; centered RMS 0.956505963021 versus 1.172496880968; absolute peak 1.350517614 versus 1.647302403. The unchanged primary criteria—at least 2% drag reduction, RMS ratio at most 1.05, mean-bias ratio at most 0.10—are all satisfied. The first 6.2-D/U secondary window still fails the 10% mean-bias criterion (12.78%); a 20% sensitivity reference is not a threshold change.

All 1,600 solver segments reported 20 steps and clean solver completion. Maximum Courant number was 0.245443056 and maximum absolute per-step global continuity error 1.66462556e-12. Maximum applied |omega| was 0.655404120684; maximum increment 0.096280708909; neither saturation nor rate limiting occurred. Minimum recorded MemAvailable was 117,067,710,464 bytes.

The two exact owned containers were independently absent from `docker ps -aq` after completion. Their retained terminal inspections report OOMKilled=false and sleeping-container cleanup exit 137, not solver failure:

- `b6df0829ca7d35429752e74d2a1fd311444b357b59dde78ba98eaad672552f3b`; inspection SHA256 `2a92be4c04e411e6c21f249ebd458745753e3ed56436af1e018c00cf9a9c6ea2`.
- `05827c2d689f506097c63428245e8b434211ceeb0ccf8bd5f8e4e791b0072bb4`; inspection SHA256 `0fe71fa5ec28158ff26bf0ed616d034703a5d427eb2ae32089997972eee5f51a`.

## Interpretation and timing

The unchanged projected policy reproduced primary physical benefit from b00 (3.89198% drag reduction, RMS ratio 0.815696, bias ratio 0.011385) in b01 (3.92364%, 0.815786, 0.027300). These are two observed restart phases of the same problem, not independent statistical replicates or a broad generalization study. b01 has historical validation exposure. Its zero branch is phase-specific; it is not asserted byte-identical to b00's zero branch.

Measured wall time was 1,125.320354 seconds for 800 paired cycles, averaging 1.406650 seconds per cycle including two CFD branches, observations, file I/O and cleanup. Each control interval is 0.1 D/U; 80 D/U were simulated overall. This demonstrates causal online feedback in the simulation, not real-time physical deployment. The saved rows do not isolate policy inference latency, so no separate inference-throughput claim is made.

No policy retraining, MPC substitution, new reward tuning or threshold relaxation occurred. The existing K1 H100 formal failure remains unchanged; the prospective H1–H5 confirmation is a separate evidence stream.
