# FC-P028 training terminal independent review

Scope: read-only unit, saved container evidence, logs and small JSON metadata. No checkpoint payload, HDF, official model loading, or new experiment was performed by this reviewer.

## Actual terminal and provenance

- Unit: `fluid-control-fcp028-flow-train-20261006.service`; invocation `c46c60f3c2634802b2646bb094f9d201`. Independently observed `MainPID=0`, `ActiveState=active`, `SubState=exited`, `Result=success`, `ExecMainStatus=0`.
- Training root: `artifacts/fcp028_flow_training_20261006`; payload is its `payload/` directory.
- Actual container: `a41bbd6eb258faa4f7c3258bcc28dc1d7a0af5a153dab3ea7c85594f463d8191`; official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`. Saved terminal inspect reports exit0, no OOM, PID0; started `2026-10-06T01:27:46.198812782Z`, finished `2026-10-06T01:38:07.754985875Z`.
- Result SHA256: `74bc0d491d82da8c3b897a330e1397ac7db2e92465221801ae1868a57114840d`; status `FC_P028_TRAINING_COMPLETE_NOT_ADMISSION`.
- Training protocol SHA256: `fd3fffdb38533b50ee3e10adc36e7e3cfab2b95c15da61e620b8fc950052fdad`. Its actual JSON and digest agree with result, manifest and result-bound approved source specification.
- Source specification binds the previously independently verified 421-file training source manifest `72c0513e03c154b032f162d9c4c3cbd5ddfad7c478694f4e652aeb949759012c` and launcher `62ff300395785291de790e26e5c60d072a89fa43f165f2711747e4b0afd2b8e0`.
- Root subsequently reports successful terminal auditor receipt `candidate_audit.json`, SHA256 `dd3390d0ac09f8f8e4673ed8eb48d2fbe47293a1dd1689a7abae29972ddddaea`. This report does not portray Root's checkpoint-level audit as an additional independent model reload by this reviewer.

## Counts, objective and identity

Independent log recount: 1368 `training_window_complete` events and exactly sequential updates1–171. Result contains171 groups of8 windows. All1368 rows have10 finite per-step losses, complete-ten-step-gradient declarations and total losses consistent with the ten-step mean. Protocol remains flow-only H10 masked normalized state MSE, original H100 window inventory/order, raw8 gradient accumulation, clip1, AdamW learning rate1e-5, seed20261003. Order digest is `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`; no validation or frozen-test access is declared.

The per-record `optimizer_step_performed=false` is inherited from the accumulation helper, which deliberately performs no optimizer step. The enclosing trainer performs `optimizer_update` and validates optimizer step state; the successful terminal audit checks the171 actual steps. This helper flag must not be interpreted as zero training updates, and the original result is preserved unchanged.

Manifest identifies `FC_P028_FLOW_ROLLOUT_REPAIR`, trained flow epoch1 and frozen P026-K1 aerodynamic epoch1. Recorded flow tensor digest changes from `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb` to `fac5f2984ca47ab041273a4ed32201b68accfd34005bb3a3134ded38d34c5406`; aerodynamic tensor digest remains `b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb`. Frozen aerodynamic model/state byte identities remain `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5` and `ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3`. These are metadata consistency checks here, not independent tensor recomputation.

Unused force-head rows are reported preserved; unused force outputs are explicitly not claimed preserved because the shared flow representation changes. First/last group losses use different windows and cannot establish matched scientific improvement.

## Resource evidence and conclusion

- Saved host watcher:307 samples, minimum MemFree20.834857940673828GiB and MemAvailable108.46442413330078GiB.
- Result internal checks:15221 samples, minimum MemFree/CUDAfree22.334442138671875GiB and MemAvailable108.89851760864258GiB.
- Outer GPU guard:310 samples, exit0, minimum CUDAfree21.236846923828125GiB and MemAvailable108.7003173828125GiB.

All recorded minima remain above the unchanged20GiB floor. Differences reflect sampling locations and times, not interchangeable measurements. Operational training completion is supported; checkpoint tensors have not independently been officially reloaded by this reviewer. Official CPU dual reload and unchanged formal evaluation remain required. No accepted surrogate, physical-control success, PPO authorization or threshold relaxation follows from this training result.

## Subsequent actual official CPU reload

The required official reload has now completed and its saved evidence was independently reviewed without rereading checkpoint payloads. Unit `fluid-control-fcp028-official-cpu-reload-20261006`, invocation `5edc5a00bd14492c9203a4b7d4d676b3`, is active/exited, PID0, success/status0. Receipt `artifacts/fcp028_official_cpu_reload_20261006/dual_reload_receipt.json` SHA256 is `685d55a9a2116d1554f14c26e54ce2d2913a4f9c47553c70407f3c4e25d3aecd`, status `FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION`.

Its audit digest matches actual `dd3390d0…dddadaea` receipt bytes (full digest above); complete seven-file candidate map and both tensor digests exactly match that audit. The nine runtime source hashes were independently reread and match CPU manifest `2899934d3f822749e1821088ae8dcb0e369e488fa8e170ffa03143219d26ad25`. Training config, protocol, manifest and result identities match the training evidence. Official source pins are FNO `e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9` and checkpoint API `0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e`.

Saved actual container inspect identifies `f8292f9e759a0b0f9133b7e167336bcb105c0165ad2dbc91781091540e1abf37`, the same official b40 image, runc, no GPU device requests/devices, network none, read-only root, memory/swap limits both8GiB. Exact source/config/candidate/audit mounts are read-only; only reload output is writable. Its command invokes the frozen verifier with actual audit/source-manifest SHA arguments. Started `2026-10-06T01:41:36.188160975Z`, finished `2026-10-06T01:41:47.347929655Z`, exit0/noOOM/PID0. Six host samples have minimum MemFree32.503631591796875GiB and MemAvailable113.77254867553711GiB.

The actual official CPU process verified the pair and tensor identities. This reviewer audited its receipt, source bytes and runtime evidence rather than launching a second reload. Receipt explicitly records no forward, optimizer, save, GPU, scientific admission or PPO authorization. Unchanged formal evaluation remains outstanding at this review point.
