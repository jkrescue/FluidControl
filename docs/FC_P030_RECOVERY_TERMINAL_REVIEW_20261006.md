# FC-P030 r2 independent terminal review

Status: diagnostic execution COMPLETE, NOT scientific admission. Original P029 formal rejection remains authoritative. This review used existing JSON/source only, with no HDF/model reread or GPU computation.

## Identity and resources

- Unit `fluid-control-fcp030-train-horizon-r2-20261006.service`, invocation `4b89d3cb85c540448eeebd8ef5c7c3b3`: independently observed active/exited, MainPID=0, Result=success, ExecMainStatus=0.
- Approval SHA `6c4edae1e955268dcae718c4eb2f6b24a216b8961189d56a09f4d4da7c091378`; v3 source manifest `73ac42127e0ace741c675cb7a5a53a6339171565462d0771c11665124d0f08e6`. Independently rehashed all 17 frozen source members and every result source binding.
- K1 manifest `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`; P029 manifest `72b52ff2b6c702c9d100cc5fbade0952875289d25e826bb6b88020cb042d9b16`. Small manifest files were rehashed; large checkpoint bytes were not reread by this reviewer. Execution uses the reviewed official role loader, common frozen K1 aero and pinned normalization/config.
- Actual container `6dbad92751fb3a943f05827479669908c29eec3611726fdca07207400ace6539`, official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`; created/terminal identities match. 04:23:01.838755865–04:25:41.102012050 UTC, exit0, OOMKilled=false. Diagnostic elapsed 152.5683 s.
- Outer 79 samples: minimum MemFree 22,928,797,696 bytes; MemAvailable 119,176,384,512 bytes. Inner guard exit0, 79 samples, minimum CUDA free 21.34380340576172 GiB and MemAvailable 110.97858810424805 GiB. No recorded floor breach/OOM; sampled observations are not a proof of every unsampled instant.

The first failed v2 attempt and its report remain preserved. No optimizer, model save, validation/frozen access or scientific admission is asserted by the successful result.

## Evidence and independent arithmetic

`selection.json` contains the fixed 44 start0/H100 origins, family20/8/16 and canonical phase15/15/7/7. Its exact approval binding agrees with `raw_records.json`; raw rows and records equal those used in result. Both arms contain 44×100 points and report 4400 flow plus 4400 aerodynamic evaluations each. All 44 lead1 four-force predictions are exactly identical across arms.

Recomputed the entire summary from raw records using the frozen core and the AST-extracted frozen canonical field metric (no model imports): exact equality for all/family/phase/action groups, paired signs and separately named cumulative prefixes. Additional independent NumPy pooling for every reported lead and prefix reproduces force MAEs, total-drag MAE and velocity relative L2, maximum floating discrepancy `4.85722573273506e-17`.

| Lead | Velocity relative L2 K1 → P029 | Rear Cl MAE K1 → P029 | Total Cd MAE K1 → P029 |
|---|---|---|---|
| 1 | 0.00191421 → 0.00185294 | 0.01673912 → 0.01673912 | 0.00366456 → 0.00366456 |
| 10 | 0.01448968 → 0.01435478 | 0.02352598 → 0.03487054 | 0.00820552 → 0.00963843 |
| 25 | 0.02322766 → 0.02582566 | 0.04377734 → 0.06003383 | 0.01432936 → 0.02134160 |
| 50 | 0.03415950 → 0.03938092 | 0.04501820 → 0.04958477 | 0.01750973 → 0.02910900 |
| 100 | 0.05084972 → 0.06290258 | 0.05562057 → 0.06647598 | 0.01565244 → 0.01894857 |

These are **at-lead** metrics, pooled across 44 cases, not prefix averages. Velocity improves in 41/44 cases at lead1 and 29/44 at lead10, but worsens in all 44 cases at leads25/50/100. Rear-Cl absolute error worsens in 32/44 at lead10, 34/44 at25, 26/44 at50 and 23/44 at100. At lead100 total-Cd signs split22 improved/22 worsened despite worse aggregate MAE; signs are descriptive, not a gate.

Family/phase exceptions must not be hidden: at lead100 train16 rear-Cl MAE improves 0.04961730→0.03957725, while base worsens 0.06022703→0.08230413 and train8 0.05611096→0.08070303. Train8 total-Cd MAE improves 0.01832665→0.01711290. Phase b02 rear-Cl improves 0.04853421→0.04288365; b00/b04/b06 worsen, with b06 0.06448496→0.11551878. All family and phase pooled velocity metrics worsen at lead100.

The separately reported 4400-point prefix1..100 values are velocity 0.03486155→0.04101756, rear-Cl MAE 0.03945242→0.05183984 and total-Cd MAE 0.01326588→0.02027250. They must not replace the lead100 table. Overall44 aggregation is not an equal-weight average of three families.

Interpretation: this fixed train panel supports short-lead field gains reversing beyond the trained H10 exposure, and exposes force deterioration already at lead10. It does not isolate a unique cause, prove capacity insufficiency, or authorize a new objective/horizon. It is not the prior origin51 H10 panel and cannot be relabeled a same-origin replication. Physical mean tolerance and surrogate prediction-error thresholds remain distinct and unchanged.

## Artifact hashes

All paths below are under `artifacts/fcp030_train_horizon_diagnostic_20261006_r2`.

| Artifact | SHA256 |
|---|---|
| result.json | b1042b94fde60aed135c60d348431aa1c6177b1b9b12b1ae8ba56bc9fab6ee7f |
| raw_records.json | bba2e0c988e97f0f03efcc180d2ef776a106ec68884b4d15c9bdb5189ce9d00a |
| selection.json | 453d43298f63299086f94443c373cf4214a47714eedd770c02b13c1661b17bba |
| resource_watch.jsonl | a19d8b6328f729c60ac1c9a8879e0b8e5aea9b231d3c88b180b2443a3e9ebfba |
| run.log | 78bbe10bc15d5fd70e3a3caafd5d627a1fed4e9b661be7255e7780b7666e0b44 |
| evidence/container_created.json | e2c20a160e7a86ca173921aa93bf0e1e658900ec3edb92e7d08daf77f9df801b |
| evidence/container_terminal.json | 1fc527554c89f72acaab6961a80bb3e24810009ee128e99b90d8c29577a133bb |
