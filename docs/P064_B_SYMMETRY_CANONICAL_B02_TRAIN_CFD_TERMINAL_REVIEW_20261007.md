# FC-E089 — fixed-policy b02 acquisition independent terminal review

Verdict: original six physical windows verified; saved controlled trajectory **conversion ready / 可转换**, subject to separate Lead execution approval. This is not conversion/training completion or surrogate admission.

Actual unit `fluid-control-p064-b-symmetry-canonical-ppo-b02-train-acquisition-20261007.service`, invocation `330e850af9e040eaaf10897443807173`, PID0/exited/normal exit0, Result=success. Approval SHA `4c0276eec8e4c2e871bf0fc93cd1ac780a5e3c7263096a87ef9a047a5998a966`; executed driver `b924397f648e3623f5d44145f695b13bfe59df1f9023897e59f707330c24d5d8`. Frozen E082 canonical policy5c05699e/Vec1d250051; not the rejected C50 candidate.

## Independent evidence

Read-only CPU audit invocation `d1c32a54e38d4cb3b3201e3d3eec7838` exited0 under one CPU/2GiB/noSwap/120s, CUDA hidden. Rehashed all3200 raw force files, independently merged four16000-sample streams on the exact .005 time grid, recomputed all six windows, and matched every recorded statistic within6.661338147750939e-16. All1600 solver logs have exactly20 steps, clean End and no FOAM FATAL. All800 physical69-observation canonical maps, orientation/pivot/margin/reflection-fixed fields, sign-restored policy requests and single action filter were independently recomputed. Action maxabs=.6700387001037598, maxdelta=.10000000000000003 (floating-point tolerance).

Rehashed20 original restart/constant/system files unchanged. Inspected both actual container terminal receipts: no OOM, not running,8GiB memory/no excess swap; each exact container ID is absent from Docker. Recorded4000 resource samples, minimum MemAvailable121866289152 bytes, wall1106.4016295910114s. Existing source/input closure was independently checked before launch; no unnecessary model/source-tree rerun was performed. No online FNO/MPC, no new optimizer, no policy/model change.

## All predeclared physical windows

Thresholds remain drag reduction≥2%, rearCl centered-RMS ratio≤1.05, absolute mean rearCl / paired-zero RMS≤10%. Every window passes all three in this run; earlier failures in other runs remain unchanged.

| Window | Samples/branch | Drag reduction | RearCl RMS ratio | Mean-bias ratio |
|---|---:|---:|---:|---:|
| early(106,118.4] |2480|.04807630785323169|.8972238968792253|.009684736369439928|
| first(106,112.2] |1240|.04574525149175335|.9398886905294406|.029179187493947326|
| trailing(112.2,118.4] |1240|.05040838356169164|.852016252463565|.009802209775037627|
| primary(126,186] |12000|.042628021235349256|.8229719660309436|.04452841272048061|
| historical inclusive[126,186] |12001|.042630129117561144|.8229772293866411|.04449425008449266|
| full(106,186] |16000|.043801238405595244|.8371744272362053|.017209452867772597|

Primary:4.2628% drag reduction,17.7028% lower lift fluctuation,4.4528% mean bias. No comparison to a different-phase zero baseline was invented; this trial's zero is its actual same-restart pair.

## Snapshot inventory and conversion handoff

Both `case_mpc` (actual policy-controlled branch; historical directory name is not MPC execution) and `case_zero` contain exactly801 distinct times106..186 at .1 spacing. Every time has regular nonempty U and p files:3204 files total,1,361,731,056 bytes. All were SHA256 hashed into the independent inventory. This checks availability and provenance, not numerical field interpolation or official Curator conversion, which are subsequent separately authorized work. Source originals remain unchanged; copied working boundary files may differ after action configuration and the actual terminal working snapshots are the inventoried sources.

| Bound artifact | SHA256 |
|---|---|
| artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007/result.json |6883ceda495a443bc0edf95047f61bdbfc585cea9217112a92cb21e78299b9eb|
| same output/progress.json |40c9ea6e8fa7617bf15633778f2d156738f4103000a2b041b765b6feb11d0878|
| artifacts/p064_b02_terminal_audit_20261007/audit.jsonstream |2aa79a8dc18455a2280832ec6c10f8a82993d30fc138f67cc824b38ff42a051b|

The audit file contains two JSON documents: first raw/statistics summary, second full branch/time/field path-byte-SHA inventory and terminal identity. Progress rows exactly equal result rows. Controlled801frames may enter a whole-trajectory train conversion after approval; do not mix zero into that training trajectory or randomly split frames into validation/test.

b02 already had eight12.8D/U exploratory trajectories in train16. This acquisition adds the fixed current policy's80D/U coverage, not a new unseen phase or statistically independent test. It does not prove an augmented surrogate will improve. C50 remains rejected; prior full prediction FAIL, old negative seeds and early failures remain. Official components+RL+real CFD online feedback already form the basic requested closed loop; optional online FNO/MPC is not an added mandatory goal. Overall completion remains limited by control-relevant prediction quality and validation scope, not an unproven assumption that32 updates are insufficient. No automatic conversion, longer training or new experiment is authorized here.
