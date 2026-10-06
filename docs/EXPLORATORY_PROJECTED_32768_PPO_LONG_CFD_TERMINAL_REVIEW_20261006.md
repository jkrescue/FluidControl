# FC-E058 independent projected-policy 800-cycle CFD terminal review

**The predeclared b00 primary physical criteria are met. This is not broad robustness or formal K1 surrogate admission.** One fixed post-training controller projection was tested; the policy was not retrained or selected from this CFD outcome.

## Actual execution and identity

User unit `fluid-control-exploratory-projected-32768-ppo-long-cfd-20261006.service`, invocation `afa5cde4daec474eb52b61c08f86746f`, independently reports PID0/exit0. All800 paired cycles completed in1097.174830s. Approval SHA `87944e807a68caab6ce7a46e01207e1d7ba432b86ccd639b6f47424de481c64c`; executed driver `5c3f40728cd383913a256a2f46b6bfaf0b02cc7d91586e198c354a007fcb9e76`.

Result `artifacts/exploratory_projected_32768_ppo_long_cfd_20261006/result.json` SHA **`199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6`**. Nine imported source files and eight approval inputs independently match their hashes, including actual final policy `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a`, identity VecNormalize, training result and original FC-E055 result.

## Independent numerical checks

All3200 raw coefficient-file hashes match. Each body and branch contains exactly16000 ordered samples at0.005 spacing over(148,228]. Every saved branch mean, centered/total RMS, rear-lift peak and all historical metric deltas were recomputed; agreement is within1e-14. Entire new zero trajectories, both bodies and all raw columns, exactly equal FC-E055 zero arrays.

All800 records satisfy `projected=.5*(raw-reflected)=requested`; the existing amplitude/rate filter, applied once using the actual prior applied action, reproduces every applied action exactly. Reflection indexing was reviewed against32 physical probes atx17/y6..9 abouty7.5;u/Cd even,v/Cl/omega odd. Raw policy outputs were not independently re-inferred with model deserialization during this terminal review.

| Predeclared window | Samples | Drag reduction | Rear-Cl centered RMS ratio | Absolute mean / paired-zero RMS |
|---|---:|---:|---:|---:|
| Early (148,160.4] |2480|3.23097211%|0.876660029|0.070704128|
| Early first (148,154.2] |1240|3.37883353%|0.921834902|0.135461748|
| Early trailing (154.2,160.4] |1240|3.08310628%|0.823955243|0.005947647|
| **Primary (168,228]** |**12000**|**3.89197994%**|**0.815695748**|**0.011385207**|
| Companion [168,228] |12001|3.89181467%|0.815694872|0.011290106|
| Full (148,228] |16000|3.75168096%|0.824355527|0.013698861|

Primary totalCd2.208119871719125 versuszero2.2975396541652002; rearCl mean−0.013464213427872102, centeredRMS0.9646466626471325 versuszero1.1826059717635977; absolutepeak1.350830981 versuszero1.647306236. Original requirements—drag reduction≥2%, RMSratio≤1.05, meanbias≤0.10—are all met in the primary window. Using the original fixed train baseline RMS1.1826535012844825 also yields bias about0.011385, not a threshold-dependent reversal. The20% figure remains sensitivity only.

Do not hide the earlyfirst6.2 window: itsbias0.135462 fails10% even though below20%. Primary/companion/full and other early windows are separately reported; primary was not selected after seeing outcomes.

Against the unchanged-policy unprojected FC-E055 primary, drag reduction improves2.377965%→3.891980%, meanbias drops0.322287→0.011385, while centeredRMSratio rises0.795798→0.815696 (still belowzero). This is evidence for this controller intervention on this matched phase, not proof that policy asymmetry was the only source of earlier bias.

## Resources, health and cleanup

Max|omega|0.6553709208965302; maxdelta0.1; zero saturated endpoints and two rate-limited endpoints. All1600 solver segments ended cleanly with20steps each; maximumCourant0.245427254 and maximumabsolute per-step globalcontinuity1.66568816e-12. The4001 resource observations have minimumMemAvailable121917501440bytes, above22GiB. MemFree is not the Spark UMA admission criterion.

Both exact owned containers are independently absent from `docker ps -a`; retained states are OOMKilled=false:

- `094a12336a8e0df42b2940612490902c5e182a39d0ed1015322f9e0b2f9d61a9`, receiptSHA `d9dabafc530591d0b79488d8c7a0eaa36dd5ae86b39f82d35cf369c604a958ee`.
- `403cf948056088cf8c1d61a61767a83ffba2b8cd582dfc9624d13b1675910512`, receiptSHA `1ab93d3339af49af7dea09fce1d439b5cf9e1271e9c7f73d4f88de6372b541a8`.

Containerexit137 represents stopping sleeping container processes after clean solver segments, not solverfailure. The driver reports the original restart unchanged; the reviewer did not reread large originalfield payloads or run any model/HDF/CFD.

## Scope and next decision

This establishes a genuine surrogate-trained PPO plus explicit reflection wrapper achieving the unchanged primary physical criteria in one b00 real-CFD80D/U paired run. The underlying K1 H100 formal evaluation remainsFAIL. It is not an accepted universally accurate surrogate or a cross-phase robustness result. The fixedb01/restart130 replication may be separately approved using the same policy/projection/protocol and original criteria; b01 was historically used forvalidation and is not a fresh independent finaltest. No automatic retry, architecture change, threshold relaxation or further optimization is authorized by this review.
