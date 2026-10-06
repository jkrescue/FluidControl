# FC-E104 — G-policy b01 paired real-CFD independent terminal review

**All six predeclared physical windows pass the unchanged2%/1.05/10% criteria.** The primary window has3.951345776959% drag reduction, rear-lift fluctuation ratio0.816377520382 (18.362247961805% reduction), and mean-lift bias3.071693562112% of paired-zero fluctuation RMS. This is an explicitly approved exploratory control result, **not** G prediction admission, a replacement for retained B, or evidence of new independent generalization.

## Actual identity and independent checks

Actual CFD unit `fluid-control-p064-g-symmetry-canonical-b01-cfd-20261007.service`, invocation `c14a66a1464a4919a3904c1f4d2efb2d`, independently observed MainPID0/SubStateexited/Resultsuccess/ExecMainStatus0. Approval `docs/P064_G_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json` SHA `f72cace2befb6aef72ed6878ec83d675cc92b4992ec0b9e6e45cd38be97bbb48`. Output `artifacts/p064_g_symmetry_canonical_b01_cfd_20261007`.

- Result SHA: `f6319771a93541275f4fe7183d49fba9d607100fcdb7d9bf31b9fc92946594a4`.
- Progress SHA: `8b6c25349d76550e51753d51c83a39bb8aebd69009efe0db9da963f9318034dd`;800 rows exactly match result.
- Matched retained-B E085 result SHA: `c56a5cbccdf218953a0e1ea040da1c1ee7e2f1b574e7b509939a622f65b7495f`.
- G policy: `c1157806b2efc54fcf979df4734e09e846068908562f5415af1a43775881821b`; VecNormalize: `043125f7b6ccab8bea70cbee1b43c6797c747ce0a107b088d2fad9d5bb5e5de9`, bound by the actual approval and prior [PPO engineering review](P064_G_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md).

The authorized terminal-only CPU audit ran once as `fluid-control-g-b01-cfd-terminal-audit-20261007.service`, invocation `41d0d172f869480680f8d618a79fb561`, PID0/success/exit0 under2GiB/noSwap/CPU1/120s. Wrapper SHA `9beb443485e896b892f0003f49e4c2776885b858ff452bea6c16607010fec596` and original raw-checker SHA `cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8` were verified before execution; source-difference tests and Sota independent review had passed. Full journal export receipt SHA `8f910ca5028bf9bdb5d3c6220bb9f58c7b4848b949f379950b072fa3f99465e3` (`/tmp/g_b01_cfd_audit_receipt.txt`). No policy/model inference, training or CFD rerun occurred in the audit.

Independently rehashed all3200 force files and reconstructed16000 samples per branch/cylinder at .005D/U spacing. All six windows' means, centered RMS and paired ratios were recomputed with maximum numeric difference4.440892098500626e-16. All1600 solver logs have20 steps and cleanEnd/noFatal. All800 physical observations were independently canonicalized, including pivot/margin/orientation, policy-request sign restoration and exactly one physical rate/amplitude filter. All799 consecutive feedback links are exact; raw endpoint front/rear Cd/Cl match output observation64:68 with zero FP32 difference. All20 restart hashes remain unchanged. All columns of both zero-cylinder force histories exactly equal E085, not merely the first interval.

Both owned solver containers exited without OOM and were removed; recorded limits are8GiB each with no excess swap.4001 resource samples have minimum MemAvailable121999273984 bytes (above22GiB runtime guard). Maximum |omega|=.666481792927 and |delta omega|=.10000000000000003 (floating-point representation of the unchanged .1 limit). No online FNO or MPC action selection occurred: this is learned CPU-policy deployment with actual OpenFOAM feedback.

## All six physical windows

Ratios and bias below are dimensionless, not force-error percentages. Every row is checked against drag reduction≥.02, rear-Cl fluctuation ratio≤1.05 and absolute mean bias/zero RMS≤.10. Startup windows are retained, not excluded after seeing the outcome.

| Window | Samples | Drag reduction fraction | Rear-Cl RMS ratio | Mean-bias/zero RMS | Original gates |
|---|---:|---:|---:|---:|---|
| early12.4 | 2480 | .0544215401642 | .918717034654 | .0388590438769 | PASS |
| early first6.2 | 1240 | .0586990847875 | .982031622249 | .0800538451896 | PASS |
| early trailing6.2 | 1240 | .0501435748966 | .848715817912 | .00233301454106 | PASS |
| primary final60, (150,210] | 12000 | .0395134577696 | .816377520382 | .0307169356211 | PASS |
| historical inclusive final60 | 12001 | .0395126437169 | .816380635833 | .0306278859539 | PASS |
| full80 | 16000 | .0420072902828 | .834229014725 | .00150597289856 | PASS |

## Matched B comparison, timing and action cost

Same-phase retained B has primary drag reduction4.009068948527%, RMS ratio.817190132157 and bias.036366076181. G therefore has **0.057723171568 percentage points less drag reduction**, slightly lower lift fluctuation and lower mean bias on this one trajectory. This is not consistent superiority, statistical significance, a seed study or a reason to erase G's prediction-selection FAIL. Controlled trajectories genuinely differ: maximum applied-action difference.115677988529; maximum physical-observation difference.115678012371. Bitwise equality was required only for the paired zero, not the different policies' controlled branches.

Actual recorded wall time1108.864259365015 seconds for800 feedback cycles gives1.386080324206 seconds per cycle when dividing total wall time by800. This includes run overhead; it is not a pure solver latency benchmark. Simulation physical coordinate130→210 is inD/U, **not wall-clock seconds**, and this experiment does not establish real-world physical real-time control performance.

| Descriptive action proxy over800 applied endpoints | G | retained B |
|---|---:|---:|
| mean omega² | .2393349267626727 | .2404474707148053 |
| omega RMS | .4892186901199429 | .4903544337668472 |
| sum omega² × .1D/U | 19.14679414101382 | 19.23579765718443 |
| delta-omega RMS, first previous action0 | .0627316913918734 | .0627281408782247 |

These are action-square cost proxies, **not physical power or energy**. Torque/sign/normalization-to-power conversion has not been verified here, so no net-energy-saving claim is made. No additional moment analysis was performed.

The basic official-components→surrogate-trained RL→real-CFD online-feedback chain remains demonstrated. This b01 phase was already opened and is not a new holdout; broader robustness and complete surrogate accuracy remain unresolved. Existing early/seed failures, B's full prediction failure and G's fixed-six retention failure remain intact. Any further experiment or replacement decision requires explicit authorization.
