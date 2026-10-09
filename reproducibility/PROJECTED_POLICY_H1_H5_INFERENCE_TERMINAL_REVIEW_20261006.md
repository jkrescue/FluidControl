# Projected-policy H1–H5 realized-action replay: independent terminal review

Status: COMPLETE_NOT_ADMISSION. Independent saved-array audit, 2026-10-06. No model reload, training, CFD solve, or prediction rerun was performed for this review.

## Terminal identity and verification

Actual user unit `fluid-control-projected-policy-h1-h5-inference-20261006.service`, invocation `627cb6b8b59a40aca3bb9159fb617eb3`, is exited, MainPID=0, ExecMainStatus=0. Supervisor returncode=0/error=null. Result `artifacts/projected_policy_h1_h5_inference_20261006/result.json` SHA256 `247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d`; approval SHA `99f7c348afbd5b094018dd3d3cb0e0ee9b7bda50ab7e28bcc4e44b2a57f401fa`; executed driver SHA `94e583182b453d8200a91700f1e68a997e47a9033debf1a9e08ed37cf297841f`.

Independently rehashed all 16 saved NPZ artifacts, 378 source bindings, 192 runtime bindings, and five input bindings. Conversion result remains `a22c3aa67509e9b3a342071398ae85da2ce4e07c74a3cbbd87e2a493c2b248bf`. Exact selection is two branches × starts 0,100,…,700 × leads 1,…,5: 16 origins, 80 endpoints, no missing/duplicate pair. Legacy directory label `mpc` means the already-completed projected PPO CFD branch, not a new MPC run.

Recomputed masked physical u/v/gauge-pressure SSE, reference energy, initial-field persistence SSE, all four force-channel absolute errors, initial-force persistence errors, and total Cd absolute errors directly from saved arrays. Per-origin and pooled/branch statistics agree within relative tolerance 1e-12 (maximum absolute SSE/reference reduction difference 1.93e-9 from summation order); force errors agree exactly. All arrays are finite. Saved force arrays equal JSON predictions/targets; action endpoints match recorded transition actions after the documented float32 storage cast, rather than incorrectly demanding equality between decimal 0.1 and its float32 representation.

Actual cgroup memory.max=12 GiB, swap.max=0; systemd MemoryPeak=1,359,867,904 bytes. The 23 saved resource samples span 11.01 s with minimum MemAvailable=121,518,190,592 bytes. No resource failure. This is sampled monitoring, not proof of every unobserved instant or an isolated inference benchmark. Official K1 manifest is `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`; verified-load high/TF32 precedes explicit highest/no-TF32 inference, with 6 GiB allocator cap. Executed source enforces before/after tensor digest equality and no gradients; receipt records unchanged model and zero optimizer steps. Independent review checked source/receipt, not a new tensor load. This process owns no CFD containers.

## Branch-specific results

Each row is an eight-origin aggregate. Field relative L2 pools masked SSE/reference energy; force metrics are endpoint MAE. Numbers are model / initial-state-or-force persistence.

| Branch | Lead | Velocity relative L2 | Rear Cl MAE | Total Cd MAE |
|---|---:|---:|---:|---:|
| Projected PPO | 1 | .009955 / .024658 | .186710 / .088235 | .048653 / .017126 |
| Projected PPO | 2 | .019175 / .049213 | .171253 / .173239 | .044985 / .033657 |
| Projected PPO | 3 | .027605 / .073564 | .150239 / .256439 | .042979 / .050260 |
| Projected PPO | 4 | .035241 / .097612 | .132798 / .340993 | .039022 / .067839 |
| Projected PPO | 5 | .042127 / .121267 | .149926 / .423392 | .035954 / .084684 |
| Zero | 1 | .001705 / .025006 | .013311 / .101617 | .002687 / .022923 |
| Zero | 2 | .003302 / .049899 | .013156 / .199927 | .003497 / .044789 |
| Zero | 3 | .004733 / .074579 | .015563 / .300061 | .004540 / .065729 |
| Zero | 4 | .005983 / .098949 | .013122 / .398459 | .005545 / .087606 |
| Zero | 5 | .007050 / .122920 | .010806 / .495168 | .007414 / .107840 |

Controlled H5 pressure relative L2 is .13585919 versus zero .01878750. Pooling branches hides the controlled-trajectory error: overall H5 velocity=.030196, rear Cl MAE=.080366, total Cd MAE=.021684. Controlled H1 beats persistence at only 3/8 origins for rear Cl and 1/8 for total Cd; H5 beats it at 8/8 and 7/8 respectively. Beating a deteriorating persistence baseline is not a universal accuracy claim.

## Controlled-branch origin map

Vectors list H1,H2,H3,H4,H5; no origin is omitted or chosen by performance.

| Start | Rear Cl absolute error | Rear Cl persistence error |
|---:|---|---|
| 0 | .014126,.001248,.009908,.005805,.005576 | .171187,.333832,.488296,.637496,.780814 |
| 100 | .098817,.104745,.109082,.109312,.108976 | .101750,.193961,.273236,.339831,.392291 |
| 200 | .264717,.223380,.163211,.096747,.027689 | .010931,.007816,.008137,.036212,.074989 |
| 300 | .252288,.299638,.331598,.348098,.356568 | .069669,.148872,.237464,.335203,.441652 |
| 400 | .261344,.346981,.407974,.436787,.433137 | .129463,.264684,.404680,.548268,.692378 |
| 500 | .081398,.049914,.025849,.001315,.025190 | .129311,.253092,.370086,.479135,.576203 |
| 600 | .104047,.051254,.002171,.057921,.114659 | .058752,.102455,.131530,.146875,.147512 |
| 700 | .416941,.292861,.152123,.006398,.127617 | .034814,.081198,.138081,.204926,.281299 |

| Start | Total Cd absolute error | Total Cd persistence error |
|---:|---|---|
| 0 | .007041,.002231,.002682,.004768,.001213 | .005861,.003150,.005071,.018550,.039381 |
| 100 | .099274,.100161,.098191,.091959,.084564 | .028982,.061439,.094436,.127174,.157454 |
| 200 | .007871,.005101,.007260,.001755,.018135 | .013629,.020472,.022726,.019829,.011562 |
| 300 | .065178,.072664,.071254,.057442,.027736 | .016594,.036950,.058152,.082193,.107743 |
| 400 | .057140,.057036,.055761,.056592,.063186 | .025260,.047156,.066239,.080527,.088507 |
| 500 | .047875,.036559,.030680,.023972,.015325 | .012266,.030494,.052688,.081279,.111818 |
| 600 | .052978,.031377,.020848,.018758,.023440 | .029488,.054531,.074574,.088816,.097860 |
| 700 | .051866,.054753,.057159,.056929,.054035 | .004925,.015068,.028196,.044338,.063144 |

Controlled velocity relative L2 grows from .001750→.007250 at start0, but .007601→.031706 at start100 and .008883–.012848→.038436–.051593 at starts200–700. Zero-branch H5 velocity remains .006619–.007431; zero rear Cl errors across all leads/origins are at most .029321. Thus the initial near-zero-action state is not representative of later controlled states. Later force errors can already be large at H1, before recurrent field-error accumulation; H5-only or pooled accuracy claims would obscure this distinction.

## Meaning and next hypothesis

This is retrospective, realized-action-conditional free autoregressive replay initialized from actual CFD fields. It uses the five commands that were actually applied; it is not an online forecast of unknown future actions, online FNO/MPC control, policy-observation equivalence, or another physical control experiment. Existing physical b00/b01/b03 results and the separate fixed-action H1–H5 confirmation retain their own populations and claims. H100 FAIL remains unchanged.

Evidence supports a controlled-state/action-history distribution gap as a testable explanation, not a proven cause. A next separately approved train-only diagnostic could hold true current fields fixed and compare force readout error across action histories/states, distinguishing immediate force-model mismatch from recurrent flow drift. Do not tune on this retrospective panel, change thresholds, or infer that H5 is generally accurate merely because late-horizon persistence is worse. No new experiment is authorized by this review.
