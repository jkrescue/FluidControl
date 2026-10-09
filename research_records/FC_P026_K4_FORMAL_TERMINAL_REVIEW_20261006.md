# FC-P026 K4 formal terminal review

## Conclusion

**Operationally complete; scientific development admission FAIL.** K4 did not
repair fluctuation-RMS prediction errors on the four rotating branches.
Joint window pass remains1/6, total Cd5/6, rear-Cl fluctuation RMS2/6,
rear-Cl mean4/6—the same counts as matched K1. No PPO admission follows.
The physical acceptance criterion remains unchanged: paired real OpenFOAM,
dense final60D/U, drag reduction≥2%, rear-Cl-prime RMS ratio≤1.05, and
absolute rear mean Cl/baseline Cl-prime RMS≤0.10. That physical10% condition
is distinct from the surrogate prediction-error gates examined here.

## Independent execution and provenance checks

Output root:
`artifacts/fcp026_history_training_k4_20261005/posteval_fc_p026_k4`.
Retained actual unit `fluid-control-fcp026-k4-formal-20261006.service`,
invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e`, is active/exited,
Resultsuccess, ExecMainStatus0, MainPID0. Receipt SHA-256:
`729f9ce1f307d5462307470af20491806f6cfe31b5c81fc74ec284a2461841d9`.

Independently rehashed **all35 output files** listed in the receipt and
**all411 frozen source files**; every value matches. Copied formal approval
SHA03f6880893a730307a6a6acf0ed19276ca8dc958fc3266ebf14e8e2a1323cbb8,
seven-file candidate map and411-source map agree with the actual approval.
The entire source map and protocol list also equal K1's. K4's aerodynamic
archive is01275829b8cbb894eb940c4dafa0636913359f01eed164a5ab39ffba1819dc9c;
manifest9d1fb9bd61e125451c5c8be802d447077928174e047364b1ef34ba64e2ae1b5e.

Eight saved actual terminal container inspections all report exited0/noOOM
and the same official image as K1. Their commands were compared token-for-token
to K1 after substituting only the arm/profile and exact model/manifest hashes.
All eight mount the actual K4 candidate read-only; source/data/config mounts
are read-only and the output is the intended K4 directory.

| Stage | Container ID prefix | Actual finish UTC |
|---|---|---|
| precision | 1f3ed33695717db1 | 2026-10-05 23:36:53.717799 |
| validation10 | 0be58547d0ccd85f | 2026-10-05 23:47:58.299985 |
| validation diagnostic | 6c8c045b93d02939 | 2026-10-05 23:48:01.704724 |
| endpoint gate | b4d517f47b7bf720 | 2026-10-05 23:48:04.315963 |
| dynamic6 | 0e94dd9738ff2127 | 2026-10-06 00:13:34.345743 |
| dynamic diagnostic | 65e25f7069ce05d8 | 2026-10-06 00:13:35.403991 |
| force window | d6487645f7656eb7 | 2026-10-06 00:14:05.056756 |
| development gate | bbfd89c193038484 | 2026-10-06 00:14:07.508485 |

Terminal evidence remains in `evidence/*_container_terminal.json`; removed
completed Docker containers are not treated as failures. No model/HDF reload,
new experiment or scientific task was launched during this review.

## Original gate recomputation

Frozen auditor SHA
`ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412`
was verified before import. Its original `audit()` recomputed the saved raw
`force_window/result.json` under system Python3.12.3 with **exact full-dictionary
agreement** to `development_gate.json`. Gate SHA:
`ce60621723ce364e4f8cdc165268a64ca5fd1a0185b7529bc21c1f91ac8aab9e`;
force-window SHA:
`156804ee807b19a6670652c931d2275ef694035980e1df22845f60de071c62e1`.
An initial host-env Python3.11 recomputation differed only at floating-sum
roundoff leaves; rerunning the same unmodified source in Python3.12 produced
exact agreement. No tolerance, gate or saved evidence was changed.

The window uses62 sampled endpoints spanning6.1D/U, representing the requested
trailing6.15D/U interval. Original prediction-error limits remain1% of same-window
zero Cd and2.5% of zero rear-Cl-prime RMS for both fluctuation and mean errors.

| Branch | K1 RMS absolute error | K4 RMS absolute error | Original maximum |
|---|---:|---:|---:|
| b01 minus | 0.068338795820 | 0.068235535105 | 0.029415509233 |
| b01 zero | 0.005053055751 | 0.005154798317 | 0.029415509233 |
| b01 plus | 0.122261818126 | 0.122341797254 | 0.029415509233 |
| b05 minus | 0.068218008915 | 0.068306371027 | 0.029387165797 |
| b05 zero | 0.002476237614 | 0.002572254422 | 0.029387165797 |
| b05 plus | 0.081353676819 | 0.081255653164 | 0.029387165797 |

Only b01-zero jointly passes. K4 b05-plus Cd error0.023934022073 still exceeds
0.023031084720; mean errors b05-minus0.049164724626 and b05-zero0.030549243303
exceed0.029387165797. Small changes of mixed sign do not resolve these failures.

## Matched endpoint comparisons

Verified matching split/data/normalization/config/predeclaration/action setup,
validation10 stride25/batch4 and dynamic6 stride1/batch8. Four horizon field
metrics and segment/failure counts are exactly equal to K1 in both datasets,
consistent with the unchanged frozen-flow model. Force metrics are not being
presented as field accuracy. Validation endpoint readiness and the dynamic
action diagnostic pass; these components do not override the window failure.

| Metric | K1 | K4 |
|---|---:|---:|
| validation rearCl MAE H1 | 0.020217614953 | 0.020207039814 |
| validation rearCl MAE H10 | 0.027807793315 | 0.027759916178 |
| validation rearCl MAE H50 | 0.029489039630 | 0.029191357727 |
| validation rearCl MAE H100 | 0.040186283652 | 0.039902067941 |
| validation H100 pooled CdNRMSE | 0.006097181276 | 0.006075173467 |
| validation H100 macro CdNRMSE | 0.005809682472 | 0.005782571929 |
| validation start0 ΔCd MAE | 0.019129693508 | 0.019186198711 |
| dynamic H100 rearCl MAE | 0.084981295319 | 0.084848083874 |
| dynamic H100 pooled CdNRMSE | 0.005015374200 | 0.005059978777 |
| dynamic H100 macro CdNRMSE | 0.015717869831 | 0.015702732228 |
| dynamic start0 ΔCd MAE | 0.010346457362 | 0.010386630893 |

Pooled, macro and strict-start0 quantities are intentionally kept separate.
Dynamic non-tie sign4/4 and cross-action ordering6/6 pass in both arms.

## Resource evidence and limits

Host `memory.jsonl`:1126 samples, minima MemFree20.770393 GiB and
MemAvailable110.100964 GiB. All three GPU guards exited0:

| Stage | Samples | Minimum CUDA-free GiB | Minimum available GiB |
|---|---:|---:|---:|
| validation10 | 133 | 20.948410 | 110.102154 |
| dynamic6 | 306 | 27.768471 | 110.748047 |
| force window | 6 | 29.254292 | 112.249886 |

The separately approved r13 train-file cache pass is documented in the formal
maintenance note; it changes neither evaluation inputs nor scientific rules.
Operational completion is verified, but K4 is **not admitted for PPO**.
This experiment does not establish that history can never help; it establishes
that this preregistered matched K4 run did not repair the required rotating-branch
fluctuation accuracy under the unchanged protocol.
