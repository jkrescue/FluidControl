# K1/B/G b00 train-fit: independent terminal review

Engineering/JSON audit ACCEPT; diagnostic only, not scientific admission.

Actual unit `fluid-control-p064-k1bg-b00-train-fit-20261007.service`, invocation
`04335bedb4954871ac118381da81721c`, independently observed MainPID0,
active/exited and ExecMainStatus0. The supervisor records null error and returncode0.

Repository-relative evidence:

- Approval: `docs/P064_K1BG_B00_TRAIN_FIT_APPROVAL_20261007.json`, SHA256 `d03794c76333d3be9cc89d400f60d0928b3ec4fa11810d1e45955844c6875150`.
- Result: `artifacts/p064_k1bg_b00_train_fit_20261007/result.json`, SHA256 `bb256396fe174f7d1f8ef3cd68a06bc1653c0e8ff039604714479f44391da26c`.
- Supervisor: same directory, `supervisor_result.json`, SHA256 `1be5c8b3d4877f0ed1459bbcfe1e2b3ead4e6b68d1f832e576d20bcbc1af465a`.
- Executed source SHA256 `0022521f674400668625b71b02828e881f897bb8f7c8545f37c595566b506b5e`.

Independent checks used saved JSON and byte hashes, without model reload/forward,
training or CFD. All 668 source/runtime/data/model hash checks passed. All120
rows occur exactly once (3 models × 8 mechanical starts × 5 current-state steps).
Each of the40 points has identical state/input/target hashes, force truth,
current/next actions and actual timestamps across K1/B/G. Target array hashes
were independently reconstructed from the saved float32 force values.
All2025 saved statistics (135 model/origin/lead groups × 5 force channels ×
MAE/RMSE/signed bias) were independently recomputed with maximum difference0.
Total drag uses signed front-plus-rear coefficients before error/absolute value.

The receipt records120 aerodynamic calls,0 flow calls and0 optimizer steps.
All three recorded model tensor digests are unchanged before/after; their
manifest identities and four checkpoint byte hashes each were independently
checked. This confirms recorded digest equality, not a new independent model
deserialization. Execution enforced highest/no-TF32 through the reviewed source;
the result does not separately snapshot effective precision flags per row.

## Results

All40 train-fit points pooled (five separate true-state one-step predictions,
not an autoregressive H5 forecast):

| Model | Rear-Cl MAE | Total-Cd MAE | Rear-Cl RMSE | Total-Cd RMSE |
|---|---:|---:|---:|---:|
| K1 | 0.200856927 | 0.051195516 | 0.252425335 | 0.061298368 |
| B | 0.181738779 | 0.050531504 | 0.230941693 | 0.060001238 |
| G | 0.179079254 | 0.046448530 | 0.228624075 | 0.056092591 |

The first step alone has rear-Cl/total-Cd MAE K1=.186709877/.048652880,
B=.164999189/.047704265, G=.163145971/.044144750.
The fifth true-state step is K1=.201529823/.049480788,
B=.185645483/.048859626, G=.182329386/.046305321.
Full per-origin/per-step/four-force statistics remain in the bound result.

G improves pooled train-fit rear-Cl MAE by1.46% and total-Cd MAE by8.08% versus B.
Both B and G improve these pooled errors versus K1, so this sample does not
support the claim that training has learned nothing. Residual force errors
remain, however. These40 correlated points on one training trajectory neither
establish optimization convergence nor isolate representation, objective or
generalization as the cause. They do not override G's original selection FAIL,
its long-AR retention regression, or establish control superiority.

## Resources and scope

Actual cgroup: MemoryMax12GiB, MemorySwapMax0, CPU quota1, TasksMax64,
RuntimeMax630s. MemoryPeak=1746726912 bytes. The30 memory samples span14.5133s;
minimum MemAvailable=120939913216 bytes (112.6341GiB), above the22GiB runtime
guard. This is the sampled supervisor interval, not a guaranteed full unit wall time.
The already-open train-only b00 data, original normalization and official
Reader-backed project adapter were reused. No new data, holdout access,
model update or physical experiment was performed by this audit.
