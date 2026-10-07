# Fixed 40-point B fitting diagnostic — independent saved-output review

Engineering completion and saved arithmetic PASS; preregistered fitting target NOT MET. No candidate was saved or promoted. This is train-panel optimization evidence, not development, causal-control, or capacity certification.

Actual unit `fluid-control-p064-fixed-small-fit-r2-20261007.service`, invocation `a36a5f0f9d21455c84e7cf0c85e67d9f`, was observed by Lead at PID0/success/exit0. Supervisor receipt independently reports returncode0/errornull. R1 precision-load failure remains separate (no forward/optimizer); R2 loading uses legacy high/TF32 then switches to highest/noTF32 before any forward.

Bindings:

- Approval `docs/P064_FIXED_SMALL_FIT_R2_APPROVAL_20261007.json`: `f3f5701dca46f10e31647b843cd1e4840d2f90d4f1fc2b12645e30d70b91a6ba`.
- Result `artifacts/p064_fixed_small_fit_r2_20261007/result.json`: `cfee8ea602cc70990094041acc10d6fcaa723be5179905fa6e3bc2b57d6c1d6c`.
- Supervisor receipt in that output: `607bcb742c8805fd1bc468ed922f930564b93facb63e6546402f46c6dad7d891`.
- Reviewed frozen worker `2f43366dde682bfc8e7fadf76140ddb2c5dc5ce7d4b4d0d5077ad17f9bf85c9c`; core `e0623157281a041b05257d9843ca46e8e1557ffca83c5872d9280fbacdd5a6c0`.

Independent CPU arithmetic (`/tmp/audit_fixed_small_fit_saved.py`, no torch/model import) rehashed bound code/reference files and normalization, matched all 40 panel records to E105 by input/state/target SHA, truth, action and times, and recomputed initial/final plus all 140 accepted saved predictions. Maximum difference from saved metrics was 1.400971e-6, consistent with float32 physical-output roundtrip; this numerical comparison tolerance is not the fitting threshold.

Weighted normalized loss decreased from 0.0404872373 to 0.00142032525, a 96.4919% reduction. Final per-channel normalized RMSE (front Cd, front Cl, rear Cd, rear Cl) was 0.0582983, 0.0458753, 0.0524186, 0.0249466: all remain above the unchanged 0.01 target. Independently recomputed physical MAE was 0.000431219, 0.011130166, 0.007650582, 0.024983640. Total-Cd MAE decreased from 0.050531495 to 0.007738246.

There were 140 accepted points and 300 closure evaluations. Attempt 141 raised the global closure-budget exception, recorded successful restoration, and retained exactly the previous accepted measurement as final. All accepted losses were nonincreasing; the last ten decreased from 0.0015800301 to 0.0014203253. Thus this stopped at the fixed closure budget while still improving, not by a demonstrated loss plateau. No conclusion of insufficient model capacity or insufficient inputs follows.

Counts reconcile: 300 gradient panels plus 141 no-grad panels, four microbatches each, equals 1764 aerodynamic forwards; flow forwards 0. Reported fitting elapsed 255.041s; last supervisor resource sample 264.079s. Minimum observed MemAvailable was 102.5386 GiB, above the 22-GiB guard. Launch contract was 24-GiB/noSwap/CPU4 with 16-GiB allocator cap.

Evidence limits: no forward or training was repeated by this review. Frozen-flow/bias checks and successful restoration of actual in-memory tensors are supported by the pinned producer assertions and independently tested transaction implementation, not a saved model reload; no model was saved. The saved final measurement was checked, but was not independently inferred again. Optimizer history, trial-vs-accepted semantics and rollback were separately exercised with synthetic CPU integration fixtures. A trial optimum never constitutes fitting success. No dev or physical admission criteria were changed.
