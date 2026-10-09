# FC-E112 — y-reflection paired training independent terminal review

Conclusion: **engineering ACCEPT; original retention condition FAIL; not admission or promotion**. No fixed-development80-endpoint evaluation was run. Retain B and all previously verified physical controllers; this negative training experiment does not invalidate their real-CFD feedback evidence or change the original physical gates.

## Actual execution and evidence

- Training unit `fluid-control-p064-y-reflection-paired-20261007.service`, invocation `766ad5ca993d483bb6a42d0f2fc09bd8`. Reviewer and Lead observed PID0/inactive-dead/success/exit0 before the transient unit was collected. Current systemctl no longer retains its invocation or limits; these missing fields are not fabricated.
- Approval `docs/P064_Y_REFLECTION_PAIRED_TRAINING_APPROVAL_20261007.json`, SHA `6e5ca18a42a1ad4410260fb9df4f14b457907ff2b5bceadfcfb0f3d621bb81e6`.
- Result `artifacts/p064_y_reflection_paired_20261007/result.json`, SHA `fa57ada9b47414c20c5ee245e8b9e9663e6f568c5f06e4ac9d17dd16b9b53d60`.
- Manifest in the same directory, `dual_model_manifest.json`, SHA `a2c66d2858dbcfd8c83d1b12112f7c659beb0a6b3bc9167951565f860eb3520f`; protocol SHA `82947c35d249e0d7e939bc1aa7971fd0ed2a14b24aa627572fbf3af2d114aad2`.
- Successful bound `supervisor_receipt.json`, SHA `a49037c901209cf06b901e2d699dab1a40d5fb5dd6a7eb07ee76919d1b5b07ff`, binds the actual invocation and result. Original invocation journal remains. The unique subsequent manager completion at timestamp `1791342014400898` records34min4.480s CPU,4.7G memory peak,0B swap; no manager failure message was found.
- Independent CPU audit unit `fluid-control-p064-y-reflection-terminal-audit-20261007.service`, invocation `cb367f5183394782ac482b0d40f1b91e`, PID0/exited/success0. Budget8GiB/noSwap/CPU1/120s, CUDA hidden. Receipt `artifacts/p064_y_reflection_terminal_audit_20261007/receipt.json`, SHA `10ab43ae2c8f7d73833e1b119da4f7f0dca5c74fa9b1962df118b5b706a350f5`.

The audited checker SHA is `10c42b107764baa344fdafbd1667d49965c8bdd71a2a33e47ed023bcbc4e12f5`;13 CPU fixtures passed, with independent Sota review. Its narrowly approved collected-unit fallback requires all lifecycle evidence above before candidate access. No failed scientific run or restart occurred; obsolete V3 preflight path mismatch was corrected before the single actual launch.

## Engineering checks

Verified256 original windows in the unchanged B order,512 ordered original/reflected branches and32 updates. All512 branch journal entries match saved branch input/state hashes and report100 frozen-flow calls plus10 aerodynamic calls each (training totals51200 and5120). Paired source identities/history agree, masks match, each saved pair objective equals half of each branch, and each branch retains equal H1/AR scalar arithmetic. These are source/log/saved-scalar checks, not independent FNO loss replay or proof of learned equivariance.

Rehashed the434 source closure and bound overlay/direct sources and metadata. The large b00 HDF was not rehashed again in this audit; its SHA was checked by the bound supervisor and remains explicitly pinned. Inspected saved optimizer tensors with CPU weights-only loading:28 Adam states all at step32 with original hyperparameters. Both frozen biases equal the K1 parent; frozen flow model/state files equal parent bytes. Saved checkpoint hashes and producer official fresh-reload evidence agree. The reviewer did not independently construct/forward a scientific model.

Training used16GiB allocator,24GiB cgroup/noSwap and original50/22GiB startup/runtime guards. The bound supervisor checked actual unit limits before spawning. Minimum training-reported Available105.8797798157GiB; supervisor samples minimum105.8962707520GiB. Supervisor wall elapsed2050.827562374s. Post-GC systemctl limit fields are unavailable, not observations of an unlimited actual run.

## Fixed-six retention: both conditions fail

Compared the same six unreflected windows at final consumed256 with retained B result `artifacts/fcp064_controlled_aero_arm_b_20261006/result.json` (SHA `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`). All six global indices, train identities, history records and frozen-flow trajectory hashes agree. Independently checked each saved four-channel MSE against `.5*mean(four)+.5*rearCl`, then averaged the six saved objectives; aggregates agree.

| Original normalized objective | B | Reflection candidate | Relative change |
|---|---:|---:|---:|
| H1 |0.003976855262105043|0.004143848258536309|+4.1991218%|
| Continuous AR100 |0.008946200483478606|0.009076183021534234|+1.4529357%|

Both are higher, so the preregistered AND selection rule is already unmet. An additional80-endpoint evaluation could not reverse this retention failure and has not been executed. Its H1–H5 accuracy is therefore **unmeasured**, not reported as failure or success. These modest deterministic differences do not establish statistical significance or a universal rejection of reflection augmentation. No automatic retraining, sweep, PPO, CFD, threshold change or replacement of B follows.
