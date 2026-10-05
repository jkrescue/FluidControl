# P026 history resource probe — independent terminal review

## Verdict and scope

Operational and engineering checks pass for this one production-size warm training window. K1 and zero-added-column K4 reproduce exactly at initialization, and the new history columns receive a finite nonzero gradient. This is not an optimizer run, candidate, full-training stability test, accuracy improvement or scientific admission.

## Execution identity

- Unit: `fluid-control-fcp026-history-resource-20261005.service`.
- Invocation: `8667f3c9c82146d6ab861d634c460107`.
- Independently observed terminal: `ActiveState=active`, `SubState=exited`, `MainPID=0`, `Result=success`, `ExecMainCode=1`, `ExecMainStatus=0`.
- Result: `artifacts/fcp026_history_resource_20261005/result.json`.
- Result SHA256: `8e1113efc903c6c95cd24755d8c9b081fb098a01ce60aa3c2477e932052f3589`.
- Approval SHA256: `77e9129e150cd383e38ac1051b33ed95c867d25d3f1f8a19fdd6f8048938277f`.
- Probe SHA256: `b806ded8258c787807e67ccb42b5166dbd06e36fc40eba5025d9bda0769eedc6`.
- Launcher SHA256: `5206b31f40df940917cdf65e9f5d5bf03f35eaa17bd573fa4f52a85fff71c07c`.
- Actual container: `6c84aa2ab9551d80a2faef9c289259fd550be32a97c799f842b129517b8051aa`.
- Actual official image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Runtime receipt SHA256: `0666c661764194871721e3355e700e4666eeeb5c1cd8d50b730bba776eebd70f`.

Approval, source/dependency mappings and copied launcher hashes were independently rechecked. Runtime receipt binds the approval label, GPU device 0, 12 GiB container memory limit, network none, user 1000:1000, readonly rootfs, dropped capabilities and readonly input mounts; only `/workspace/output` is writable. The receipt is explicitly a running-container snapshot, not terminal evidence. Terminal evidence is the actual retained unit state, guard completion and `container_exit_code=0`.

## Numerical, identity and count checks

The sole sample is original global index 816, train8 local index 96, case `dynamic_train8_b00_prbs`, start 90, horizon 100. K4 history frames are `[87,88,89,90]`, with no padding. Stored prescribed action semantics remain unchanged and are not newly certified exact nominal-time commands.

Flow and aerodynamic parents remain distinct, loaded at epochs 0 and 1. Their recorded metadata were independently checked against the actual P018 manifest and the reviewed validator; precision also matches the manifest.

- Flow tensor SHA256: `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`.
- Aerodynamic tensor SHA256: `6f58aea89ecdde46bfaafac0f181d2603bbc96e3bba1faa7d8a818dcf6f7984d`.
- Shared frozen-flow history SHA256: `2f69fb73d22171831ea024ecd2db647ffa204d5d996a3e980dd786912a9c402c`.

There were 100 flow calls. Each K1/K4 arm used ten mixed-20 aerodynamic forward calls and ten backward chunks. Saved normalized H1/AR predictions and objective dictionaries were independently compared and are exactly equal between arms: maximum absolute replay difference 0, within the declared rtol 1e-5/atol 1e-6. Shared flow-history hashes agree.

Both arms report H1 balanced objective `0.005017083138227463`, AR balanced objective `0.013695226982235909`, and total `0.009356155060231686`. The runtime requires all 28 trainable tensors to have finite gradients and both original frozen biases to have no gradients. K4 adds 288 lifting coefficients with gradient norm `0.04620205733626772`. This establishes nonzero aggregate gradient, not that every individual coefficient is nonzero.

The complete result was checked finite. Optimizer steps are 0; no optimizer was instantiated or checkpoint saved. Parent and arm tensor immutability checks execute before result completion, including failure cleanup. This is runtime integrity evidence, not an independent terminal-model reload: no terminal model artifact was saved.

## Measured resources

| Arm | Forward/backward elapsed seconds | Peak CUDA allocated GiB | Peak CUDA reserved GiB |
|---|---:|---:|---:|
| K1 | 3.277426071 | 3.640007496 | 4.33203125 |
| K4 | 3.046989948 | 3.729327202 | 4.21875 |

Total harness elapsed time: `22.678514556` seconds. These single-arm timings are observations, not evidence that K4 is generally faster.

- Host watcher: 14 samples; minimum MemFree `29.1618347168` GiB, MemAvailable `107.8577194214` GiB.
- Internal checks: 123; minimum MemFree `29.1219253540` GiB, MemAvailable `107.8178100586` GiB.
- GPU guard: 14 samples; minimum reported CUDA free memory `29.1221389771` GiB and MemAvailable `107.8180236816` GiB; exit code 0.

All observed host minima exceed both 20 GiB floors. K4's two Adam moment buffers are projected at `377783992` bytes (`0.3518387601` GiB); this is an arithmetic projection, not measured optimizer-step peak memory or a bound on future training.

## Limits and next boundary

No scientific acceptance criterion changed. This one warm window does not establish all-window resource bounds, long-run optimizer behavior, saved-model compatibility, surrogate accuracy or control benefit. Any further CPU roundtrip or training experiment requires its own authorized scope. The reviewer performed only read-only inspection and CPU receipt calculations, with no new GPU execution, retry or source changes.
