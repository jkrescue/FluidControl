# Total-plus-pressure parameterization: synthetic CPU compatibility

PASS for the limited auxiliary-weight-zero structural compatibility test; not evidence of model accuracy or component-supervision benefit.

Canonical fixture: `tests/test_p064_total_pressure_compatibility.py`, SHA256 `33388ff34696d55c3821c0402a96416e4a6e913cccb455f09892cec69eb9848c`.

The actual official PhysicsNeMo FNO source (`e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9`) and FullyConnected decoder (`2a5abce0334b04c0eeecc5c8757f238499348a07bf2ff8351f2afdda30abcf82`) are SHA-asserted. This is a tiny synthetic configuration with six input channels, float32 activations and official internal complex64 spectral weights. It has 18 parameter tensors, not the production model's 30.

The expanded official FNO preserves all seven old output rows and shared parameters; four appended pressure rows initialize to half of the old force rows and biases. The loss explicitly uses only old total-force outputs at auxiliary weight zero. Both original lifting biases are frozen using the exact production names. New construction is enclosed in a CPU RNG-preservation context.

Prespecified tolerances: forward/gradient absolute 2e-6 and relative 2e-5; update absolute 2e-7 and relative 2e-5. Actual maximum differences were **zero** for old outputs, every old-row/shared gradient, and one original AdamW update after clipping. All 16 old trainable tensors had a nonzero update, so this was not a no-op comparison. Added pressure-row gradients were zero, both frozen biases remained exact, and the data RNG state remained exact. Initial old/new loss was `0.08749277889728546`.

Actual bounded run: `fluid-control-p064-total-pressure-cpu-fixture-r2-20261007.service`, invocation `226996870b3b4843ae1cbf6ce0f6da7f`; PID0 / Result=success / ExecMainStatus=0. Actual unit properties: MemoryMax 2147483648 bytes, MemorySwapMax 0, CPUQuotaPerSecUSec 1s, RuntimeMaxUSec 2min. CUDA hidden; **1 passed in 2.75 s**. The journal retains the printed metrics. Warp CUDA error100 is the expected hidden-CUDA import warning, not a failed GPU task.

Earlier authorized fixture `19c5b156234cfd5c5fbab0e05f7cb35bc057dfc89aef3c58e61d2a99a7fe758b` used eight synthetic input channels and lacked the explicit nonzero-update assertion; invocation `e6443700edf54702a64f4875620f9d1a` passed in4.08s. R2 changes only six-channel alignment and the additional assertion/metric, not an observed numerical failure or retry of scientific training.

No CFD data, scientific checkpoint, candidate model save, or real training was used. Small synthetic FNO forward/backward/AdamW computation did occur. Nonzero auxiliary weight intentionally changes shared/total gradients; this test neither selects that weight nor proves production-training equivalence or predictive improvement. The component-label coverage audit is separate. Existing total-force labels and normalization must not silently be replaced by raw pressure+viscous sums.
