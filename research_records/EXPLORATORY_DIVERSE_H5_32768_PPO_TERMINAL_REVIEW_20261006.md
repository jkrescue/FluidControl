# FC-E054: independent 32768-step exploratory PPO terminal review

Status: TRAINING COMPLETE, NOT SCIENTIFIC ADMISSION. This review reads receipts, source bytes and telemetry; it does not deserialize a policy, run CFD, or infer physical improvement from reward.

## Actual execution and recovery

The retained R2 user unit `fluid-control-exploratory-diverse-h5-32768-ppo-r2-20261006.service`, invocation `a19900b2bfa64d8d8372b67bc0564139`, independently reports MainPID=0, ExecMainStatus=0, active/exited. Approved R2 SHA is `1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132`.

The first invocation `c406bd4af9a84219840027961119e6ab` failed before training because its approval encoded ent_coef as integer 0 instead of the strictly required float 0.0. Its output and approval remain preserved. R2 corrected the serialization and used a separate unit/output, the same immutable trainer and a fresh initialization; it was not checkpoint recovery.

Actual output: `artifacts/exploratory_diverse_h5_32768_ppo_training_20261006_r2/payload`.

- Result SHA: `ff3532a604b6816fb3ad4c7a11edfcd579bcb924445abca52a2fdab8ea4dcf20`.
- Supervisor receipt SHA: `74f515e740da71ee69a372ee332641c9558aaa9b6ba4666ca90832595196a5e3` (distinct from the payload result).
- Final policy SHA: `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a`.
- VecNormalize SHA: `3161ba46c65bac3bc23fa4ccb300c52c95fda63ee4190d9f30d2f0bd4b9040ec`.
- Transition SHA: `e6ac22c4e850cea6427330636c9eaae2c7535cef7694f084652af9681ce62968`.

## Independent checks

All six result-listed artifact hashes match their actual bytes. The saved source specification's 45 source files and 192 runtime files were independently rehashed and match. The immutable trainer remains `4d681771736b63b628712d3b62fcdde831601e80221aef6f1fd78a4b6840ff01`.

The transition file contains exactly 32768 rows. Telemetry records 256 PPO epochs and 512 sequential optimizer steps, exactly eight optimizer calls at each successive 512-transition boundary. Each of the four phases has reset counts `[274,273,273,273,273,273]`. This is the fixed 24-start panel, not expanded validation data or policy selection.

Independently compared all first 4096 transitions against the prior 4096-step diverse-start run: requested/applied/delta actions, predicted forces, source case, CFD time, episode step, reset case/frame, the complete canonical ledger and all reward components have zero mismatches. Wall-clock monitor fields are excluded. This supports the executed budget-only contrast, not convergence or improvement.

The policy tensor digest changes from `6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf` to `96146e21cfb0946f5457f3d1de8bb2b192a5f16446ba2916785a6023681ac51b`. Executed training checks report unchanged frozen FNO tensors (`207932d9da5a9e4f256e16997fbfb4b74a31aaf63db0ca3087d8fe9b62892b0c`), with original high/TF32 loading followed by highest/no-TF32 inference. These tensor claims are execution telemetry, not an independent model reload.

## Resource and scientific limits

Training wall time was 575.562779 s. The supervisor reports returncode 0, error null, memory.max 12884901888 bytes and memory.swap.max 0. Minimum sampled MemAvailable was 119260291072 bytes, above the 22 GiB runtime threshold and 20 GiB physical reserve. No GPU restart or reviewer computation was performed.

Mean surrogate episode return was -3.692436; all completed episodes were length five. These are learning diagnostics only. The frozen K1 model's prior formal failures, short-horizon/bootstrap limitations, and reward history not fully represented in the 69-dimensional observation remain. No real-CFD success or original acceptance criterion follows from this training completion.

## Next consumer review

Independent ACCEPT for the separate 800-cycle direct-policy CFD driver `17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa` and tests `f32b911d99dadbefd0102f14ef17a749b228a16a1eeb29675fe940bb44be4a2a`; four tiny CPU tests passed in 0.20 s. The approved time-only budget is 3600 s inner / 3750 s outer with 120 s cleanup. Policy/action/solver semantics are unchanged, with explicit early comparison windows, primary (168,228], inclusive historical companion [168,228], and full 80D/U reporting. This review does not execute that trial or authorize an extra 124-cycle run. Actual Root approval must bind the policy and review above.
