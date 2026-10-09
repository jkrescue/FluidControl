# FC-P028 resource probe R3 — independent terminal review

Engineering outcome: the actual one-window H10 forward/backward completed.
This is not training completion, candidate admission or PPO authorization.

Retained user unit `fluid-control-fcp028-resource-r3-20261006.service`, invocation
`3ef6ba85c5f34bf58216a74b74f9be13`, is active/exited, MainPID0,
Result=success and ExecMainStatus0. Actual container
`e6d35e8cfb2ce00f13929016eca7f3e7ad553dee52b3745ae9b5396c09084e57`
used pinned official image b40d5888…a22e, exited0 without OOM after about27.27s.
Inspection records GPU0,12GiB memory with no additional swap, read-only root,
network none, and only the dedicated output mount writable.

Approval SHA `dcf52aa2619f7896bf7e98a7edc33bd739dfe14a8d609bd1c97e65b3fc219320`
matches the result's complete source-spec dictionary. Result:
`artifacts/fcp028_flow_resource_probe_r3_20261006/payload/result.json`, SHA
`e808095f9c4de77f838c7132615427ba76985f78d3c0c7803b1d8528008a40f1`.
Terminal Docker evidence SHA
`32e58aeb0e3637dac7b7b3ed3dbe1f707adc91458236e8efcf1d9231587c9be5`;
resource-watch SHA
`4825cc944459231f4ebaacdb869e6e630539d5fa6c9069de7d56e96fad52f3b6`.

## Actual computation and invariants

The original global816 window, `dynamic_train8_b00_prbs`, start90, used the
first10 transitions of its original H100 sample. The ten-step field loss was
0.008573819883167744; per-step values ranged from0.0004228757170494646 at lead1
to0.017602618783712387 at lead10. This is descriptive train-window evidence,
not a fitted-model improvement or scientific acceptance result.

All30 flow parameter gradient norms are finite and nonzero, ranging from
0.0019007059194029307 to0.7396764665482106. Flow tensor SHA before and after is
`89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`;
the fixed aerodynamic tensor SHA is
`b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb`.
The frozen runner checked both invariants. No optimizer was created, no update
was taken and no model/candidate was saved. The protocol's171/1368 fields describe
future training; actual probe counts are zero updates and one window.

## Measured resource limits

Fourteen external samples give minimum MemFree20.90283585GiB and
MemAvailable110.31497955GiB. The continuous GPU guard reports13 samples,
minimum CUDA-free20.90512848GiB and exit0. Internal minimum physical-free was
20.91607285GiB. CUDA peaks were allocated1.75091124GiB and reserved1.96484375GiB.
All observed floors remained above20GiB, although physical headroom was narrow.

Flow parameter bytes are188,891,132. Two Adam moments alone would add377,782,264
bytes (0.35183715GiB), before scalar steps and optimizer/foreach temporaries.
This accounting is a lower bound, not measured optimizer-step peak capacity.
The no-optimizer result cannot by itself certify full-training memory safety;
the original continuous guards and a separate resource decision remain required.

## Earlier failures are preserved

R1 failed before model/HDF loading because its418-file closure omitted the
explicitly required P011/P013/P026 helper modules. R2 source preparation added
those exact three files without changing the previous418. The next execution
failed before model loading because the pinned configuration expected
`/workspace/base`, `/workspace/train8`, `/workspace/train16`, whereas the launcher
had mounted host-path destinations. R3 corrected only those read-only destination
aliases. Earlier failed artifacts remain intact; neither failure was disguised
as a scientific result or numerical training change.

This independent review read existing JSON/logs and unit properties only; it
did not launch a job, reopen model/HDF payloads, modify sources or run cache advice.
