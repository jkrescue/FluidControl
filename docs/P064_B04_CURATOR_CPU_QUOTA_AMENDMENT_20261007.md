# P064 B04 Curator CPU-quota amendment — 2026-10-07

At `2026-10-07T05:48Z`, Lead authorized an operational-only live resource amendment for the already-running Curator unit `fluid-control-p064-b04-long-excitation-curator-20261007.service`, invocation `777b61f9027244bbb069db9ee8acda90`.

The launch limit was CPU quota 100% (`CPUQuotaPerSecUSec=1s`). Actual `cpu.stat` showed severe quota throttling while the unchanged official VTK sampling pipeline exposed many CPU threads. The same process and invocation were therefore changed in place with `systemctl --user set-property --runtime ... CPUQuota=400%`; the observed terminal property after amendment is `CPUQuotaPerSecUSec=4s`.

No process restart, source, input, VTK, HDF schema, sampling grid, action/force labels, normalization, precision, thread setting, scientific protocol, or output identity changed. `MemoryMax=8589934592`, `MemorySwapMax=0`, the 50/22 GiB memory guards, 20 GiB disk guard, and 1950-second unit limit remain unchanged. The original approval remains immutable and records the launch limit; terminal reporting must distinguish launch CPU1 from the dated live CPU4 amendment.

At `2026-10-07T05:55Z`, after the same invocation had reached 200/801
sampled frames, Lead authorized a second operational-only in-place adjustment
from CPU4 to CPU8.  `systemctl --user set-property --runtime ...
CPUQuota=800%` changed the observed property to
`CPUQuotaPerSecUSec=8s`; the PID and invocation remained unchanged.  This
second adjustment likewise changed no source, data, thread setting, numerical
protocol, memory/swap/disk guard, or timeout.  Terminal evidence must therefore
report the resource sequence launch CPU1 -> live CPU4 -> live CPU8 rather than
rewriting the immutable launch approval.

These amendments authorize no model training, PPO, CFD, or additional Curator
run.
