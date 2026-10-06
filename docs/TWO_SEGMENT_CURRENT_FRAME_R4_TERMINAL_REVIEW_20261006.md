# Two actual zero-action CFD segments: independent terminal review

Engineering completion only; no model, policy, autonomous feedback or scientific admission.

Actual unit `fluid-control-two-segment-frame-cpu-r4-20261006.service`, invocation
`99e5c019c08240668241f7ac036320f3`, ran 05:20:43–05:21:09 UTC. Independently
queried terminal state: active/exited, MainPID0, Resultsuccess, ExecMainStatus0.
Approval SHA `9e268e48f2201eeefbbcb67fefcd3f3184d82877339a429b3f847a2134a9ddd6`;
executed driver `ce443c9b575f7b3492609808de7d322cb10babacd7733d093b6637fe9e990b35`.
Output: `artifacts/online_two_segment_cpu_20261006_r4`.
Result SHA `be3c57e00003d7092b116058604a47d2ea2b2c1f033551adb39188f1c91f7584`.

## Actual solver and frame evidence

Raw solver logs independently contain 20 consecutive dt=.005 steps each:
148.005 through148.1, then148.105 through148.2, both ending cleanly with End.
Reported maximum Courant numbers are .238318304/.239906941 and maximum
absolute per-step global continuity errors 4.02944934e-13/4.29930251e-13.
Both applied-now and next commands were fixed zero. Fresh separate exact-time
VTK exports and Curator outputs were generated, not a reused prior frame.

Sample SHA at148.1: `6a584ede59f1cd7ae29b02594321706dbc2dcb9c080ce3b9822e6c570a509cb7`;
at148.2: `ca456c09a0be9122fc14fc611877960b109a3fa95eb1c1086bff21af868b1806`.
These independently rehashed files match result rows. Their recorded FP32 times
148.10000610351562 and148.1999969482422 satisfy the existing timestamp tolerance.
Canonical normalization/packing produced [1,6,128,256] inputs, with hashes
`176e7e75266c7381d46ee2563ccc6b9ee72034114b2b49cea3388b2135053c5e` and
`41d14e78b8aa9941d6766f6f159f67cbccb9fdff948a1ad845cfcb9ad102a224`.
Packing validation is execution evidence from the reviewed driver, not a second
independent numerical replay. Independent source rehash confirms all ten project
pins and original restart148/constant/system trees unchanged against approval.

## Isolation, resources and cleanup

Actual saved container inspect binds CID
`77a7b4008b1808bbbdef193778f32e4c59477b6b09c1fe11e9d00c71a029a0a4`
to pinned image `sha256:24205c9677d39c95221eb903988094dd7a228fc41a2054df3eaa13f80e465fcb`.
Only the exclusive copied case is mounted RW at /case; root is read-only,
networknone, runc, no DeviceRequests, dropped capabilities, no-new-privileges,
8GiB memory/swap-total8GiB (no added swap),2CPU. Actual outer unit properties:
MemoryMax4GiB, MemorySwapMax0, CPUQuota2seconds/second, Runtime330s,
TimeoutStop45s; recorded MemoryPeak2,081,218,560 bytes.

77 saved memory observations: minimum MemAvailable122,615,746,560 bytes
(114.195GiB), minimum MemFree2,567,946,240 bytes (2.392GiB).
This run explicitly approved CPU-only startupAvailable50/runtimeAvailable22;
MemFree is diagnostic, not the CPU guard. It is not a claim of CUDAfree20
verification or a relaxation of GPU requirements. Resource minima are sampled,
not mathematically continuous bounds.

The persistent sleeping container was stopped during normal cleanup, producing
exit137 with OOMKilledfalse; this is not the solver exit status or an OOM result.
Both docker-exec solvers succeeded. Cleanup errors are empty and an independent
Docker container-list check confirms this exact CID absent. Cleanup SHA
`176c4419be678560ac81af6c1b98b88649ac95f7b739f8a441cd1cb24459bdee`;
memory log SHA `2701481cdd135f85e81ce9810c154e6ad6e73f6215d642be5eabb3921c909cdb`.

## Scope and next step

First startup-memory failure and R2 bare-exec PATH failure remain preserved;
R4 invokes /openfoam/run for both solver and export. 21 CPU regressions passed
before execution. Recommend preserving/promoting these exact tested driver and
sequencer bytes with canonical-path test imports, subject to Lead review.
The priority next step is the separately approved paired ten-cycle MPC trial;
predictions versus next actual CFD truth are compared inside that trial, not
as a separate shadow-only gate or prerequisite task. Existing model FAILs remain.
These two zero-action segments prove a working field transport bridge, not
closed-loop control benefit, an accepted world model or changed physical gates.
