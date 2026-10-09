# Independent review: actual paired ten-cycle H2 MPC/CFD trial

Actual exploratory feedback executed, but the short paired window does not show
drag improvement and does not establish robust control or scientific admission.
No thresholds are changed.

## Identity and operational completion

Unit `fluid-control-exploratory-short-h2-real-cfd-20261006.service`, exact invocation
`e3b9eb7b58724a1c9ec4e64d63ac7bbe`, ran05:34:11–05:37:14UTC. Independently queried
active/exited, MainPID0, Resultsuccess, ExecMainStatus0. Actual result:
`artifacts/exploratory_paired_h2_real_cfd_20261006/result.json`, SHA
`45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb`.
Approval SHA `1679f6bee293eae200c95bc078a20db5873e81afcd54d81b81bb06445015f8b2`;
immutable driver SHA `c8260b21d742f54814bf04e81bb13eeff491a971df4d177f3a384c8f8c940e3f`.
Official K1 flow/aero identity remains non-admitted. The project controller used
fresh CFD fields before choosing each command; only the selected first H2 action
was applied. The paired zero case started from the same copied restart.

Both saved owned-container terminal records show OOMKilledfalse and exit137
from stopping their persistent sleeping processes during cleanup, not failed
solver commands. CIDs `257e7829da44448e441166308bfa4ee79399886d15a4f80d46951a8ff27dfb22`
and `e49a8a5a3542536e70673e13ec442b5adcd434013587cb279d541d5658b5e710`
are no longer running; the driver checks their complete removal before successful
exit. Each had8GiB/no-added-swap/noGPU device requests. Root requested outer8GiB,
no swap,4CPU,930s,stop120; queried outer MemoryPeak3,137,253,376bytes.

60 saved resource observations give minimum MemAvailable121,930,932,224bytes
and MemFree1,591,545,856bytes. This was explicitly CPU-only Available50 startup/
Available22 runtime protection; cached pages are not a CPU MemFree admission
failure. GPU requirements were not relaxed. Inference latency per decision was
3.625–4.414 seconds, not a real-time guarantee. Whole unit wall duration was183s.

## Independent actual-force recomputation

Read original coefficient files through the unchanged force reader; both branches
contain exactly200 unique times equal to148+.005*[1..200], maximum grid difference0.
All recomputed `force_metrics` leaves equal saved metrics exactly. Twenty solver
segments report20 steps each (400 total), maximum Courant .24522487.

| Metric over actual148.005–149.0 samples | MPC | Paired zero |
|---|---:|---:|
| Mean total Cd | 2.4192140666805 | 2.413592168615 |
| Mean rear Cl | .857989757705 | .887113848192 |
| Rear Cl fluctuation RMS | .372887030442092 | .395380531450926 |
| Rear Cl total RMS | .935516521392651 | .971234649456266 |
| Peak absolute rear Cl | 1.390924178 | 1.468057611 |

Paired drag reduction is **−.2329265954%**: drag worsened. Rear fluctuation RMS
ratio .943109234726 means5.6891% lower in this short window. These are descriptive
short-window measurements, not application of the original long-window gates.
Rear Cl remained positive and rising across this early window; oneD/U is shorter
than the shedding period, so mean and fluctuation decomposition cannot stand in
for original80D/U control evaluation or its physical10% mean-lift requirement.

Selected omega increased by .05 each cycle from .05 to .50. Maximum change.05
and maximum magnitude.50 respect existing .10/.75 bounds. Exact prescribed ramp
integral of omega squared is .0833333333333D/U, an actuation proxy, not motor energy.
The ten matched next-endpoint prediction MAEs, ordered frontCd/frontCl/rearCd/rearCl,
are .0000855565071/.000861443579/.004545694590/.012297362089. These compare
predictions to actual applied-action endpoints, not merely model self-rollout.
The direction-agreement diagnostic involving realized paired-zero endpoints is
not a model-only counterfactual action effect.

## Interpretation and next decision

This is a genuine short FNO-assisted action-selection/real-CFD feedback loop,
not a shadow-only run or a CFD-only PPO baseline. It is only one paired ten-cycle
exploratory trial; it does not complete the overall accepted-model/control goal.
Prior K1 formal failures remain unchanged.

Before simply extending to800cycles, review the fixed cost against the intended
control objective: H2 rear-Cl variance plus squared H2 mean equals H2 meanCl².
It is an instantaneous two-prediction lift-energy penalty, not a physical
long-window mean-bias criterion. The observed drag/lift tradeoff is consistent
with that cost but does not prove causation. A causal62-history cost may reuse
actual past force observations only after its timing and initialization are
explicitly checked; no future CFD truth or new threshold is permissible.
Long-enough paired evaluation is ultimately needed, but its objective and
throughput should be agreed before the next separately approved execution.
