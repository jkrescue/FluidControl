# Independent terminal review: canonical causal-history H2 feedback

**Actual ten-cycle feedback completed; ten HOLD actions, zero paired benefit.**
This is not physical-control completion or long-window admission.

## Actual evidence

Unit `fluid-control-exploratory-causal-h2-real-cfd-20261006.service`, invocation
`6f554f10e87e4b9f9d6b6ed8b555c548`, ran05:48:59–05:52:01UTC. Independently
queried terminal active/exited, MainPID0, Resultsuccess, ExecMainStatus0.
Approval SHA `b273716de8edb19b8517126d3d8232cc50cc5b2ab5a7af04cb16f96aeae1e5a1`;
executed driver `ace9871ac27e2b92de0d90010aa1db3f4cd037cdcf3e49ce5f3f87deb32f06cc`.
Result `artifacts/exploratory_causal_history_h2_real_cfd_20261006/result.json`, SHA
`74a28d45dce9b84ec5044700fe470390cde899a2fcf40a0b893c1b28817d99ca`.

Independent raw force reading confirms200 samples per branch on the exact
148+.005*[1..200] grid. All recomputed force metrics equal saved results.
Both new branches' front/rear/time arrays equal the previous trial's zero branch
exactly. Both have mean totalCd2.413592168615, mean rearCl.887113848192,
rearCl fluctuationRMS.3953805314509264, totalRMS.9712346494562659 and
peakabsolute rearCl1.468057611. Paired drag reduction0 and fluctuationRMS ratio1.
This avoided the prior instantaneous-cost trial's .23293% drag worsening, but
did not produce positive benefit: all ten selected actions were exactly0.

Using the original raw141.9–148.0 prehistory, independently reconstructed every
history update from actual CFD endpoints only. All50 candidate/100 stage ledgers,
components and H2 totals match saved values exactly in the executed Python3.12
environment. An initial Python3.11 strict-equality attempt had floating reduction
differences; no source, receipt or numerical tolerance was modified. History
finished at149.0. Source force hashes remain
front `bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1`,
rear `654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b`.

Ten actual next-endpoint prediction MAEs (frontCd/frontCl/rearCd/rearCl):
.000115096569/.000779956579/.002581012249/.008247816563. These describe this
zero-action trajectory, not independent evidence of reliable nonzero action effects.
CPU inference latency3.662–4.436s per decision; whole-unit duration182s.
60 resource records: minimum MemAvailable122066739200bytes; MemFree2506014720bytes
was logged only under the approved CPU Available50/22 contract. Outer MemoryPeak
2438762496bytes. Both owned containers were independently absent from Docker's
all-container listing; their saved terminal records have OOMKilledfalse and
exit137 from normal persistent-container cleanup, not failed solver steps.
No new model training, PPO or GPU execution occurred.

## Was physical mean-lift10% the blocker?

**No, not in this trial's selector.** Across all100 predicted canonical stages,
the largest mean-bias ratio was .0017321599493, far below .10. Mean-bias penalty
was0 in100/100 stages; fluctuation penalty was also0 throughout. Relaxing .10
to .15/.20, or removing that inactive penalty, would not change these scores.
This concerns the causal62-point scoring windows, not the separate oneD/U
physical mean reported above, and not surrogate prediction-error thresholds.

HOLD won because the modeled drag benefit over this short horizon was smaller
than unchanged action/rate costs. Best drag-component gain across candidates was
only .00001724–.00075835 per decision; a held±.05 candidate already incurs mean
rate+.actuation cost .0012944444. Every nonzero candidate's total exceeded HOLD;
the smallest excess per decision was .00085647–.00127720. This establishes the
recorded arithmetic, not proof that longer-horizon predictions will be accurate.

## One bounded next experiment recommendation (not approval)

Predeclare **H5 instead of H2 only**, retaining K1, the same restart, five held
action candidates, canonical62 actual history, unchanged component weights and
constraints, paired zero branch and ten real-CFD cycles. Continue averaging the
unchanged canonical cost over the predicted stages and execute only the first
action. Hypothesis: additional response time and lower per-stage dilution of the
one initial rate cost can expose a benefit hidden by H2 and select a useful
nonzero action. Report predicted component margins, actual action sequence,
matched force prediction errors and all200 actual paired samples. HOLD again,
or nonzero actions without actual drag benefit, is a negative result—not grounds
for a weight or threshold sweep. Retain the bounded CPU resource/deadline limits;
do not claim real-time capability.

The user's permission to explore relaxed mean bias/accuracy need not be used
where the bias constraint is demonstrably inactive. Any future relaxed profile
must be explicit and prospective; original results and gates stay unchanged.
OneD/U remains shorter than a shedding period and cannot establish original
80D/U physical criteria or complete the overall goal.
