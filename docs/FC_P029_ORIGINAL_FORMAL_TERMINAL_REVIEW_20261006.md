# FC-P029 original formal terminal — independent review

**Complete scientific admission FAIL.** Execution completed normally; no PPO admission follows. This review reads existing reports/logs/source files and recomputes the original development audit from saved force-window JSON only. No model/HDF loading, new GPU computation, restart or gate change was performed.

## Identity and operational evidence

Retained user unit `fluid-control-fcp029-original-formal-20261006.service`, invocation `85ae29a422fc48739136418317de8ca5`, is active/exited, MainPID0, exit0. Final development-gate container finished `2026-10-06T03:47:03.068072892Z`.

Output `artifacts/fcp029_original_formal_20261006`; receipt SHA `96e207af491ef4abe0c9e9c85983672111d86d70fe88b2d88551b29d0739a334`; approval SHA `b339d1175ea17dfd110763c044b0d9a2a15ce088062b1834c3627969f279fec5`.

All35 receipt-listed output files and411 frozen numerical-source files independently rehashed exactly. All eight initial/terminal Docker evidence pairs retain their respective exact IDs, official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`, exit0 and no OOM. Seven numerical stages plus precision completed. Across1,117 host-memory samples, minimum MemFree22,763,802,624 bytes (21.200443GiB), MemAvailable118,114,594,816 bytes (110.002789GiB), above the unchanged20GiB floors.

Original frozen development auditor rerun on existing force-window JSON reproduces every discrete decision and structure; maximum float difference4.44e-16 reflects host/container reduction roundoff, not a threshold change. Candidate tensor/official reload integrity was separately reviewed in `FC_P029_TRAINING_TERMINAL_REVIEW_20261006.md`; it is not scientific acceptance.

## Completed endpoint subsets

Validation10 original endpoint gate **PASS**: action-minus-zero Cd MAE `.020433813333511353` against `.023`, signs8/8, ordering20/20; pooled H100 total-Cd NRMSE `.010406120329067193`, rear-Cl MAE `.04511747685228956`. This improves action/rear-Cl errors versus P028 (.02739353/.06937622) but remains worse than K1 (.01912969/.04018628); pooled Cd NRMSE is also worse than K1 .00609718 and P028 .00997537. Passing an endpoint subset cannot override full-window failures.

Dynamic6 original diagnostic **PASS**: pooled H100 Cd NRMSE `.01799069197410125`; strict start0 delta-Cd MAE `.01296532154083252`. P028 values were .02551845/.01575005 and K1 approximately .017893/.01039302. The separate force-window-derived endpoint subset also passes: Cd NRMSE .0056335801, delta-Cd MAE .0131820142, signs4/4, ordering6/6. Its population differs from the pooled dynamic6 diagnostic.

At the03:26:54UTC deadline, dynamic6 was still running and no complete admission decision existed. The later terminal result here does not retroactively turn that earlier partial report into a completed result.

## Complete unchanged tail62 window gate

Thresholds remain1% of same-window zero Cd for drag error, and2.5% of same-window zero rear-Cl fluctuation RMS for both RMS and mean errors. No physical10% mean-lift tolerance is substituted for these surrogate-error thresholds.

| Branch | Absolute Cd error | Absolute rear-Cl RMS error | Absolute rear-Cl mean error | Cd/RMS/mean | Joint |
|---|---:|---:|---:|---|---|
| b01 minus | .01181968 | .02868027 | .01338063 | T/T/T | T |
| b01 zero | .00451520 | .03618451 | .01729135 | T/F/T | F |
| b01 plus | .00157291 | .14219647 | .03676343 | T/F/F | F |
| b05 minus | .01442316 | .08518612 | .06442331 | T/F/F | F |
| b05 zero | .00507944 | .04106958 | .00988583 | T/F/T | F |
| b05 plus | .00617313 | .02845611 | .00189555 | T/T/T | T |

Counts K1 → P028 → P029: joint **1/6 →1/6 →2/6**; Cd **5/6 →2/6 →6/6**; RMS **2/6 →2/6 →2/6**; mean **4/6 →2/6 →4/6**. P029 gains rotating b01-minus and b05-plus RMS/joint passes, but loses both zero-branch RMS passes. b01-plus and b05-minus RMS errors worsen versus both K1 and P028. Thus neither the increased joint count nor all-Cd passes establish all-branch adequacy. The original rule requires all six branches, not best-case selection.

Key output SHA256:

- Development gate: `aa7dd557bc516e898339655517eba8bf16cf27b4579751c6b9f75a4de9153c53`.
- Force-window result: `b9c533674de471c7ded7d4d57fcf971c928443d7aba343ee989026a363cc7890`.
- Validation10 endpoint gate: `caddf56fecb492075480966ad84818737a7592b9a6a9c23d3b8a17881865e6e2`.
- Dynamic6 diagnostic: `34312f37d8f3f2d073561b1719fbca2db463c7b2e2e3a2deee0285e3da9dfe22`.

## Interpretation

The fixed control-aware flow intervention changes the force-error distribution and repairs total-drag window errors, but does not deliver consistent lift-amplitude fidelity. It is not a complete surrogate repair, does not prove a unique failure cause, and does not authorize policy training or real-CFD control. Preserve all positive/negative branches and original thresholds. Any subsequent experiment needs its own reviewed hypothesis and approval; no automatic retry, coefficient sweep, inherited PPO readiness or architecture expansion follows from this report.
