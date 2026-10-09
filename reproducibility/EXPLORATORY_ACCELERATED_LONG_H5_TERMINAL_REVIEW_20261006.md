# FC-E049: actual124-cycle H5 run and offline metric recovery

## Operational outcome: CFD completed; original process failed in summary

The original unit `fluid-control-accelerated-long-h5-20261006.service`, invocation `a601eec2da7649b4af6f9354a4deb470`, ran06:28:56–06:40:19UTC and remains **failed/exit-code/ExecMainStatus1/MainPID0**. All124 actual paired CFD cycles reached160.4; both branches' per-cycle solver records report20 clean solver steps. The process then failed while constructing trailing-window metrics. No original `result.json` exists, and this failure was not overwritten or relabeled exit0.

Exact cause: reviewed legacy `read_force_window` includes `[begin,end]` with time tolerance. Full interval starts148 where no newly produced initial force row exists, yielding2480. Trailing interval starts154.2, which is present, yielding1241 rather than intended1240. This is a reporting-boundary error, not a CFD interruption. The fixed protocol's intended `(begin,end]` partition produces2480 full and1240 in each half, without interpolation, extrapolation or excluding any other rows.

Lead reviewed and authorized a single offline recovery. It executed successfully using `/tmp/long-h5-recovery/recover_accelerated_long_h5_metrics.py`, SHA256 `5003424f0e8abf4e89733f62d6515a72239eb98a01690188813cadd08d542e7b`. Seven synthetic CPU tests passed before execution; original metric function is extracted unchanged from pinned source SHA `866b4dce33c7401eb447e897641d734828e04b10b38ccde8a0b724e8b5e65d3d`. No CFD, model, optimizer or control action was rerun.

New evidence: `artifacts/exploratory_accelerated_long_h5_real_cfd_20261006/recovered_metrics.json`, SHA256 **`1605604dc27f106acd05e6a721f26c4ba24527ac53996d6e65fbc70c601fa2b1`**. Its explicit status is `OFFLINE_METRICS_RECOVERED_FROM_FAILED_POSTPROCESSING_NOT_ADMISSION`. It records the original failed unit, missing original result, recovered windows and provenance.

## Actual physical measurements

Each branch has2480 unique raw front and rear coefficient samples, exactly following0.005D/U grid. Recovery verifies all raw rows, strict per-window counts and expected grids. Same unchanged `force_metrics` arithmetic is used for every window.

| Fixed interval | MPC mean totalCd | Zero mean totalCd | Paired drag reduction | RearCl fluctuation RMS ratio | MPC mean rearCl |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full (148,160.4] | 2.3147998854906047 | 2.29983717243621 | −0.6505988004% | 0.8322411751 | −0.000889705610 |
| First (148,154.2] | 2.2052307745139514 | 2.299871511958065 | +4.1150445558% | 0.9135812066 | +0.203433729716 |
| Trailing (154.2,160.4] | 2.4243689964672583 | 2.299802832914355 | −5.4163844731% | 0.7003219964 | −0.205213140935 |

Full rearCl RMS is0.9803322922520775 versus zero1.1779425502882805; first1.0761367097172383 versus1.177932188122736; trailing0.8249463352345163 versus1.1779529123145933. Absolute mean rearCl divided by each matched zero-window RMS is full0.0007553047552, first0.1727041096 and trailing0.1742116674. This last normalization is explicitly paired-window, not a substituted original long-baseline admission calculation. Opposite half-window means cancel in the full mean; the near-zero full mean must not be described as stationary low bias.

The favorable first-half drag response reverses in the trailing half. Full and trailing drag fail to improve even without applying a mean-lift threshold. Therefore relaxing10% mean bias cannot turn this recorded trial into a drag-reduction result. Report all preregistered windows; do not select only the favorable first half. This12.4D/U exploratory run is not the original80D/U physical assessment or a model/PPO admission.

Selected one-step prediction MAEs across124 cycles are frontCd0.0007643343941, frontCl0.0049955406632, rearCd0.0584886280760 and rearCl0.0905680774726. Maximum absolute action0.75; maximum change0.10000000000000003 is floating representation of the0.1 constraint;38/124 endpoints are at absolute0.75. These expose substantial later-state prediction errors and saturation, but do not alone identify their causal contribution to drag reversal.

## Accelerated prefix reproducibility

The first ten actual actions match the previous CPU H5 trial exactly. All200 raw force records through149.0 match byte-value-for-value in all four front/rear×MPC/zero numeric arrays (maximum difference0). Prior CPU result SHA: `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`. This is an observed end-to-end prefix check, not a guarantee for all subsequent states or hardware.

## Provenance, resources and cleanup

- Execution is identified by immutable driver SHA `4cca28757f44e80f693d2d4a33c15cea0ea5bc74368eda669aead292b37464bf`. The same source bytes were retrospectively captured in commit `aae121a` after the06:28:56 launch; this is not a claim that `aae121a` was HEAD at launch.
- Approval SHA `03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75`.
- Complete124-cycle progress SHA `c3b004251d6528bb5d65a194d1ad1f3cba1b03e4d8e41555b7c493c1c3337f43`.
- Recovery records536 individual raw-file hashes (496 this trial,40 prior CPU prefix), all approved source hashes, unchanged original restart148/constant/system rechecked against the complete approved tree, and unchanged immutable driver identity.
- Supervisor memory log SHA `1bf62643d379ebd8ceabc69219818e05fbef196dd2a8edc2c2ab1d057d7d744d`; failed-worker log SHA `eaa661cf6d48f85efd723b016849090834f2c7e782e7fef59bf88df07df1c772`.
-1353 supervisor observations: minimum MemAvailable120401592320 bytes; observational MemFree1102553088 bytes; last recorded elapsed676.767893s. Approved UMA available-memory floor22GiB remained satisfied. Original total unit duration was683s including cleanup; no resource failure is inferred from low cached-memory-adjusted MemFree.
- Owned CIDs `e3b3d594f19cfeafc84477c5e892f8fae77f70bd07f06c8745ff96f3188f4cdb` and `a5060cd2e540ffd2700b8a9eb00963fd058ec8bfd7966685eb9a55e855a8c000` were independently absent from Docker's full container list. Saved exit137/OOMfalse records reflect stopping persistent containers, not a solverOOM. Terminal-record SHAs respectively `b93854b41befe55de5d9b6407ff0e19be73cf97b91b47b8d60000a0b0a440fc2` and `228651c7d6e521b7cc94ebb6dd996d838da64460348544b2ba3f61102d583d6f`.

Conclusion: genuine124-cycle accelerated real-CFD feedback was executed and its complete fixed-window physical metrics recovered independently. The original operational summary failed and remains preserved. The scientific drag result is negative overall and worse in the trailing window; reduced lift fluctuation does not establish the drag goal. No automatic rerun, threshold change or new experiment is authorized.
