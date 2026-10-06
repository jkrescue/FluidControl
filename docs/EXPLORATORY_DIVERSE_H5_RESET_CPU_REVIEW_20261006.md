# Diverse-real-start H5 reset engineering evidence

Preparation and approved CPU data verification completed; no new PPO/model training, model inference, CFD or GPU execution. This does not establish control improvement or repair the original formal failure.

The fixed panel is four phase environments (b00/b02/b04/b06), each cycling originalzero frame0 then frames62 of m075,m0375,zero,p0375,p075. Only reset distribution changes; K1/H5/reward/action contract and proposed4096 PPO budget remain unchanged. All44 old trajectory frame0 actions werezero, so adding rotating filenames atframe0 would not solve the coverage problem. Existing canonicalframe0/zero-prehistory guards remain untouched.

## Actual bounded24-packet verification

Retained unit `fluid-control-diverse-h5-reset-packets-cpu-20261006.service`, invocation `cc150546e55c446a9e8af5d0928a52e9`,07:21:09–07:21:22UTC: active/exited, MainPID0, Resultsuccess, ExecMainStatus0. Queried limitsMemoryMax8589934592, MemorySwapMax0, CPUQuotaPerSecUSec2s. Internal supervisor elapsed12.5133364s, returncode0, nofailure.25 sampled resource rows had minimumMemAvailable122649051136bytes, above50GiBstartup/22GiBruntime; MemFree isobservational.

`artifacts/exploratory_diverse_h5_reset_packets_cpu_20261006/result.json` SHA **f85f84a4b82e0c21eaf011281e0b98b570bfaa083805c604e1fbe04aaa14583b**, status `DIVERSE_H5_REAL_RESET_PACKETS_VERIFIED_CPU_NOT_TRAINING`. Contains all24 exact packet identities and source evidence. OfficialHDF5Reader SHA `cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0`; only selected stateframe0/62 was loaded, plus staticcoordinates/attributes. Full HDF bytehashes were streamed against pinnedtrain20 split; no futurefield/force target was used.

All4zero0 and20frame62 packets are finite and have realomega in {−.75,−.375,0,.375,.75}; exactFP32 equality to source action schedules held. Endpointtimes are phase-specific:148/154.2,106/112.2,120/126.2,134/140.2. Each packet has exactly62 raw same-trajectory causal endpoints, no interpolation, ending at nominalcurrenttime. Frame0 uses verified original restart history; frame62 uses the actual controlled trajectory history. HDFcase/split/config_json match rawcaseconfig. Raw force file hashes were checked before/after history reading; frame0 also matches its originalsource forcehash declaration.

Storedfloat32 HDFtime differs from nominaltime by at most3.051757815342171e-6, checked against explicitmax(2float32ULP,1e-7), not silentlyrounded input forces. Curated-vs-raw endpointforce maximumabsolute difference is5.927673329964023e-8 forzero0 and5.594639929218381e-6 forframe62. **The rawendpoint is used**, and these differences are reported rather than called exact. The24 recorded mask/x/y digests each have one unique value, establishing identical sampled geometry/grid across thispanel. K1 input history is one currentstate withfalsepadding; its localindex0 is not a claim that originalframe62 is trajectoryframe0 or that paststates were invented.

## Code and CPU coverage

- Adapter `exploratory_diverse_h5_resets.py`: `a01c0af63b6f8d1c23ce5a7f8dd3a85d0d8f9410ac68e4a8e07b228606465319`.
- Executed verifier `verify_diverse_h5_reset_packets_cpu.py`: `3b11f2c37da1798b8b711c1d7d4b5ab2f612d430abaeea165a1287b4fd1b44d2`.
- Synthetic tests `test_diverse_h5_resets.py`: `31181c74be13028968b4ad1a9fb2c42d72bde48ee86e8728fcc00fc9a1784bd1`,10passed0.56s.
- ActualHydroGym/SB3 lifecycle fixture `check_diverse_h5_actual_lifecycle.py`: `cc7f3fad970eefabda5bf5bbdb695cab6024032cd4a6504345c1fa6b549cffe5`,2passed0.108s. This fixture uses synthetic states/stepper, not officialFNOweights or realdata.

Lifecycle tests exercise the unchangedcanonical reset/state/history/snapshot methods with actualFlowEnv andDummyVecEnv:12resets traverse allsixstarts twice, nonzeroaction/currentK1buffer and raw62 history restore without previous-state aliasing, five-steptruncation is retained, terminalobservation is not confused with nextresetobservation, and eachsuccessfulreset/transition logs itscase/frame/slot. Step reward/action logic remains inherited from the original H5 audit.

The24-packet data run and synthetic lifecycle run are complementary, not a claim of trainedmodel performance. The remaining prospective test is separatelyapproved PPO training with this resetpanel, then the same fixed real-CFD paired evaluation. The hypothesis is reduced reset/action/history distribution mismatch, not proof of solecause; short creditassignment, frozenmodel actionresponse error and curated-vs-raw observation differences remain possible limitations. Moreepochs on the originalfourzero starts would not itself add this missing support.
