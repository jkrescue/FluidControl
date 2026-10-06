# Projected-policy H1–H5 saved-frame conversion: independent R2 review

Engineering conversion completed, not model-accuracy or physical-control admission. Actual unit `fluid-control-project-policy-h1-h5-conversion-r2-20261006.service`, invocation `60aaab28df8d46508bdcb483a2c63074`, ended MainPID0/SubStateexited/ExecMainStatus0. No scientific job was rerun by this reviewer.

Output `artifacts/projected_policy_h1_h5_conversion_20261006_r2/result.json` SHA256 `a22c3aa67509e9b3a342071398ae85da2ce4e07c74a3cbbd87e2a493c2b248bf`; approved specification SHA256 `da1fb69dc51cde9f955a6e9fe2a69dfd32afa00c488fb6ae04fe42579a77d9ed`.

## Preserved failure and narrow repair

The first conversion failed at its first sampler view because `os.link` could not hardlink root-owned0644 exported VTU files under protected_hardlinks=1. Its artifacts remain preserved. R2 replaces only that operation with source SHA256, `shutil.copy2`, and destination SHA256 equality. Executed R1 converter2258eafafd11c8fa21c49ffdc590311db2bca997f97c767c43d7ef59d971301d was compared byte-for-byte with the old stage; the exact minimal diff was independently read. R2 converter304fece8b7c205dbd8182b9cdae24701fc917b102884406b54734f7e0d0cad68 and tests5771d4bdaa496eaa52e98e3654f48a338fd8bafb07181b996c1de1424a769d77 passed15 independent CPU tests in5.24s, including synthetic officialReader roundtrip. No numerical sampling change was introduced.

## Actual output verification

- Fixed branches `mpc` (historical directory label for projected PPO) and `zero`; eight starts0,100,…700 each, six frames/start:96 packets and16 mini-HDF trajectories,80 prospective comparison endpoints.
- Independently rehashed all216 source-inventory entries and checked their saved sizes against the original completed b00 trial. SelectionSHA `26506735c33417b26c15c407d2a1fe5e39d4812d03436e6c67c6147229c921ac`; source-inventorySHA `5e57223d7cd851e509c45445e89e7fac819ec5cdfd34d68becfcb1ae5e86b5eb`.
- Independently verified all16 HDF file hashes and exact96 packet-to-HDF state/mask/x/y array equality. State shape is6×3×128×256 per trajectory; fields finite. All16 saved officialHDF5Reader roundtrip receipts are true. The terminal review did not rerun Curator, officialReader or a model.
- HDF applied-action samples exactly equal original recorded applied endpoints after the specifiedFP32 cast; all80 four-channel force targets equal corresponding recorded real-CFD observations afterFP32 cast. Initial force equals the fixed selection record. No inferred future action or predicted force was substituted.
- HDF timestamps deliberately preserve sampled VTK times. Although stored in float64, VTK timestamps have float32 rounding: maximum deviation from nominal148+0.1×index is6.103515630684342e-6. They must not be described as exact nominal decimal times. An initial reviewer assertion of1e-12 nominal equality failed for this documented rounding; exact packet equality and the source-contract tolerance are the appropriate checks, not a production tolerance change.
- One common grid/mask identity across16 trajectories: maskSHA4f5e6bdd55ba321065a98d43c6156a9d8556b6c375b05e15b7c691f9d624742b; xSHA95c851fee353f12bb1282c3617376b49f6575a344d7e0e8506e97fbe37a86088; ySHA5757b711dd06a52320e4999d0f42d7df2ea46d5027843257e83e5fbdb9f0aa29.

## Cleanup and scope

Actual export containers `47943eb2db8a4bf736e5b16572129fc023404269983d3614bb3b10939f6cfb5c` and `df42434d8e78aed8f35b58c29f011cdb93e34e8bd20cd734ba55a8c04d107abe` have retained terminal exit0/OOMKilledfalse records and were independently absent from Docker after completion. These executed export only, not a new CFD solver trajectory.

All1192 resource observations were read: minimum MemAvailable122120433664bytes, first-to-last observation span156.551350seconds. The actual noSwap12GiB CPU-only conversion contract was unchanged. Result records model_loaded=false, optimizer_steps=0, cfd_executed=false, scientific_admission=false. b03 live CFD is a separate experiment and was not interrupted.

This establishes the fixed converted input bundle for a separately authorized retrospective realized-action H1–H5 replay. It does not establish prediction accuracy, counterfactual action quality, new closed-loop success or H100 admission.
