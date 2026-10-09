# Predeclared b03 physical-policy confirmation — preparation only

No execution is authorized by this document. Lead selected b03 before this trial's outputs, not by comparing candidate physical performance. Only new staged driver/tests are prepared; no model/HDF/CFD/GPU execution or canonical source edit occurs.

## Single intended change

Reuse the completed b01 800-cycle paired-CFD implementation with the identical final32768 PPO, identity VecNormalize, reflection projection `.5*(pi(o)-pi(Ro))`, single amplitude/rate filter, 69D observations, rear-only actuation, CFD numerics and resources. Change only the fixed initial phase/restart: b03, real restart144, end224. No training, optimizer, FNO inference online, MPC substitution, reward change or tuning based on the retrospective96-frame replay or frozen H5 confirmation.

Policy SHA256 `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a`; VecNormalize `3161ba46c65bac3bc23fa4ccb300c52c95fda63ee4190d9f30d2f0bd4b9040ec`. The original training invocation and artifact preflight remain unchanged. No new candidate selection occurs.

## Existing metadata and exposure

The existing phase manifest `artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json`, SHA256 `6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603`, selects b03/frozen_test/restart144 at phase2.3050412654150345rad. Pinned U SHA is `7bb667038ffcb6c3fb21e429b7ce29f41d86e4c6835f167670489709570e49b2`; p SHA is `0931ff502abc92cb66de4a98fc83c39ced438929531eac1e470a25f31b206665`.

Actual `cfd/tandem_cylinders/cases/tandem_backward_dt005/144` is present with U,p,phi,U_0,phi_0,uniform. This preparation checked names/stat availability, not payload equality. Before any approved launch the existing whole restart/constant/system tree preflight must bind and verify the exact source and paired copies. Actual matched zero config `cfd/tandem_cylinders/cases/matched_start_acquisition_frozen_test_b03_zero/case_config.json` SHA256 `923b82c01a0bd2b38d8b28a2db3b0875195d520552b9f1c4a906cfb917051da0` specifies start144/end224/analysis164..224,dt.005,zero action and the same source provenance.

A bounded read-only search of current docs/state/experiment records and `artifacts/*projected*b03*` found no prior projected-policy b03 control trial. This is not a universal filesystem-access claim. b03 fixed-action H1–H5 payloads have already been opened under FC-E060; therefore this is a new physical-policy trial, NOT a universally unseen phase, and the common limit-cycle phases are not statistically independent samples. The physical policy is frozen before this proposed trial.

## Fixed reporting and decision

800 intervals of0.1D/U, two branches from the identical restart, controlled versus pairedzero, dt.005. Preserve all six windows: (144,156.4], (144,150.2], (150.2,156.4], primary(164,224], companion[164,224], full(144,224]. Expected branch counts2480/1240/1240/12000/12001/16000. Never substitute the inclusive companion for the primary.

Primary retains the original ≥2% total-Cd reduction, centered rear-Cl RMS ratio≤1.05 and absolute rear-Cl mean divided by pairedzero RMS≤0.10. Report each component and all windows, including peaks/transients;20% is sensitivity only. A complete operational trial with any failed primary component is negative physical confirmation, not grounds to retune or choose another phase. No H100 surrogate admission follows either outcome. Compare relative-window results with b00/b01 descriptively, without statistical-independence claims.

## Unchanged execution protection and pending work

CPU policy only; controller8GiB/noSwap, two8GiB/noSwap solver containers, startup MemAvailable50GiB/runtime22GiB with20GiB reserve, inner3600s/outer3750s/stop120s. Preserve exact owned-container cleanup, exclusive output, source/runtime/image pins and original state preservation. Intended separate output: `artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006`.

This source preparation does not create an executable approval. Independent source/test review, a new immutable driver/spec binding with actual source-tree hashes, and separate Lead execution authorization remain required. No new scientific prerequisite or numerical threshold is added.
