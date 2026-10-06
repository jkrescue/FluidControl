# Same24-start deterministic H5 policy comparison — preparation only

Evaluate only the unique final4096 and32768 PPO policies, actual result hashes pinned, on each of the same24 verified train reset packets. Fresh deterministic six-slot phase wrappers per policy, identical frozenK1/H5/canonical62reward/actionconstraints/normalization. Historicalhigh/TF32 load identity then explicit highest/noTF32 inference. All24 cases macro-weighted equally; record each return, component sums and all5 requested/applied actions/predictedforces. Report paired differences, no best-case/checkpoint selection.

No learn/backward/optimizer step, no policy/model save, no CFD, heldout or newdata. Predictedforces are surrogate responses, not a claim of force accuracy against unobserved CFD. This tests deterministic surrogate objective changes, not physical benefit or convergence. Existing80D/U CFD continues independently onCPU.

GPU allocator cap min(.06,6GiB/total) and observedallocated≤6GiB; outer12GiB/noSwap/1CPU, supervisor240s polling.5s and outer260s cleanup20, MemAvailable50startup22runtime reserve20. CUDAfree is observational, not a gate. Source/runtime/data/packet identities reuse prior reviewed execution. No execution approved by pending preparation.
