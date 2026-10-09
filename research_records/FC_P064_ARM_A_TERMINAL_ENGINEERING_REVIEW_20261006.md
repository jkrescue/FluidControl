# FC-P064 arm A — independent terminal engineering review

Status: ENGINEERING_COMPLETE_NOT_SCIENTIFIC_ADMISSION.

Actual training unit `fluid-control-fcp064-aero-arm-a-r2-20261006.service`, invocation `50de1d8b43ce42ac923752fad76ca4d9`, MainPID0, ExecMainCode1, ExecMainStatus0, Result=success. Approval SHA `773daa7329a46930b532586d884502d0c4170ea2c3a689ba73b2fb3d5da8b568`. R1 invocation `cfc40285f8ec49fdba9a99defe2960cc` remains a pre-update CUDA failure, not overwritten or reinterpreted as scientific evidence.

Independent CPU review unit `fluid-control-fcp064-arm-a-terminal-cpu-review-20261006.service`, invocation `8b65dab971bf48ce92268e8d6056d570`, normalexit0. Actual MemoryMax8589934592, swap0, CPUQuota1s, CUDA_VISIBLE_DEVICES empty. Checker source `/tmp/p064-terminal-review.RY8Rx3/check_p064_terminal.py` SHA `8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587`; six synthetic tests independently passed before real artifact inspection. Receipt `artifacts/fcp064_arm_a_terminal_cpu_review_20261006/receipt.json` SHA `2d51c4f84b53fa9b29748c77200cac279ab5a3a4b8ebb5aa6309b4ae9a45f52a`, stderr empty.

Verified after true terminal, never from incomplete candidate:

- Result SHA `03951fee3c1ba66ae48d451fe35aeb7735f092e40ef0b76deff90738c9f8e6a1`; dual manifest SHA `049d29df94a410ba3dbcedf52484447e828cde7787593ad837cbe38be7abd778`.
- All434 frozen source hashes under manifest `05c00539ad03cc9059dc101cf9c9cab47d523a8b5bdae29b4af1cea88c66a9c0`; executed trainer `8066f4a1e092c566e1b84f706ba56737afa06e13998446fee2f79dc2590920dd`, exact direct dependencies, training manifests and normalization bytes.
- Exactly256 journal window identities follow original fixed A schedule, with32 interleaved update events. All32 records contain8 train H100 windows, consumed counts8..256; finite JSON, independently recomputed objective means and clip scales agree. Fixed panels0/256 are diagnostics, not selection.
- CPU weights-only deserialization of hash-bound saved state finds28 Adam parameter states, eachstep32, finite values and exact learning rate1.5625e-7/betas(.9,.999)/eps1e-8/weight_decay1e-4. No optimizer operation performed by review.
- Candidate model archive read on CPU without model construction: tensors finite; `spec_encoder.lift_network.0.conv.bias` and `.2.conv.bias` exactly match K1 parent. Copied flow model/checkpoint bytes remain parent-identical. No HDF data read, model forward, GPU, PPO or CFD execution.
- Training sampled minimumMemAvailable106.84455108642578GiB; actual training MemoryMax12884901888/swap0 and MemoryPeak9884188672B. Reviewer MemoryPeak is unavailable, notzero; sampled minima do not assert unseen continuous extrema.

Producer performed official fresh CPU checkpoint reload separately for aerodynamic epoch1 and flow epoch0 and checked metadata/tensor equality. The independent review confirms the saved checkpoint/tensors and source/terminal evidence, but did not instantiate the new dual-loader pair or repeat official model reload. Result is written before producer's final frozen-flow check; requiring actual normal exit0 closes that ordering gap. Original44 HDF payloads were not rehashed by this checker; data identity relies on bound approved inventories/manifests and earlier verified views.

No candidate accuracy or generalization claim follows. A development-PENDING preparation subsequently failed before writing output because the434training source closure lacks the historical selector; preserve this operational failure and review only the minimal import correction. B separately began from the same K1 parent/freshAdam under its own approval; it is running, not yet successful. OriginalK1 H100FAIL, existing three observed-phase physical control results and original10% physical bias threshold remain unchanged.
