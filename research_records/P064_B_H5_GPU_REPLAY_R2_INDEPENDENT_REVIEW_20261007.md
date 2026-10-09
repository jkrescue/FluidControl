# B H5 GPU replay R2: saved-result independent review

PASS for bounded saved arithmetic and engineering identity; not numerical bitwise equivalence, prediction admission or control improvement.

Unit `fluid-control-p064-b-causal-history-h5-gpu-replay-r2-20261007.service`, invocation `36f70a0387cc4e2484e07c19c4e1c6f6`, independently observed MainPID0, Result success, ExecMainStatus0. Approval SHA `c6da541d31bcec12af4bd51d886f1f5f1c9e609ae5c0e2b3611cec6e817b5519`; executed source SHA `720885db593b963a7ea9451031ce8fd95cb6298219ba614ecfbac3c37babb943`.

Result `artifacts/p064_b_causal_history_h5_gpu_replay_r2_20261007/result.json`, SHA `56ffc30db493a100ed291a498c3dbdefc5270e192503dcf20821f89a38c2eebf`. Adjacent `supervisor_result.json`, SHA `97ad870a7f704d87dd673c31b7075eb340e7dd4fe4d615de9b674528a9edb3f9`.

Rehashed all declared source and input pins, including the ten saved actual CPU current packets and B CPU result `f189508e962e17c5e98fa8fa6381c18664a4af9a58bd23da5da6977a50324939`. No model was loaded or forwarded by this reviewer. Pure CPU arithmetic reconstructed GPU force/cost values as CPU plus stored differences and checked all 10×5×5×4 force differences, all 50 candidate costs and 250×6 component differences, their reductions, all candidate cost ranks and reported selected-action equality.

- Selected index and exact action agree on **10/10** packets; all-five cost ordering agrees on **10/10**.
- Maximum absolute force difference: `2.4437904357910156e-6`.
- Maximum absolute candidate-cost difference: `1.1830912585164555e-6`.
- Maximum absolute individual component difference: `1.6064035139873312e-6`.
- Synchronized GPU decision time: first `.7464634 s`; remaining nine `.3606044–.3880779 s`. This excludes live CFD/current-field acquisition and is not an end-to-end real-time certificate.

The CPU reference used high/TF32 settings; this GPU replay explicitly used highest/noTF32 after identity-verified loading. Agreement supports only these ten recorded states under those exact profiles, not arbitrary future-state equivalence. It does not measure improved force accuracy, physical control benefit or closed-loop stability; no new CFD, optimizer or saved candidate ran.

Actual service limits were 12 GiB/no swap/CPU1. Supervisor exit0 binds the executed source and actual cgroup; sampled minimum MemAvailable was `119992520704` bytes, above 22 GiB. Model-tensor unchanged is the reviewed producer's before/after check: no tensor digest values were saved for independent recomputation. Full GPU decisions/state bounds were also not saved, so this review cannot independently reconstruct GPU state-feasibility selection beyond reported choices and saved cost differences.

R1 invocation `4ec322d61c424dafb96dbadc6f9d64c2` remains a pre-model Python import-shadow failure, not a scientific result. R2 moved `importlib.util` to module scope and changed exclusive output/unit identity only; independent four CPU fixtures passed before recovery. No failure was hidden and no CFD was repeated.

Read-only arithmetic source: `/tmp/audit_p064_b_gpu_replay_saved.py` (prepared as `stage/audit_p064_b_gpu_replay_saved.py` for archival). Actual CUDA-hidden CPU command exited0, with single-thread environment and 120 s timeout; it did not launch another GPU job. B default and all historical prediction/physical gates remain unchanged.
