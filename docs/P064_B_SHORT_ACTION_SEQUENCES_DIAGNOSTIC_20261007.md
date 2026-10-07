# Saved B short action-sequence diagnostic

Descriptive arithmetic only: no prediction admission, new controller approval, or changed physical threshold. No model, optimizer or CFD was run.

## Evidence and protocol

Single CPU execution: `fluid-control-p064-b-short-action-sequences-20261007.service`, invocation `4a52c07dbca34213a082ec581c3b12cb`, independently observed PID0/exited/success/exit0. Actual limits CPU1/2GiB/noSwap/120s, CUDA hidden; CPU0.054161s, memory peak9695232 bytes. Three synthetic arithmetic fixtures passed. Root read the full source before authorizing this execution; Sota separately accepted its source/identity/arithmetic scope.

Result: `artifacts/p064_b_short_action_sequences_20261007/result.json`, SHA256 `ce4dab24d07fef642be9d43faf9897519271eb0b1281c71e7b0488c06598e6df`. Source `analyze_p064_b_short_action_sequences.py`, SHA256 `4dd652894e727efc905260ebd5ef2b8f34e5adb070a3e6b52fb88c28cb3326be`.

Inputs are the existing B AR force-window result (`artifacts/fcp064_arm_b_formal_resume_r3_20261006/force_window/result.json`, SHA6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173), matching H1 result SHA1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef, and existing dynamic6 alignment report SHAea9d0a857d55bf6c5c52e4019f0ce8aa134067a384f3bb123f1d4f7274b6f376. Six actual case configurations were checked against their recorded hashes. Model manifest92766915…/aero57d4634d…/flowdc41fc91… and original normf1b4607e… are pinned in the result.

For all b01/b05 × minus/zero/plus branches, use endpoints1–5 after q0, cumulatively H1 through H5 (0.1–0.5 D/U); never include q0 in the average. Initial times are130 and102. The three branches within each phase share declared restart-field hashes and initial force/action, and saved clocks/actions/targets match the bound H1/alignment evidence. No full initial fields were reread in this diagnostic. Each branch follows its original rate-valid action table; subsequent divergent states are consequences of different sequences, not matched local states at each endpoint.

These are saved **high/TF32** B predictions, not the newer highest/noTF32 development protocol. No cross-precision inference is made. Report separate cumulative mean totalCd, mean rearCl and mean rearCl², trajectories and action-minus-zero differences. No PPO reward was computed: the complete causal reward-history input was not bound. Exact sign and exact-tie sets are numerical descriptions; physical ties/sensitivity remain unresolved without an uncertainty bound. No invented tie tolerance is used.

## Results

| Phase | Horizon | B drag-only choice / CFD best | CFD regret of B choice | CFD choice cost minus zero |
|---|---|---|---:|---:|
| b01 | H1 | plus / minus | .0008728504 | +.0004432797 |
| b01 | H2 | plus / minus | .0029158294 | +.0014880598 |
| b01 | H3 | minus / minus | 0 | −.0032025178 |
| b01 | H4 | minus / minus | 0 | −.0059102327 |
| b01 | H5 | minus / minus | 0 | −.0096566081 |
| b05 | H1 | plus / plus | 0 | −.0002562404 |
| b05 | H2 | plus / plus | 0 | −.0010221004 |
| b05 | H3 | plus / plus | 0 | −.0024930040 |
| b05 | H4 | plus / plus | 0 | −.0048299283 |
| b05 | H5 | plus / plus | 0 | −.0081763625 |

All three Cd pairwise signs disagree at b01 H1/H2; all three agree at b01 H3–H5 and b05 H1–H5. No exact numerical ties occur. These nested horizons and shared zero branches are not independent observations; do not present the counts as generalization accuracy or significance. Earlier E093/E094 first-step response errors remain valid rather than being erased by the later cumulative ordering.

The H5 drag choices have a lift tradeoff: b01-minus increases true mean rearCl² by .05954937 versus zero (prediction+.05835769); b05-plus increases it by .05836667 (prediction+.05157466). Thus matching drag-only ordering provides **no license for a pure-drag controller**. This short, uncentered square proxy is not full-cycle fluctuation RMS, and neither it nor these five-point means determine the original long-window physical acceptance criteria.

The bounded finding is that current B's existing predictions preserve the three-way cumulative drag order at H3–H5 for these two previously opened development starts despite specific early errors, while predicting the observed direction of the H5 lift-square tradeoff. It does not prove arbitrary-action ranking, later-state replanning, MPC competence, optimal PPO reward, new physical performance or surrogate admission. Keep B's default deployed policy, all original prediction failures and physical thresholds unchanged. No further execution follows automatically.
