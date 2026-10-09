# E109 one-interval recovery replay: independent review

PASS for technical restart recovery only. This replay repeats known time 327.9→328; it is not a new physical performance validation and does not execute the proposed 328→408 extension.

- Actual unit `fluid-control-p064-b-e109-last-interval-replay-20261007.service`, invocation `083fac39b66e43f79281f167494e1bbf`: independently observed PID0, exited, Result success, ExecMainStatus0.
- Actual limits: controller8GiB, swap0, CPU400%,300seconds; two solver containers8GiB each with no additional swap.
- Approval `docs/P064_B_E109_LAST_INTERVAL_RECOVERY_REPLAY_APPROVAL_20261007.json`, SHA256 `4fd42e9249aff2d0584746d2bbf74842bf05143ca96da5c33144ce6b97b8b9f5`.
- Executed source `/tmp/p064-b-continuation-review/replay_e109_last_interval.py`, SHA256 `aabff4ffff3d4ed5f8afa5fe8e8851b42962652dcaee0b43b01f4f021e3e83af`.
- Result `artifacts/p064_b_e109_last_interval_recovery_replay_20261007/result.json`, SHA256 `86d80da3781490d9d8e127385b0ebffc623ac8520e7c08bfe0b1790fea168df7`.
- Parent E109 result SHA256 `d53cb2ea32f66af6c4067d0c7eb90e7aecc8b815634588b6fe448d291acc5b98`; source-tree inventory SHA256 `2d199ebf6c5d61ccc19e24b357ae324a101c50d71c755959fadfda2639093ff0`.

Independent read-only checks after terminal:

1. Rehashed all input bindings and the driver. All56 files in the two branches'327.9/328/constant/system trees still match the bound inventory.
2. Recomputed the single rate/amplitude filter from saved row799 double action0.2990577340126038 and row800 physical request. Every action field equals row800; applied action is0.19905773401260382. The float32 observation action is deliberately not substituted for this double limiter state.
3. Read actual replay probes and front/rear force endpoints through the existing transport reader: both physical69 observations exactly equal saved E109 output observations after float32 conversion.
4. All columns of four raw front/rear coefficient files exactly equal the corresponding E109327.9 interval. Both solver logs contain20 time steps, clean End and no FOAM FATAL.
5. Independently parsed numeric internalField blocks of U,U_0,p,phi,phi_0 for both branches: all10 match exactly. Both uniform/time files also match byte-for-byte. This does not assert all boundary/header bytes are identical.
6. Both exact owned container IDs are absent from Docker inventory; terminal records show stopped/noOOM and8GiB memory=memory+swap. Idle container shell exit137 results from owned cleanup; the actual solver logs and controller exit are successful.

No policy/model forward or CFD was repeated by this independent review. The source's earlier b65 revision had an undefined inventory variable in execute; its acceptance was withdrawn before execution. The executed aabff4 revision fixed local binding, with7 independent CPU fixtures and default preflight passing. Fixtures cover the copy seam, not the entire solver execution; the present real replay supplies the runtime evidence.

For the proposed extension, inherit E109 t328 output69 and its distinct double previous action, preserving both branches' backward restart fields. Report full new(328,408] as primary with four20-unit blocks; joined(248,408] and(268,408] are separate summaries. Existing six-window conclusions and physical criteria are unchanged. This review does not itself authorize extension execution.
