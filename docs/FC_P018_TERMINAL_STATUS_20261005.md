# FC-P018 terminal verification — not scientific admission

Observed 2026-10-05. Existing tandem-cylinder scope and acceptance criteria are unchanged.

- Exact training invocation `1ca4654aab074278bb2efdfff8dbc1eb` ended successfully: retained active/exited, PID 0, exit 0.
- 171 AdamW updates / 1368 windows, 44 real CFD training trajectories, LR `1.5625e-7`, fixed order and protocol verified. Frozen flow and two prescribed lifting biases unchanged. Independent audit reproduced the complete saved audit.
- Completion SHA: `bce5fb4688e6fc784c309a4e63979a06603270c8b1e5b336572a1092306205bf`; audit SHA: `03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`; result SHA: `5ca668810110676dc500f501fb602a3e2fbc1cd9a266ad6ad745484be403f926`.
- Host minimum MemAvailable/MemFree: 106.618/24.190 GiB. Training resource guard exited 0.
- Actual official CPU dual reload succeeded, no forward, optimizer, GPU or model save. Receipt SHA `d36258772d50db8ecdde1bd6437a0abee407df45f778f56830f038215391bdde`; external Docker evidence in `FC_P018_CPU_DUAL_RELOAD_EXECUTION_20261005.json`. Two prior attempts failed on read-only default cache directories; both containers/logs retained. Only `XDG_CACHE_HOME` and `LOCAL_CACHE` were redirected into temporary container storage for the successful attempt.
- Frozen formal chain: `artifacts/p018_posteval_chain_d46622b7cb51_immutable`, receipt SHA `6a97e0b4792f2e15156694b91e148267f577ac6e4eaf0f134caad1dae50efe3f`. Staging and supervisor preflight passed; this does not mean formal evaluation executed.

## Fixed training diagnostic — limited interpretation

Same six initial diagnostic windows exactly reproduce the P015/P009 initial baseline. H1 objective changes from 0.003651720137 to 0.003522910670 (-3.53%); autoregressive objective changes from 0.008840553239 to 0.008851685920 (+0.126%). For the five nonzero-action windows, tail62 centered residual MSE increases 2.265% (H1) and 0.165% (autoregressive). Centered residual MSE is not the lift-amplitude RMS error. These train-only observations do not establish convergence, validation performance or control benefit.

## Next action

Execute the original complete validation10, dynamic6 and force-window protocol after independent execution review, with both MemAvailable and MemFree >=20 GiB. Preserve all results. Only complete original admission permits compatible PPO training followed by real-CFD paired feedback validation. Otherwise use measured failure evidence for the next intervention, without relaxing thresholds. No new PPO or surrogate-assisted CFD success is claimed.

Control-interface preparation was integrated in `5820a98`: 197 CPU regression tests passed including P013/P015 behavior, P018 protocol and execution bindings. Software tests are not scientific admission.
