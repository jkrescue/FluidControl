# G AR5-reset training: independent terminal engineering review

Conclusion: **ACCEPT for the separately authorized fixed development evaluation; not scientific admission or controller promotion.** No model forward or additional training was performed by this review.

## Actual execution and bindings

- Training unit: `fluid-control-p064-ar5-reset-g-20261007.service`; invocation `a8e2f0a18136461c99e8154df953aa7e`; independently observed MainPID=0, Result=success, ExecMainStatus=0.
- Approval: `docs/P064_AR5_RESET_G_TRAINING_APPROVAL_20261007.json`; SHA256 `0b01cc385142b188482fa33cf4c1185fb6048a237ddb39b748eb7f80a1514a9a`.
- Output: `artifacts/p064_ar5_reset_candidate_g_20261007`.
- `result.json`: `d6b406038f452806b4852c36cf818d43600684fa0f7a034dfc7460bf722e5581`.
- `dual_model_manifest.json`: `6123b587065ad46104c328f276949ba07d3d7b61266b6c781cb5e9a538cfd681`.
- `training_protocol.json`: `565a47c5c6751fdf1809b46ce3b43b6b4dc4c40a3de228916fda227c15ab41f7`.

## Independent checks

The bounded, CUDA-hidden CPU checker verified terminal state before opening candidate checkpoints; 434 original source files plus all three G overlays; bound input identities; 32 update records, 256 ordered training windows and matching journal events. Checkpoint inspection used weights-only CPU loading: all 28 Adam states have step 32. Both frozen aerodynamic biases and the flow checkpoint remain unchanged. Producer official fresh reload is verified in its bound result; this audit did not independently instantiate the official model or recompute training losses.

All 256 training state-audit entries match their recorded window identity/state hash and require reset indices 0,5,...,95, 100 flow calls, 100 supervised AR points, 20 discarded block-end updates, no future-state inputs and no optimizer inside the state helper. Training retains equal H1/reset-AR5 weighting. The saved mean training objective is 0.010668747818954216; this is recorded-loss arithmetic, not an independent forward recomputation. The fixed-six before/after diagnostic panels retain **continuous AR100** and the original half-H1/half-AR diagnostic total; they were not silently changed to reset-AR5.

Actual training MemoryPeak was 4,195,954,688 bytes, with 12 GiB MemoryMax and zero swap. Minimum recorded MemAvailable was 106.59253311157227 GiB, above the unchanged 22 GiB runtime guard. Dataset payload arrays were not rehashed or reopened in this engineering checker; its receipt reports that limitation explicitly.

## Checker repair and actual audit

The first audit invocation `a07c2f88e94f4a93824e44367c3ab5e5` exited 1 on an absent redundant approval key after core checks. This was an audit-schema error, not a training failure. The original failure is retained. The minimal fix keeps supervised points hard-checked at 100 in the protocol and every window, and binds the reset-module hash to the actual approval argv file rather than an absent duplicate approval field. Ten CPU fixtures passed, including absent-key and wrong-99 rejection cases; Sota independently accepted this change.

Corrected checker SHA256: `3685311b95c33187bf90f38b1f90f0389f84fa0a7ac736d8009820fa9988f603`; tests: `ee19a0a5d3e14aa9023adb4a326fc4e1f1af90ea696a40f0e76fcbdfee7dcf18`. Original pinned checker: `8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587`.

Actual R2 audit unit `fluid-control-p064-ar5-reset-terminal-audit-r2-20261007.service`, invocation `9544954f08154befbd01e987fce35455`, completed with MainPID=0, Result=success, ExecMainStatus=0 under 8 GiB/no-swap/one CPU/120-second limit. Its journal contains both the original engineering receipt and G-specific 256-window reset checks. No training source was changed and training was not repeated.

Next: run only the already conditionally authorized fixed 16-origin/80-endpoint development evaluation and independently recompute its saved arrays against B, including fixed-six retention. Existing B controllers and all historical prediction failures remain unchanged.
