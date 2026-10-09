# FC-P026 formal runner — independent CPU review

## Verdict

**ACCEPT for canonical CPU integration and later arm-specific preflight.** This
does not authorize formal evaluation, GPU execution, scientific admission,
frozen-test access, PPO, or real-CFD feedback.

Reviewed staged bytes:

- runner: `afb6144fdb289dc8db7c296358a5f20d9c327d4ba6eb41edc8b64f16b0c27a22`
- focused tests: `480f055e8bb43ef4da532098ffefca33c9cbe9d80a743309fa47e02571830323`
- dev30 identity-only overlay:
  `b44407ab3e828d65f99fd079b8b0a0701706849c4f211da021a36b25d682a94b`
- production P026 dual loader used by the review:
  `3343dba367dd6e45fdc914fc321e90b94efc2d8a00553c61b025ef7d776fc2a8`
  (`P026_REVIEW_LOADER_SOURCE` was set explicitly for the isolated test path)

## Verified scope

- P026 K1/K4 manifests are validated by the exact frozen loader before the
  runner can emit `PREFLIGHT_ONLY`; incomplete roles, role architectures,
  history-module identities, protocol bytes, or checkpoint identities fail.
- The numerical base is regenerated from Git commit
  `7216214b545fbbd50b2fb5ed866f231039b06b18`. A byte-bound source-chain
  receipt permits exactly seven reviewed overlays and rejects other base-tree
  changes, missing files, or extra files.
- Both arms are deliberately Main-only sequential user services with
  `RemainAfterExit=true`; the runner strictly requires the approved invocation
  to be `active/exited`, successful, and PID-free. No remote/system-unit
  fallback is allowed.
- The original validation10, dynamic6, force-window, and development-gate
  commands retain horizons, strides, batch sizes, action mode, datasets,
  numerical auditors, aggregations, and thresholds. History profile and both
  official checkpoint identities are additional explicit inputs only.
- Containers are byte-pinned, networkless, read-only, capability-dropped and
  owned by exact container ID. GPU steps retain the 20 GiB guard. CPU audit
  steps explicitly use `runc` with NVIDIA/CUDA visibility disabled. No `HOME`
  override is introduced; reviewed cache variables use `/tmp`.
- The dev30 overlay changes identity handling only: it requires the exact P026
  candidate kind, manifest and history profile while leaving pooled metrics and
  ranking calculations unchanged.

## CPU evidence

Independent combined run: **57 passed**, Ruff passed. Coverage includes the
focused runner/source-chain/candidate failures, unchanged legacy dual behavior,
P026 loader contracts, and dev30 identity/numerical regression. Supplemental
actual-main synthetic fixtures separately exercise evaluator and force-window
history routing; they are software fixtures, not held-out evaluation.

## Still required before any execution

For each real K1/K4 arm, Lead must wait for terminal training and independently
verify the actual candidate bytes, official dual reload, terminal audit,
training invocation, source-chain receipt, exact overlay map, immutable data
identities, and a new exclusive output path. Lead must then issue a distinct
byte-bound formal approval matching the final canonical runner SHA. A passed
formal suite still does not automatically authorize PPO.
