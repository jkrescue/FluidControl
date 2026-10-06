# G exploratory b01 CFD: authorized terminal-only audit plan

This is a read-only saved-output audit, not another CFD solve, training run or model forward. Root authorized one bounded CPU audit after the actual job is terminal. No running driver or approval will be modified.

- Actual CFD unit: `fluid-control-p064-g-symmetry-canonical-b01-cfd-20261007.service`.
- Required invocation: `c14a66a1464a4919a3904c1f4d2efb2d`; require MainPID0, Resultsuccess and ExecMainStatus0 before opening result.
- Approval SHA: `f72cace2befb6aef72ed6878ec83d675cc92b4992ec0b9e6e45cd38be97bbb48`.
- Output: `artifacts/p064_g_symmetry_canonical_b01_cfd_20261007`.
- Reviewed wrapper: `/tmp/audit_g_cfd_terminal.py`, SHA `9beb443485e896b892f0003f49e4c2776885b858ff452bea6c16607010fec596`.
- Original raw checker: `/tmp/audit_b02_base_readonly.py`, SHA `cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8`.
- Matched retained B reference: `artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/result.json` (E085; same restart130 and six windows).

Two CPU source regressions passed: after substituting only identities/output/approval and descriptive comparison label, wrapper AST equals the E095 wrapper; terminal gating precedes raw reading. Live first-row schema and identical b01 restart-tree metadata were checked without reading incomplete result. Sota independently accepted the wrapper.

The audit rehashes3200 force files, reconstructs16000 samples per branch/cylinder at .005 spacing, recomputes all six force/statistics windows, checks1600 clean20-step solver logs, all800 canonical observation/orientation/sign-restored requests and single physical filters,799 feedback-continuity links and raw endpoint Cd/Cl→observation64:68. It verifies progress/result agreement, restart hashes, container exit/noOOM/removal and actual resource samples. The paired zero arrays must equal E085 zero; G/B controlled observations/actions are compared descriptively, **not required to be identical**. Original2% drag reduction/1.05 rear-Cl fluctuation ratio/10% mean-bias criteria remain. Report every window, including early failures, without changing prediction-selection FAIL or retained B status. Action-square costs remain proxies, not physical power or net-energy claims.

After terminal, verify both source hashes and run once:

```sh
systemd-run --user --unit=fluid-control-g-b01-cfd-terminal-audit-20261007 \
  --property=Type=exec --property=RemainAfterExit=yes \
  --property=MemoryMax=2G --property=MemorySwapMax=0 \
  --property=CPUQuota=100% --property=RuntimeMaxSec=120 \
  --property=TimeoutStopSec=20 --property=KillMode=control-group \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  /home/USER/env_isaaclab/bin/python /tmp/audit_g_cfd_terminal.py
```

The actual audit invocation, result/progress hashes, resource outcome and independently recomputed numbers will be recorded afterward. If it fails, report the precise failure; do not restart CFD or silently rerun the audit. No need to load the policy/model to inspect saved CFD evidence.
