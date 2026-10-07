# H pressure-auxiliary training: independent terminal review

Engineering checks PASS; original fixed-six retention FAIL. No development evaluation, PPO, or CFD was launched by this review. Default B and prior prediction conclusions remain unchanged.

## Actual execution and evidence

- Training unit: `fluid-control-p064-total-pressure-aux-h-r2-20261007.service`; invocation `7b6a95342f3c4965a2a71b0542ddf574`.
- Approval: `docs/P064_TOTAL_PRESSURE_AUX_H_TRAINING_APPROVAL_R2_20261007.json`, SHA256 `af2057df4a6b010e24ce2d318f24235bf9db5d9c4c2371c4341dc695830baae9`.
- Result: `artifacts/p064_total_pressure_aux_h_r2_20261007/result.json`, SHA256 `14487bbdb73d217a92813a21fbeec0fee8140526a7edcc8d6ddd1600ec122d94`.
- Manifest SHA256: `90c61683947170df5153ceec8236a249168dbfe1893ce559e5b8018c643149a7`.
- Independent audit unit: `fluid-control-p064-pressure-aux-h-r2-terminal-audit-20261007.service`; invocation `156d61e0b727401892503fa4b277bc67`; PID0, success, exit0. Actual limits: 8 GiB, swap0, CPU1, 120 seconds, CUDA hidden.
- Receipt: `artifacts/p064_total_pressure_aux_h_r2_terminal_audit_20261007/receipt.json`, SHA256 `14fdc40557d1e9addd88c7bc89de7b27a3ddce512930f5819d84f6613e17c5d5`.
- Checker `/tmp/audit_p064_pressure_aux_terminal.py`, SHA256 `916782bb8a31f484dc230463325088de69aadc173c518ed37cb00113bdca5aff`; separately reviewed and seven CPU fixtures passed before use.

The original transient service had been collected. Terminal evidence is the exact-invocation supervisor COMPLETE with captured limits, worker journal and unique subsequent manager completion, with no failure evidence—not fabricated retained systemd properties. The receipt's top-level `unit` contains the post-collection query; `terminal_evidence` supplies the verified lifecycle evidence. Captured training limits were 24 GiB, swap0, CPU800%, Tasks2048, 3660 seconds. Manager reported 17min20.211s CPU, 5.5G peak, zero swap. Minimum recorded available memory was 105.4988021850586 GiB.

## Independent checks

Verified 434 bound source files and input bindings, exact 256-window B schedule, 32 update records and matching journal, auxiliary-loss arithmetic, 28 Adam states each at step32, out11 checkpoint dimensions, expanded-parent initialization, two frozen lifting biases and frozen flow. Official fresh reload is a bound producer record; this audit inspected saved tensors on CPU but did not construct/forward a model or independently repeat official reload. It did not independently recompute training losses or rehash large field payloads.

Fixed-six identities and history bindings match B. Below are independently recomputed balanced means from saved component metrics, excluding auxiliary pressure loss; small differences from producer summary decimals arise from recomputation, not changed criteria.

| Original fixed-six metric | B | H | Relative change |
|---|---:|---:|---:|
| H1 balanced error | 0.003976855239898214 | 0.003913135178398382 | -1.6022725912% |
| Continuous AR balanced error | 0.008946200532894485 | 0.008982218210121573 | +0.4026030614% |

Both must be nondegrading. AR is higher, so retention FAIL is decisive under the original AND rule. These small differences are not a significance claim. The fixed 80-endpoint development evaluation has not been run and remains unknown.

R1 is retained as an engineering launch failure: missing working directory/relative supervisor path prevented entry into the worker, with zero training. R2 changed only execution paths/working directory and new run identity, not scientific source. No retraining or additional scientific run was performed by this audit.
