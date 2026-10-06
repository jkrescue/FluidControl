# FC-E087 C50 training terminal independent engineering review

Verdict: engineering ACCEPT, not prediction admission. The C candidate may be bound to the separately approved fixed development evaluation; no accuracy improvement is established by training completion.

The actual R2 unit `fluid-control-p064-controlled-dose-c50-r2-20261007.service`, invocation `d80f61c62da64497bf378b6c7fd9c917`, ended with PID0, normal exit0 and Result=success. Approval SHA `6d0d0f8dbdbae17a89d3b7dcc1717145b8e5a44464e928b5cb1a4e6debf2800f`. R1 invocation `9595da6d9092476298f597ef8ed08fb2` remains an import/PYTHONPATH engineering failure before model loading or training; R2 changed the execution environment, not the scientific source.

## Independent checks actually executed

The reviewed A/B checker `8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587` was reused with only the C scheduling extension: 128 of 256 windows, replacement positions [0,2,4,6] per update, evenly integer-spaced starts 0..700. The adapter SHA is `13f4b86c7c089159ffac76312cf1a13025b2cfdae6370cc23aa63c5ef8033a2a`; five CPU fixture tests passed. Actual CPU review unit invocation `dcc80c3d899a49899c0e15cf41d04f23` exited0 under 8GiB/noSwap/one CPU/120s with CUDA hidden. Candidate access was gated on actual training PID0/exit0.

- Rehashed all 439 source files, direct source/input identities, original data manifests and normalization, parent model identity, and both roles' saved model/checkpoint files. Frozen flow files match the K1 parent bytes.
- Reconstructed all 256 actual journal windows and 32 update events, including their interleaving. Each update contains eight windows. All 128 controlled starts and original-order indices match the fixed C schedule; saved record identities match the journal. This is not a strict superset of B's 64 starts.
- Recomputed every recorded mean objective and clipping scale; checked finite result values and the exact 32/256 protocol. Protocol schedule SHA `93537e23ce606732dfd48e78a3b92def987918c0cb71b1b8ca7e8126564081d1`, b00_weight=0.5, K1 parent, same learning rate/budget/objective, no validation/test/selection.
- CPU weights_only checkpoint inspection confirmed all 28 Adam states are finite and at step32, with the fixed hyperparameters. The two frozen lift-network bias tensors equal the parent exactly; candidate model tensors are finite.
- Official fresh-save/reload succeeded in the actual producer. This review checked its bound result and artifacts, but did not independently instantiate/reload the model or run forward inference. No HDF arrays were reopened or rehashed.

## Bound evidence and resources

Output: `artifacts/fcp064_controlled_aero_arm_c50_20261007_r2`.

| Artifact | SHA256 |
|---|---|
| result.json | b130f2a6a92247ee70eff10a4c35ce3670514faee9f50bb76122d34e97098e0a |
| dual_model_manifest.json | b4a150370101b4214b5085c8884516fbf44aa1560f11f527f85fc1811bf65529 |
| artifacts/fcp064_arm_c_terminal_cpu_review_20261007/receipt.json | 7e6ad1426a37d29acc6d0f0a110201d73b8d676803692d5a3914b856a22346ad |

Actual training MemoryMax=12GiB, MemorySwapMax=0, MemoryPeak=5,942,624,256 bytes; recorded minimum MemAvailable=106.50679016113281GiB, above the 22GiB runtime threshold and 20GiB reserve. Host-venv execution, not Docker. Historical high/TF32 training precision is recorded and unchanged.

This is FNO aerodynamic-network training, not PPO or CFD. Existing two specified-seed physical-primary successes, early physical failures, and the prior complete prediction failure remain unchanged. Fixed b01/b03 development inference requires separate execution authorization; training completion supplies no scientific PASS.
