# FC-P029 training and official CPU reload — independent terminal review

**Integrity/engineering checks pass; scientific admission remains false.** This review launches no model/GPU/CFD computation. It reads completed JSON/log/container evidence, hashes the seven candidate files and twelve CPU runtime sources, and recomputes recorded aggregate statistics. It does not repeat HDF scans or claim scientific accuracy from training completion.

## Preserved failure and clean recovery

Attempt1 invocation `b343d13dffb6402fa91bec055cafad32` failed on the inner physical/CUDA memory guard after one update/ten windows; no payload or checkpoint existed. The entire five-file evidence tree was preserved at `artifacts/fcp029_control_aware_flow_training_20261006_failed_attempt1`. Independent rehash confirms all five files exactly match the recovery record, with no extra files. See `FC_P029_TRAINING_ATTEMPT1_FAILURE_REVIEW_20261006.md` SHA `f38bccbcaee7f007297fb789a8d105aecc61457f0f2afef35e561235deaa4eed`.

Root separately approved exact-file clean-cache operations and one fresh restart. The successful invocation `7c8b277455444df58825338a6590d684` uses the same original approval SHA `4412292ee5695ee86e6dca1facdc1436587e161199da6e5394cea582003e3566`, identical container command, parent/data/protocol, and all423 unchanged source hashes. It begins from the original parent/new Adam, not an in-memory continuation. New container `725cb5bf368255a8ba55e8e735c42de472a61267453c603853b607a7cce08bc4` exited0/noOOM at `2026-10-06T03:04:44.476152006Z`. User unit `fluid-control-fcp029-flow-train-20261006.service` retained active/exited, PID0, success/code1/status0.

## Actual training result

Result `artifacts/fcp029_control_aware_flow_training_20261006/payload/result.json` SHA `39b246d07ff673de5b3d5fdc2e65be5e46bcb466e1290f5549f2e75d5cb81336`.

Independently checked171 sequential update records, eight windows each,1368 unique identities exactly matching actual window log order. Every update has30 finite gradient norms; all171 updates invoke clipping. Recomputing each raw-field/raw-force and normalized-contribution group mean with fsum/8 matches exactly. Source spec equals the actual approval. These observations do not establish convergence: consecutive updates consume different windows.

The fixed denominators remain field `.001456146538716282` and force `.003364271827125755`, bound to the actual parent-scales receipt `05c71e723a73457de3bc3bac6539455ff3057f58b861affd8dd7d8f7d55782ef`. Updated flow tensor SHA `470a6325327fb061ec53fa3b5d9388e57b84c99a134f0c3c15df3bbd8ea7abc9`; frozen aerodynamic tensor SHA `b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb`. Unused decoder force rows are preserved; unchanged unused output values are not asserted. Training result records internal official fresh reload success.

Actual CUDA peak allocated3.318061GiB/reserved3.822266GiB. Internal resource minimum free/CUDA22.492943GiB; outer CUDA guard minimum20.907440GiB; host watcher minimum free22,000,205,824 bytes. The outer sampling includes later save/reload stages, so it is not interchangeable with the inner training minimum. Guard exited0; no claim is made that this predicts other experiments' capacity.

Candidate manifest SHA `72b52ff2b6c702c9d100cc5fbade0952875289d25e826bb6b88020cb042d9b16`, explicit `FC_P029_CONTROL_AWARE_FLOW_REPAIR` / `FC_P029_DUAL_FNO_MANIFEST_VERIFIED`. Effective protocol SHA `97af7429b48a51fb10d533ca2547150522a093780f8695ec651f2ab14cb29e48`.

## Independent candidate audit and actual official CPU reload

Candidate audit SHA `ab5b62397c5101017d2302daa01824118c9687672051f6d51b459ba29fb2906b`, status `FC_P029_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION`, binds the successful new invocation and171/1368 counts, not failed attempt1.

CPU reload unit `fluid-control-fcp029-official-cpu-reload-20261006.service`, invocation `a2350855dcfd4d328a1f63cefa701808`, retained active/exited/status0. Receipt `artifacts/fcp029_official_cpu_reload_20261006/dual_reload_receipt.json` SHA `2023daadf611e6b4fe30146f029d142b1c432c09b41e14fe1af6bc1e7f6d9f64`.

Actual initial/terminal container ID `61327910fa863c51a8992504a6b7319e88716c9174cc0c2208e1b5c3d9f957e6` agrees; official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`, runtime runc,8GiB memory, network none, read-only root, no GPU device requests, `NVIDIA_VISIBLE_DEVICES=void` and empty `CUDA_VISIBLE_DEVICES`. Only the exclusive output mount is writable. Exit0/noOOM at `2026-10-06T03:06:08.901012669Z`. Host watcher minimum free36,092,252,160 bytes and available122,132,672,512 bytes.

Independently rehashed all seven physical candidate files against the literal `candidate/` maps in both audit and reload receipts: exact agreement. Flow model/archive SHA `2f71e25532b3002955396e7e97a2714d2e8ff46a0bbaef4cc7606783312658aa`, state SHA `f2ed2cf7b4bea2fced7e05ff8277a792c867a6c5b2e56621f41bda269031f2ef`; aerodynamic model/state retain the exact K1 parent bytes. Both freshly loaded tensor hashes equal the audited/training terminal hashes.

All twelve frozen CPU runtime files rehashed correctly; runtime manifest SHA `ec8d39553250c1c38001ddcc7ead06c6a57192c00af135188f4bb6cca747ea8e`. Receipt binds the exact candidate-audit SHA/config/protocol and records official FNO source `e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9`, checkpoint source `0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e`. No forward, optimizer, model save, or GPU use occurred in this CPU reload.

## Next decision boundary

The candidate is readable and internally consistent, not scientifically accepted. Proceed only with separately approved matched H10 comparison and unchanged complete original formal evaluation. No automatic policy training, inherited policy readiness, relaxed threshold, or CFD-control improvement claim follows from these engineering checks.
