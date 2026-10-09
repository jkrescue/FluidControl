# Six-window current-force timestamp audit

Persistent experiment record: **FC-E031**. Primary evidence now resides in `artifacts/causal_force_input_audit_20261005/`: timestamp_audit.json SHA `72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2`, persistence.json SHA `c6f71f1ad90b6c2b952ce5fa0ba4752e4c3bd2bfc8a34e5388f75751817ce146`, official_source.json SHA `e8c6ea9feaaf0d7828d3c976891c70ea8e8028c0d1b2a5639127571c5177aec1`. Temporary paths below retain the original investigation provenance.

Independent reviewer recomputed the complete timestamp-audit dictionary in memory from actual raw sources, omitting only file-writing and printing: exact equality, including274 freshly hashed raw/config files, unique606 endpoints per cylinder, interpolation brackets and reconstructed values. Observed exact raw-to-nominal timestamp difference is zero. Source fallback appears only at initial frame for windows975/1077/1233. Root subsequently authorized isolated P021 CPU engineering in commit`225d99f`, not GPU execution, training or deployment; see `FC_P021_CPU_ENGINEERING_APPROVAL_20261005.md`. This updates the earlier pending-authorization wording below without changing scientific scope.

Read-only CPU work, 2026-10-05 UTC. No HDF, targets, normalization, models or repository files changed. This is input provenance evidence, not admission.

Evidence: timestamp_audit.json SHA 72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2; producer timestamp_audit.py SHA c3d97ba8ac5b85d505086da18509e392369df1704a936b4375da90af430b0047. Both reside in /tmp/force-conditioning-review.hMRnOP. Input six-window force/time extraction is persistence.json SHA c6f71f1ad90b6c2b952ce5fa0ba4752e4c3bd2bfc8a34e5388f75751817ce146. HDF hashes remain inherited from pinned prior audit; no bulk HDF scan repeated.

## Actual result

All606 nominal endpoints (six windows x101), for each cylinder, have a UNIQUE exact raw coefficient row with atol1e-8, rtol0. Nominal time is source_restart_time +0.1*absolute HDF frame index. Per-window origin, frame, nominal/stored/raw time, source filename, source/restart designation, and all606 exact four-force values are recorded. Every read coefficient file and case_config.json has freshly computed SHA. Duplicate matches fail; no nearest-neighbor fallback.

The base20 window and FOUR train8 windows reconstruct BITWISE from np.interp at stored float32 field times, but each has40/101 nominal endpoints whose positive interpolation right bracket is the NEXT solver sample, nominal+0.005. Total200 such frames among505 base/train8 endpoints. Stored time quantization is at most6.10351563e-6; next-sample weight is at most .001220703126 (b02 .000610351563). This small weight remains a real future dependency; 1e-5 timestamp matching tolerance does not make the next solver sample causal. Largest observed HDF-versus-exact rearCl difference is1.24387443e-5.

The train16 window reconstructs BITWISE from exact nominal endpoint rows instead, with zero actual HDF future dependence. Its hypothetical interpolation-at-stored-time would also have40 future brackets, but that is NOT its implemented curation; JSON names distinguish hypothetical and actual counts.

All six selected INITIAL frame values match the unique exact raw endpoint after float32 conversion, with zero difference and no future dependency. Frames0 of windows starting at restart use the baseline source row at precisely the restart time when the branch file starts at restart+0.005. Other selected starts resolve to their own branch row.

## Source mechanics

scripts/curate_tandem_cfd.py:304-324 inserts source-restart row, then interpolates coefficient series onto float32-origin VTK times. cfd/tandem_cylinders/curate_dynamic_train8.py:208-216 maps reviewed run windows into that source, and its manifest records same-field-timestamp np.interp. The artifact reconstruction above verifies actual values, not merely intent.

cfd/tandem_cylinders/curate_directppo_train16.py:185-204 canonicalizes VTK TimeValue to declared endpoints within1e-5, retaining fields; :258-287 selects only unique raw endpoint rows at atol1e-8, including source restart; :432-436 verifies curated force values. Its actual101 values match exactly in this audit.

## Minimal causal-input remedy; inherited target contract unchanged

Do NOT reuse raw HDF force frames[:-1] without correction for new conditioning. A new explicit train-only causal-input sidecar/adapter can use the606 exact raw endpoint four-force vectors in this receipt, normalized by existing all_force_mean/std. Bind sidecar content SHA, raw source SHA, six identities and nominal timestamp origin/interval. H1 uses exact measured force at current nominal endpoint; AR uses only initial exact force, thereafter its own prediction. No future target enters AR; do not reset at checkpoint boundaries.

Keep existing HDF targets, field values, losses and normalization byte-identical, explicitly recording the small input-versus-inherited-target alignment difference. This is a new input contract, not retroactive recuration. Existing persistence.json H1 is the descriptive HDF-lag baseline and therefore inherits the tiny future interpolation dependence identified here; it must NOT be advertised as a strict online-causal H1 baseline. AR constant-initial baseline is unaffected because all six initial values are exact. A future approved conditioning test should recompute its H1 persistence reference from the exact sidecar while retaining original next-step HDF targets.

For later full training/formal/runtime, this evidence covers ONLY these six windows, not all44 trajectories or heldout cases. The same unique-endpoint contract must be proven for every used current-input timestamp under the appropriate approval. Do not silently extend this606-row result to all data. Real CFD measured current force remains the completed current solver endpoint, not interpolation requiring a future sample.

## Implementation approval prerequisites now

Root separately resolved official encoder ordering from pinned image source SHA e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9 (official_source.json in this stage): physical0:6 stays0:6, old appended coordinates6:8 move10:12, new force6:10 weight columns zero. No prefix-copy of old8. This read-only timestamp investigation resolves data availability, but does not itself authorize adapter/model implementation or resource execution. Full recurrent-gradient/checkpoint and dual20GiB resource prerequisites in the design remain.
