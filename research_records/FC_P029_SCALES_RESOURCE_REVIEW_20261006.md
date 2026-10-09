# FC-P029 actual parent scales and resource probe — independent review

Engineering completion verified; **not scientific admission or training authorization**. This review reads existing JSON/source/execution evidence only, without launching GPU work or re-reading model/HDF payloads.

## Actual parent-scale calculation

User unit `fluid-control-fcp029-parent-scales-20261006.service`, invocation `dc049ec56a1c4b67be118ad05a68bf92`, reached retained active/exited, exit0. Actual official container `564765b3ec81f92f1c0861fd89189df4748089eb5557e01dc011aedc8bd44944` exited0/noOOM. Approval SHA `d72d89f8ff43b23ca4a35a65a00a199a83a4f9b683bbcbe6fca6421c3404173a`.

Result `artifacts/fcp029_parent_scales_20261006/payload/result.json` SHA `05c71e723a73457de3bc3bac6539455ff3057f58b861affd8dd7d8f7d55782ef` contains1368 unique original train-window identities. These exactly match the1368 actual window log events in order. Independently recomputing `math.fsum(raw_loss)/1368` reproduces both fixed denominators exactly:

- Field: `0.001456146538716282`.
- Four-force: `0.003364271827125755`.

Source spec equals actual approval; original423-source/config/parent/data/protocol bindings match the independently reviewed preparation. Actual updates0, optimizer_created=false, model_saved=false; flow initial/terminal tensor digest identical `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`, frozen aero `b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb`. Inventory remains1368 windows,1300 warm/68 padded, family counts720/408/240. The runner verifies original sampler SHA `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`.

Elapsed internal time530.084s; minimum internal free/CUDA23.298351GiB and available111.685814GiB. This was forward-only, not a gradient or optimizer-capacity test.

## Actual H10 field-plus-force backward

User unit `fluid-control-fcp029-resource-probe-20261006.service`, invocation `6e6ab966328647d888e88adcbf78dbe2`, reached retained active/exited, PID0, exit0. Exact initial/terminal Docker ID `cef65f08d9c7544a29973c8324b281e2b19ef4b64c23ed4f4cfb420e77955cba`, official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`,12GiB container, no OOM; finished `2026-10-06T02:44:29.823775339Z`. Actual command binds resource-only approval `a8ad6be4046e6383d0dc728f3c710d721a6cf51761d80d3443ffe4f69e9b8ea2` and the frozen runner.

Result `artifacts/fcp029_resource_probe_20261006/payload/result.json` SHA `48e356dcda21fb70b74129aa611744e693973ae09d2f6cb7929ea00d7066f161`. Its source_spec equals approval. Source423/config/44 train-file map/parent/train-audit/protocol agree exactly with scales; fixed scales and prior result hash are bound. No fresh data-byte scan is claimed by this review; the executed runner retains the previously reviewed data verification contract and train-only mounts.

Exactly original window816 (`dynamic_train8_b00_prbs`, start90, dataset1) completed H10 backward. All30 recorded flow-parameter gradient norms are finite and nonzero. Raw field loss .008573819883167744, raw force .016131369397044182; fixed-scale contributions2.944009780883789 and2.3974533081054688, total5.341463088989258. The record correctly distinguishes nine force terms with a flow-gradient path from all ten field terms; final q10 is not force-supervised. Denominators are cast to objective FP32, explaining the recorded scalar values differing slightly from the JSON double means; weights/protocol are unchanged.

No optimizer/update/model save; flow initial/terminal digests and frozen aero digest match the scales pass. Unused decoder force parameter rows are preserved; unchanged unused output values are not claimed. Singleton observed-order SHA `bde2a44e3c6da9f32809c2f151e65342bc9e255bdab951bc107967bcb7ca891b` is not misrepresented as the full1368 order.

## Memory interpretation for a possible separately approved171-update run

Probe internal initial free32.109455GiB; minimum internal free/CUDA21.090473GiB and available109.489925GiB. Independent outer CUDA guard minimum21.073662GiB, exit0; host watcher minimum free22,631,800,832 bytes. CUDA peak allocated2,986,973,184 bytes (2.781836GiB), reserved3,409,969,152 bytes (3.175781GiB).

Adam first/second moments require **377,782,264 additional bytes (.351837GiB)** as a lower bound. Subtracting only these from the sampled outer minimum leaves approximately .722GiB above the20GiB floor, before optimizer temporaries, changing cache residency and other process variation. The reserved-minus-allocated gap is allocator cache, not guaranteed physical headroom. Foreach/gradient-check temporaries can add parameter-sized allocations (~.176GiB per full parameter-equivalent); their actual peak is not measured here.

Training backpropagates each window before processing the next, so eight-window accumulation does not retain eight H10 graphs, but Adam state persists across171 updates. Additional clean-cache headroom before startup is useful; an extra .7GiB is not a proven bound or guarantee. Keep .06 allocator,12GiB container and continuous host free/available/CUDA20GiB guards unchanged. This successful no-update probe supports a guarded, separately authorized attempt; it cannot certify the full training peak or success. No further scientific diagnostic, threshold relaxation or automatic training is authorized by this report.
