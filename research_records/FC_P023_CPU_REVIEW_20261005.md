# FC-P023 isolated96-coefficient CPU verification

Scope: engineering prerequisite only, not production GPU training, learned
accuracy, admission or closed-loop success. Root and independent reviewer read
the adapter and tests; final9 CPU tests passed (Root1.25s).

- Module `p023_force_block.py` SHA367a532b395e2cef06332990ffcbfc769677102858f0ed3a46bb28689d7b92d2.
- Tests SHAf340ca0e8d4fe10e3f802e952102af2eb1afe6651b3878c1400f130ae5b7fd0b.
- Final official CPU script SHAfe4cda1734bf5d31a9a85d28291fab1ee64d8b45169b986cd0a48b2e9609e477.

Root actually executed the final script in pinned official image b40d5888,
CPU-only,2cores,4GiB,read-only/network-none,120second bound; exit0. Exact observation
and values in `FC_P023_ROOT_CPU_EXECUTION_20261005.json`. Prior attempt failed
before Python because a nested mountpoint was missing in the read-only directory;
recovery copied the exact hashed helper into staging and used one read-only mount.
No model/source/data changes were made to solve this mounting issue.

Tiny official model: latent48 (actual24x4x1x1 block),5FNO layers,modes2,grid4x4,
decoder8. Nonzero synthetic block=.01 is ONLY an engineering fixture, not a
real-data initialization policy. Production comparison must initialize block zero.
Observed max absolute differences: zero-block vs six-input parent0; functional
vs materialized output0;96block gradient0; input gradient0; fullH100 checkpoint
vs fullgraph output0 and block gradient0. Declared output tolerance1e-5/1e-6 and
gradient tolerance1e-4/1e-6 were not relaxed. H100 block gradient norm2.3940138e-6;
initial-force perturbation changes firstAR output7.7587552e-5 (nonzero).

Synthetic AdamW updated one96-element state; all old parameters/buffers remained
unchanged, finalzero restoration verified, CUDAavailablefalse. Adapter uses full
strict functional mapping and freshly assembled differentiable lifting tensor;
old parameters have neither gradients nor optimizer state/decay. Construct after
finaldevice/dtype placement; caller must verify official source pins.

Next: prepare a separately reviewed bounded real-data LOW/HIGH step-size trainer
under `FC_P023_INPUT_BLOCK_CPU_PLAN_20261005.md`. No GPU run is authorized by this
engineering record. Keep both20GiB floors and unchanged scientific metrics.
