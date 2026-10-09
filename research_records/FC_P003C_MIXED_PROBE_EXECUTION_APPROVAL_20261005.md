# FC-P003C single mixed-update technical execution approval

Lead GO, 2026-10-05 Asia/Shanghai, after independent Evaluation acceptance of
commit `06f9e8d8ba488679a24fd271b2cf74fb66382762` and Root dry-run/validator review.

Approved launcher SHA-256:
`767ae9061b012d7e65c04e9d2c804753dc0779f62762ccd6093505598ce84f5e`.
Probe SHA-256:
`887b3610b17ff77bfae7045d1b649923678dfdbaa3ae4f54aa2ad73dd1b5b4e2`.
Completion validator SHA-256:
`80dfa0240712daabeb3d76a780b218683a9caf52283e7a16ced9de3fc592579c`.

Execute exactly once on Main Spark GPU0 in the reviewed isolated container.
Use the fixed real regular batch b04_m075/start180 and paired b00:multisine,
Main-e2 parent, H100, chunk10, lambda10 and existing force weights. The sole
optimizer update may modify only a scratch in-memory model; no candidate or
checkpoint may be saved. No validation or frozen dataset may be mounted.

The system service must verify the exact launcher SHA before exec, set the
reviewed approval token, and use the exclusive output directory
`artifacts/fcp003c_mixed_loss_technical_probe_20261005`.
Allocator fraction is 0.25; continuous outer guard requires at least 20 GiB
physical unified MemAvailable. Preserve logs and any failed output; no blind
retry, no changes to live source, no automatic increase of resource limits.

Compute executes and reports the real PID, exit status, resource minima,
completion/result hashes and loss diagnostics. Evaluation independently
verifies the result. This authorization does not include full training, PPO,
CFD execution or any scientific improvement claim. Full training remains
conditional on a successful technical result and separate reviewed approval.
