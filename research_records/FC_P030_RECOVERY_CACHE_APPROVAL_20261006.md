# P030 bounded cache recovery approval

Lead approval after confirmed first P030 terminal failure (MainPID0/exit1),
empty Docker running-container inventory and physical free27,850,375,168bytes.
This is below the unchanged startup30GiB and CUDA preflight31.3014GiB levels.

Authorize one execution of
`artifacts/fcp030_recovery_cache_20261006/advise_p030_verified_train_cache_once.py`.
It verifies reviewed helper SHA98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c,
checks the pinned audit and fuser for all44 exact training HDFs, then routes the
unchanged helper's exclusive receipt to this new artifact directory. The helper
checks confined read-only descriptors, same-file SHA/stat before and after,
and original20GiB guards, and issues POSIX_FADV_DONTNEED only on those44 files.
No global cache operation, deletion, dataset/model writes or extra pass.

The failed run and immutable v2 remain intact. This authorizes only cache
maintenance, not a model execution, changed metric, or scientific conclusion.
Root read the entire helper and new routing wrapper before this approval.
