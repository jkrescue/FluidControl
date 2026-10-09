# FC-P003C control-handoff compatibility decision

Date: 2026-10-05. Scope: CPU-only software compatibility.

FC-P003C uses the exact candidate kind `true_state_paired_step_lambda10` and
correctly records that formal candidate training occurred. Its post-evaluation
also uses an FC-P003C-specific native step-receipt status. Those two facts do
not match legacy adapters that were written before this candidate existed.

This change therefore permits only two narrowly scoped adaptations:

1. A new binder may verify a complete FC-P003C post-evaluation receipt, its
   full file-hash table, native dynamic/force step receipts, checkpoint, and
   no-frozen/no-PPO scope. It may then emit external D012 compatibility
   bindings. The binding must say `scientific_status_reused=false`; it does not
   recompute a metric, copy a PASS, or modify the original bundle.
2. Candidate readiness may treat `training_performed=true` as required only
   for the exact FC-P003C candidate kind. Its normalization must be verified
   through the candidate-completion-bound `training_data_sources.json`, with
   the fixed base/train8/train16 counts, strides, release statuses, manifest
   hashes, and one common supplied normalization hash. All legacy candidate
   kinds retain their existing `training_performed=false` and embedded
   normalization semantics.

This is not a threshold change and creates no scientific evidence. The
canonical endpoint, dynamic, window, and development gates remain unchanged.
Missing receipts, hash changes, wrong candidate kind, wrong normalization, or
wrong trained flag fail closed. No PPO or real-CFD control is launched by this
work; even a software READY result keeps `ppo_execution_authorized=false`.
