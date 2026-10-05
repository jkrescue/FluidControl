# Git history incident — 2026-10-05

At commit `77083fc179e04bcba95a39c65e79c5f4754f29fe`, another agent had added
`docs/PPO_SURROGATE_REAL_CFD_INTERFACE_AUDIT_20261005.md`. While adding a
seven-line confinement check to `scripts/audit_fcp011_candidate.py`, the
surrogate agent mistakenly used `git commit --amend` against that shared
HEAD and then pushed with `--force-with-lease`. This replaced `77083fc` with
`7df324bbccdd3a475cd719024765b057c27e0cd5`.

No content was lost:

- The documentation blob in `77083fc` and `7df324b` is byte-identical
  (`git diff --exit-code 77083fc:<path> 7df324b:<path>` returned zero).
- Relative to the original documentation commit, `7df324b` adds only the
  seven-line FC-P011 completion-receipt member confinement/hash check.
- The preceding FC-P011 identity implementation remains its own commit,
  `0f765190ec49f8fce2c6a6c61b7f30b4add213ea`.
- Subsequent commit `b86745f` is based on `7df324b`; the current tree retains
  both the documentation and FC-P011 audit change.

Remediation: shared HEAD will not be amended or force-pushed again. Further
changes are made as new, scoped commits after checking for intervening work.
