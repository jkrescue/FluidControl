# G-policy canonical b01 CFD: actual single launch

This records an exploratory physical test, not G prediction-model admission.
G's original selection FAIL and the existing B controller remain unchanged.

- Unit: `fluid-control-p064-g-symmetry-canonical-b01-cfd-20261007.service`.
- Actual invocation: `c14a66a1464a4919a3904c1f4d2efb2d`.
- Initial PID:2110235; actual start2026-10-06 23:10:24 UTC, active/running.
- Approval: [P064_G_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json](P064_G_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json), SHA256 `f72cace2befb6aef72ed6878ec83d675cc92b4992ec0b9e6e45cd38be97bbb48`.
- Driver (repository-relative): `artifacts/p064_g_canonical_b01_cfd_source_20261007_immutable/run_p064_g_symmetry_canonical_b01_cfd.py`, exact SHA256 `23bac1e31122cf227d37c4b50df05b7b5b487cbc6ec35e758e2d2386da32c10a`.
- Output (repository-relative): `artifacts/p064_g_symmetry_canonical_b01_cfd_20261007`.
- First inspected actual progress:29/800 cycles, time132.9. This is not terminal evidence.

Lead's conditional execution approval was satisfied by the successful same
G PPO invocation6aa96fbfeeb34269b1f49e04380cd417 and independent terminal review
SHAca025b858ba12ae267a58e4a237574ee01bdf26f1fb4a42203d364327a4c767f.
Final policy SHA c1157806b2efc54fcf979df4734e09e846068908562f5415af1a43775881821b
and Vec SHA043125f7b6ccab8bea70cbee1b43c6797c747ce0a107b088d2fad9d5bb5e5de9
are bound in the approval. Sota independently accepted all actual input hashes,
training identities and unchanged shared canonical adapter before launch.

The final serialized metadata preflight passed exact10 source/10 input hashes,
actual completed training unit,192 training runtime hashes, source130/constant/
system inventory, solver image, exclusive output/unit, disk>=20GiB and startup
Available>=50GiB. The driver self-hash is separate from its10 dependency map.
The earlier preparation map's redundant driver entry was removed before any
CFD execution; a sixth CPU schema fixture checks this fixed source set.

Scope remains130->210/800 paired cycles, original six windows, primary(150,210],
drag>=2%, rear-lift RMS ratio<=1.05 and mean-bias ratio<=.10. CPU policy inference
only; no online FNO, no new training. One canonical prediction/sign restoration
and one existing physical rate filter. The newly solved zero branch must be
compared with the original B b01 full raw arrays after termination.

Limits:8GiB/noSwap controller CPU400%; each solver8GiB/noSwap CPU2;
runtime Available>=22GiB with20GiB reserve;3600s inner/3750s outer/120s stop.
No retry, horizon extension or checkpoint substitution is authorized.
