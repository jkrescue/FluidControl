# P064 independent-control restart audit (read-only)

Date: 2026-10-07. Status: evidence audit only; no CFD, model inference, training, or candidate-result read was performed.

## Scope and immutable evidence

The eight selected restart phases are defined by `artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json` (SHA-256 `6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603`). The corresponding curated split metadata are:

- `data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json`, SHA-256 `1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e`;
- `data/curated/tandem_cylinders_matched_start_full40_v1/splits/frozen_test.json`, SHA-256 `1d13ee46cf697962169c98ab35d7bb8e310b18f9e266c63d084c2e57e495728a`;
- `data/curated/tandem_cylinders_matched_start_full40_v1/frozen_test_seal.json`, SHA-256 `0a78a0768c2f07c13393c6ad198fc23c6f553b5fc05731cf2103f55d3a6c0e65`.

Repository metadata and experiment records show the following selected-bin roles and subsequent opening/use:

| bin | restart | declared role | recorded opening/use |
|---|---:|---|---|
| b00 | 148 | train | controlled training data and multiple physical-control runs, including FC-E083/FC-E086 |
| b01 | 130 | validation | FC-E059/E085 physical-control evaluation and later reproductions |
| b02 | 106 | train | FC-E089 physical-control acquisition, followed by E090 train conversion |
| b03 | 144 | frozen_test | FC-E060 fixed-action H1–H5 opening and later paired-800 physical-control trial |
| b04 | 120 | train | original train split |
| b05 | 102 | validation | validation/model-development evidence |
| b06 | 134 | train | original train split |
| b07 | 110 | frozen_test | FC-E060 fixed-action H1–H5 opening and later paired-800 physical-control trial |

The frozen-phase opening is explicitly documented by `docs/SHORT_HORIZON_FROZEN_CONFIRMATION_TERMINAL_REVIEW_20261006.md` (SHA-256 `6c6c883396a142c875e0d50758be5a10646f8ec817ed7f9cc15dac522b0ea20c`). Subsequent physical b03 and b07 reviews are `docs/EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_TERMINAL_REVIEW_20261006.md` (SHA-256 `0d48a7e914ec82ad682d374e6531f2aab6a305aa81dad2dde6cd5c23dca48a0f`) and `docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_TERMINAL_REVIEW_20261006.md` (SHA-256 `6a76bbb74673dfdfdba57746471a836ce741f4855206efb5605677934133e953`). Thus none of the eight selected bins remains an unopened independent holdout. Historical labels remain useful provenance, but an opened `frozen_test` bin must not be relabelled as unseen.

## Unselected restart 124

Restart 124 is the second-nearest stored alternative to b01 in the phase manifest, with phase `0.9086797072991893`, state SHA-256 `d09d556a737119e0d746c0dd46d6bb7de03de39fe4b7d407c097eec8acf287f2`, U SHA-256 `9ed4956297d26eacf1d4002b6d5c4f6b107de4f99af24b908907aa7d0d93cf5c`, and p SHA-256 `fc14e99eaa86dde1c99219633b1dcc111a016216f80e9f67d3c4f4af0f264364`. Exact repository search found no experiment record that used this restart.

This absence is not enough to call restart 124 a holdout. It was not selected into, sealed as, or prospectively registered as an independent split; it is also a nearby state on the same baseline limit cycle and b01 phase neighborhood. At most, a future preregistered experiment could describe it as an unused restart-state robustness replication. Its expected information gain is limited, and this audit does not authorize that run.

## Policy and protocol if a future validation is preregistered

Any future independent-control validation must freeze the tested policy before opening results. The retained delivery baseline is the E082 B canonical policy `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/ppo_final.zip`, SHA-256 `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`, with canonical adapter SHA-256 `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`. B/G or other policy selection after viewing a new result would invalidate the intended independent comparison.

The existing Re=100, L/D=5, `dt=0.005` paired-control machinery is technically reusable for another fixed restart, but reuse would require a new exact restart/config/source manifest, prospective policy binding, fixed endpoints and acceptance interpretation before execution. Engineering similarity does not make the restart an existing holdout.

## Conclusion and limits

There is no genuinely independent, already selected, unopened fixed phase in the current eight-bin library. A defensible independent validation must be newly preregistered before any result is opened; it cannot be manufactured retrospectively from an unselected library member.

This audit searched repository manifests, tracked documentation, experiment ledgers, and recorded artifact identities. It did not inspect candidate field/force outputs. Repository search cannot prove that a file was never accessed outside recorded workflows or establish OS-level read history, so the conclusion is deliberately limited to the auditable project record.
