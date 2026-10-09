# FC-P026 PPO-to-CFD export compatibility review

Software compatibility only. This does not authorize PPO, CFD execution or
scientific admission, and does not change the running K4 training.

## Defects and bounded correction

The candidate export adapter classified only P013/P015/P018 as dual-model
candidates. A valid P026 K1/K4 command contract therefore reached its
single-model rejection branch despite containing the required dual binding.
The real-CFD runner also omitted P026 markers from its fail-closed dual-policy
detection: a P026 kind without a binding could return the legacy empty binding.
An isolated CPU function check reproduced that omission for both arms.

The adapter now explicitly recognizes `FC_P026_K1_HISTORY_FORCE_FNO` and
`FC_P026_K4_HISTORY_FORCE_FNO`. A lightweight JSON-only helper checks history
profile, exact scalar types, channel layout, causal flags, candidate identities,
and the existing state-history/inference source pins. Runtime copies must agree
with the dual binding. The actual upstream-verified protocol-file byte SHA is
preserved; it is not replaced with the inventory's canonicalized JSON hash.
Malformed nested bindings are rejected with ValueError.

The CFD runner recognizes P026 kinds, profiles and history-runtime markers and
calls that helper before reading/deserializing VecNormalize or staging CFD.
Existing policy/data hashes, identity VecNormalize, 69-dimensional physical
observation contract, action limits, reward, scientific gates and resource
guards remain unchanged. Legacy behavior remains covered by regression tests.

## Exact reviewed source identities

- `scripts/adapt_candidate_ppo_openfoam_readiness.py`:
  `f78f027bcf51683129633d1851fe17047d75a65e07ef7be40e5cf0816a898927`
- `scripts/run_full40_canonical_ppo_openfoam_feedback.py`:
  `3dbb20106f45ed709e0c91731817540903db961555930177f81f0394104594cc`
- `tests/test_p026_feedback_compatibility.py`:
  `ebb6a483c89e6023fef34a2f7bc8c5193122e7e576aafbc89704d558cce7088c`

The runner already inserts `PROJECT/scripts` into its import path, and its
existing execution layout exposes the project read-only. Its lazy import of
the adapter helper therefore resolves in the actual canonical layout without
adding a model import or an invented external API. The low-level readiness
stores history runtime inside the dual binding, whereas the command contract,
candidate identity and PPO audit also carry an explicit copy; validation respects
this existing distinction.

## CPU evidence and limits

Implementation-stage tests: 55 passed in0.16s. Independent reviewer: 55 passed
in0.18s. Root's canonical rerun after integrating the exact three files:
55 passed in0.25s with CUDA hidden.

Tests cover K1/K4 export and host-path translation, omitted dual markers,
conflicting histories/identities, wrong valid-length source hashes, bool/int
coercion, malformed nested binding, and legacy regressions. The positive
synthetic export reaches the existing missing-VecNormalize refusal in the CFD
validator. It does not deserialize a real trained policy, validate a real
VecNormalize artifact, execute PPO, load an official FNO, read HDF data, or run
CFD. These tests must not be represented as an end-to-end deployment result.

Future execution still requires an actually accepted candidate, a newly trained
compatible policy, its exact identity-normalization artifact and separately
approved real-CFD validation. The deployed 69D policy consumes actual CFD
probes/forces; this patch neither requires a full-field CFD bridge nor implements
online FNO inference in that CFD loop.
