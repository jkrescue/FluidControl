# Physical y-reflection: independent read-only CPU precheck

Preparation evidence only. No model construction/forward, optimizer, GPU, CFD, dataset modification or new training authorization. The checks below support a physically motivated train augmentation, not new CFD truth or exact discrete-mesh/model equivariance. No incompatible case was found within this bounded scope.

## Actual grid, mask and normalization

Read only the small `x`, `y` datasets and first `mask` frame from all45 train HDFs:44 original trajectories in `artifacts/p064_host_train_only_views_20261006/{base,train8,train16}/train/*.h5`, plus `artifacts/b00_controlled_train_dataset_view_20261006/train/*.h5`. Every128-element y vector satisfies `y + y[::-1] == 15` exactly (maximum residual0); each first mask has shape(1,128,256) and equals its y reversal bitwise. This is a first-mask/grid check, not a reread of all state/mask frames or full HDF payload rehash. The transform must leave destination x/y coordinate features unchanged while permuting physical state samples along the y axis.

Actual original normalization is nonzero in odd channels: v mean `4.374019421184832e-7`, front-Cl mean `-7.255949200950202e-5`, rear-Cl mean `-0.00013177838209514547`. Thus normalized sign negation alone is invalid. Denormalize→physical flip/sign→unchanged normalization, then exact mask-zero is required. Pure physical permutation/sign is bitwise involutive; float32 normalization roundtrips need a predeclared numerical tolerance and measured maximum error, not false bitwise requirements. Actions/mask/coordinate identity stay exact.

## Actual boundary scope

For each of the44 original HDFs, resolve its `case` attribute to `cfd/tandem_cylinders/cases/<case>/`; inspect the earliest numerically ordered saved time containing both U and p. Selected times are106/120/134/148 for base/train8 and106/148 for train16. All44 actual U boundary sets are: inlet fixedValue `(1 0 0)`, outlet zeroGradient, upperLower slip, frontCylinder noSlip, rearCylinder rotatingWallVelocity with origin `(15 7.5 0)` and axis `(0 0 1)`, frontBack empty. All44 p sets have zeroGradient at inlet/upperLower/cylinders, fixed outlet0 and empty frontBack.

The controlled b00 source is independently traced through its conversion approval to `artifacts/exploratory_projected_32768_ppo_long_cfd_20261006/case_mpc/148.1/{U,p}`. These have the same compatible boundaries and a nonzero rear omega table. This verifies actual controlled boundaries, not merely the unrotated mother case. It is limited to these snapshots, not every time-dependent boundary file. Rotation must reverse sign under y reflection; cylinder centers remain on y7.5. Existing geometry and dataset sampling support this continuous physical symmetry, but finite-volume discretization and frozen FNO are not proven exactly equivariant.

## Source pins and reproducibility

| Repository-relative source | SHA256 |
|---|---|
| `artifacts/p064_host_train_only_views_20261006/base/manifest.json` | `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2` |
| `artifacts/p064_host_train_only_views_20261006/train8/manifest.json` | `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35` |
| `artifacts/p064_host_train_only_views_20261006/train16/manifest.json` | `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b` |
| `artifacts/b00_controlled_train_dataset_view_20261006/manifest.json` | `97a82e2a157b87ecf9b9ea626ad079852a8d29ba89628ca5034ef99545929a5f` |
| `artifacts/p064_host_train_only_views_20261006/base/normalization.json` | `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` |
| `docs/B00_CONTROLLED_TRAIN_CONVERSION_APPROVAL_20261006.json` | `4fa7192e13bf7ad3a141bffb483710e2400fd8ee243caa60a6e67ab695927686` |
| controlled b00 `case_mpc/148.1/U` under the source above | `19fc8143d5d3e0398e2c1b931b31f43e83e911eb754fd7084bb97ed318353b47` |
| controlled b00 `case_mpc/148.1/p` under the source above | `aa157f27448d9b544538b1ff9739d6e22d97381b9717912656c34877274d26b9` |

For the44 original cases, the88 selected U/p files were individually hashed. SHA256 of the exact repository-relative-path→file-SHA mapping, serialized with `json.dumps(mapping, sort_keys=True, separators=(',', ':'))`, is `1c072820805aefc8409a47cda4fe1bed96b30212a7fdc65f5ef6edbfd0de5379`. The case selection rule above reconstructs this bounded inventory without scanning whole CFD histories. Small grid/mask reads were not repeated to produce these pins.

## Historical novelty and remaining implementation checks

`scripts/evaluate_tandem_fno.py` sign_flip changes action only during counterfactual evaluation; it does not reflect state/targets. `scripts/probe_fcp020_symmetric_statistics.py` adds mean/RMS statistic VJPs, not mirrored samples. The inspected P064 B training/DataPipe/history path contains no physical reflection augmentation. Existing PPO canonical coordinates are a different intervention.

The proposal is physically admissible as a bounded experiment, not evidence of likely improvement. Implementation must independently roll the frozen flow from mirrored q0/actions (never assume reflected original predictions equal this rollout), retain original `.5 H1 + .5 AR`, apply original/mirror half-losses serially, average over eight original windows once, and clip/step once. The explicit budget is256 originals/512 transformed equivalents/32 updates, with unchanged parent/norm/frozen flow/two biases. CPU module, actual Reader and paired-gradient tests remain separate prerequisites; no implementation or training PASS is asserted here.
