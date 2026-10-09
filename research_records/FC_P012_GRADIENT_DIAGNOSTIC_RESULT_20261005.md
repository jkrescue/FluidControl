# FC-P012 train-only gradient diagnostic result

This record documents a bounded diagnostic, not a candidate, admission result,
training run, or PPO authorization.

## Execution identity

- Reviewed implementation commit: `67cdc65645e5742bbf469d22ebab558a135c537a`
- Diagnostic SHA-256: `978d28da0fd42d087a22c3a5620ee15c32c1edc63bf65ef07e5602803ff924c0`
- Test SHA-256: `1f571caaf5838aeffc4c2fbd535997ad360eb7885404de7b2740707e163cdb81`
- Immutable launcher SHA-256: `12007551ca34529c3ce1b8d1d123bb3a2873bfa424679e624b89f5f0c6789a91`
- Execution approval SHA-256: `4bbf7165f0ef23af3b50c119da0d8317f512ec033ebbef8f34adc70e06516dbc`
- Runtime image ID: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`

The first unit, `fluid-control-fcp012-gradient-diagnostic-20261005.service`,
failed before entering the launcher because a nested `awk` expression was
misquoted in `systemd-run`. It started no container and performed no GPU
calculation. Its journal was retained. The operational-only recovery unit,
`fluid-control-fcp012-gradient-diagnostic-r2-20261005.service`, directly invoked
the already hash-verified immutable launcher. It completed successfully with
12/12 fixed train windows. The guard recorded exit code zero and a minimum
physical `MemAvailable` of 98.309 GiB.

## Authoritative evidence

Root: `artifacts/fcp012_decoder_gradient_diagnostic_20261005/`

- `diagnostic/result.json`: `4142cdc5c67ff04d50f2921887e01034a9ea7d09a2865ac286b52d1c911d814d`
- `completion_receipt.json`: `86f9d93178e32a9e2769444bb327f9eaf494b290c77590efff1b1726d204b26c`
- `launch_receipt.json`: `bdc41a8b8114c809bc092534e1b00ebeaeeced6caa9d20b5b08ae2119c53cbae`
- `run.log`: `94da09dcdd6580e7f42b4d7103b175882e12b5f87e31b6e8a68cfe102a627e2d`

No optimizer, clipping operation, parameter update, checkpoint save,
validation/frozen access, or PPO execution occurred. Model tensor hashes were
identical before and after the diagnostic.

## Predeclared diagnostic rules

Across the five nonzero-action-history train windows, the P009 parent had
field-to-weighted-force complete-vector norm ratios of 2.77, 2.15, 3.16, 1.71,
and 5.79. None exceeded 10. Its five cosine values were 0.086, 0.346, 0.006,
-0.152, and -0.086; none was below -0.2.

The FC-P011 decoder-tail terminal had ratios of 3.19, 1.83, 2.60, 1.92, and
2.48. None exceeded 10. Its cosine values were 0.179, 0.474, 0.185, 0.137,
and 0.156; none was below -0.2.

Thus neither predeclared signal was supported for either model: the scale rule
was 0/5 (required at least 4/5), and the conflict rule was 0/5 (required at
least 3/5). This does not support blaming clipping or immediately scanning loss
weights. Under the approved plan, the next explanation to examine is state
representation/data observability.

The final rear-Cl row's field-loss gradient was exactly zero, so its
field-versus-force cosine is correctly `null`. Direct-total versus summed
component relative residuals ranged from approximately `3.09e-5` to `9.84e-5`.
They remain observational under default TF32/high arithmetic and have no
predeclared equivalence threshold or admission meaning.
