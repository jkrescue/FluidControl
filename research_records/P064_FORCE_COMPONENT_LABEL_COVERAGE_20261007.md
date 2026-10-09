# P064 train-only force-component label coverage

This is a read-only source-coverage result, not a conversion or model result.

- Coverage receipt: `/tmp/p064-force-component-sidecar-review/coverage.json`, SHA-256 `dbd0f15fdad0cc06fe5d85f1a6ff1b02125aa28da711c881480ec32f01c2a9d2`.
- Complete: 45 trajectories, 20,493 frames, zero missing aligned component dictionaries.
- Frames by family: base20 16,020; train8 1,608; train16 2,064; controlled-b00 801.
- Maximum `pressure+viscous-raw_total` absolute discrepancy: `1.3322676295501878e-15`.
- Maximum `hdf_total-raw_total` absolute discrepancy: base20 `1.2380015916746423e-5`; train8 `1.243862659752043e-5`; train16 `9.952217716602263e-8`; controlled-b00 `6.005001229603124e-8`.

All-train physical pressure coefficient means are `[1.0418905751, 0.0004712929, 0.6358439451, 0.0083426454]`, with population standard deviations `[0.0090838590, 0.2766343213, 0.1777127019, 1.0816883745]`. All-train viscous means are `[0.3492626271, 0.0001494950, 0.2554092739, 0.0008966455]`, with population standard deviations `[0.0011164046, 0.0426447897, 0.0175931905, 0.2040648170]`. The receipt contains the corresponding base20 and per-family statistics.

These are physical OpenFOAM coefficient statistics, not replacements for the established total-force normalization. The observed HDF/raw discrepancy is retained and reported; the read-only audit does not assign it to a single cause.
