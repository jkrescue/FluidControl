# P064-B same-six cached H1 versus free-AR diagnostic

## Actual execution and provenance

Lead-authorized JSON-only analysis completed 2026-10-06 13:40:15 UTC, unit `fluid-control-p064-same6-cached-comparison-20261006.service`, invocation `6c79f976d94f4e4fa8c32aa4a1688b66`, PID0/exited/exit0. Limits: 2 GiB RAM, zero swap, one CPU, 120 seconds, CUDA hidden. `MemoryPeak` is not set and is not reported as zero. No model, HDF, GPU inference, training or CFD was run.

Executed script SHA `159831b72ed2c941b293dfe1c6040ebdaf31788745b204c53cd999d71dea6c39`; author and independent reviewer each passed 12 CPU fixture tests. Exclusive output `artifacts/p064_b_same6_h1_ar_cached_comparison_20261006.json`, SHA `7158d4e7f977a4fa126c9293d85cad350ce1799b0685d691f02225834e550460`.

Before execution, independently checked formal R3 invocation `3bded2dcb4a24f808879987962a9ef8b` was PID0/exited/exit0 and receipt SHA `30d3d0746580b8423a9f626a0ebe1b76c129acab7800158f74d7d8df3d8a1799`. Its explicit provenance distinguishes R2 reused precision/validation10 from R3 executed stages. Under `artifacts/fcp064_arm_b_formal_resume_r3_20261006`, the three actual inputs were rehashed:

- `dynamic6/evaluation.json`: `4dca64dd2a599ec009af44702af5698e0213acec77c3a0a464c822c61c7683db`.
- `dynamic6/segments.json`: `612a8af07d724964605abd0d0b0e9fe96e45f6cb65247e19a01c20b0aa5796fb`.
- `force_window/result.json`: `6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173`.

Fixed B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`, aero model `57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e`, normalization `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`, and dynamic-six manifest `bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae` match saved identities. Source hashes independently match formal receipt: evaluator `daef4a3a6eb1b9190f5cde728581b0c1b4b9656f56e865cf61a53deb1f37de88`, force-window producer `eee1f59fa269a22556048f3fde8fc7ecbc31bb913571e49075dead92618a8576`. Saved precision log SHA `a7d1eb5ae0822f0486d68cf16be09b0267418dd3817c900a26e3c408e3584373` records high / CUDA TF32 true / cuDNN TF32 true, not the later highest/no-TF32 exploratory profile. These execution-provenance checks are external to the three-JSON CLI; the CLI alone does not enforce formal terminal status.

## Alignment and result

Exactly six cases b01/b05 × minus/zero/plus, H1 starts0–99 paired with AR targets1–100: 600 endpoints, no duplicated/missing endpoint. AR arrays contain101 samples including the initial truth seed; seed0 is excluded. Times, action absolute summaries/rates, target total Cd and four-channel ordering passed checks. Total Cd is formed by signed front+rear sum before absolute error. Final62 refers to targets39–100, 372 pooled endpoints. Full endpoint, phase, action, case and lead summaries are retained in the JSON without selecting favorable cases.

| Scope / MAE | True-state H1 | Free AR |
|---|---:|---:|
| All600 rearCl | .0452479271203 | .0623860029175 |
| All600 totalCd | .0220967183510 | .0207225337625 |
| Final372 rearCl | .0483166644530 | .0736038178346 |
| Final372 totalCd | .0240335528569 | .0247963843166 |

| Case | H1 rearCl | AR rearCl | H1 totalCd | AR totalCd |
|---|---:|---:|---:|---:|
| b01 minus | .03842405 | .05394896 | .02290576 | .02369295 |
| b01 zero | .03633422 | .03489382 | .00563808 | .00433907 |
| b01 plus | .07131069 | .11313867 | .03558076 | .03593917 |
| b05 minus | .04878967 | .08521179 | .03763872 | .02068606 |
| b05 zero | .03564171 | .02912823 | .00895520 | .00980230 |
| b05 plus | .04098723 | .05799455 | .02186179 | .02987564 |

RearCl AR absolute error exceeds H1 at360/600 endpoints, falls below it at240; all four rotating case means worsen, both zero case means improve. TotalCd is mixed, with notably smaller AR error in b05-minus. Therefore the cached evidence does not support attributing every force error to predicted-state drift or claiming universal benefit from true-state input.

## Numerical and scientific limitations

The first target shares the true initial state, yet its rearCl absolute errors already differ: maximum absolute discrepancy `.0005698204040527344` at b05-plus. The formal H1 evaluator uses batch8 while the force-window trajectory uses batch1 under high/TF32. Batch-dependent numerical behavior is a plausible contributor, not a demonstrated exclusive cause. Do not claim numerically identical forwards or a causal additive decomposition of AR error into readout and state error.

H1 cached action summaries are absolute-valued and do not independently prove signed-action identity; that identity rests on the verified same-data producer protocol. H1 records retain absolute channel errors, not signed rearCl predictions. Hence this diagnostic cannot reconstruct H1 signed mean rearCl or centered RMS, and cannot directly explain the failed force-statistics gate. It is descriptive reuse of already opened development data, not new independent validation, a new acceptance threshold, or model admission. Original full-formal failure and separately measured physical control benefits remain distinct.
