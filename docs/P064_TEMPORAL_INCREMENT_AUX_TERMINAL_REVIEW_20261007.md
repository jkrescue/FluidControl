# Temporal-increment auxiliary: independent terminal review

Engineering PASS; original fixed-six H1/AR retention **FAIL**. B remains the reference/default. The very small AR deterioration is not described as statistically significant, and no threshold or tolerance was changed to admit this candidate. No fixed-development evaluation, PPO or CFD was launched by this review.

Training unit `fluid-control-p064-temporal-increment-aux-20261007.service`, invocation `0d2508de79b646f08c87d7c0f0c1d53c`: independently observed PID0/exited/Result success/ExecMainStatus0. Approval SHA `98f5638ae454e532b9ddb22b90937c95468214400b0cfe31b3031f7e3dc565f2`.

- Result: `artifacts/p064_temporal_increment_aux_20261007/result.json`, SHA `947871cacbe2a443f6d237426c3d855f9955b4659121797d915afa497cae48e7`.
- Manifest in that directory: SHA `ef68459da39bcbbaca1449d63bd52a3d43506ee682cd08a956e1232cf965e285`.
- Independent receipt: `artifacts/p064_temporal_increment_terminal_audit_r2_20261007/receipt.json`, SHA `fc6a97135ea254a1771d88cc14369e5a0557b7b38faf00eccf70e998de643e68`.

## Original fixed-six retention

Recomputed original balanced loss from all four saved channel MSEs (`.125*sum + .5*rearCl`) for each of six exact matched identities/history/frozen-flow hashes, then averaged. Auxiliary loss is excluded. The fixed B result is pinned by SHA `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`.

| Original metric | B, recomputed | Candidate, recomputed | Relative change |
| --- | ---: | ---: | ---: |
| H1 balanced | .003976855239898214 | .00397630953902232 | −.0137219195% |
| AR100 balanced | .008946200532894485 | .008947911815994303 | +.0191286021% |

H1 improves slightly; AR worsens slightly. The predeclared requirement that both are nondegrading therefore fails. Producer aggregate values differ in the last few digits due to float32 accumulation; saved channel arithmetic agrees within the established bookkeeping tolerance, while the **decision comparison itself uses no tolerance**. No model forward independently recomputed these losses.

## Engineering and objective checks

Reused SHA-pinned original B terminal checker. Verified all 434 base source hashes and three overlays, original B schedule/order and train-only identities, 256 windows/32 updates, journal ordering, checkpoint epoch1, 28 finite Adam states all step32 with original hyperparameters, frozen flow archive/state bytes, and both frozen lift biases by CPU tensor inspection. The fresh optimizer claim is tied to the exact reviewed runner (`19e0799b…`) constructing AdamW and checking initial step0, plus terminal state inspection; it is not inferred solely from final step32. Official fresh reload is the bound producer's verification, not a second model reload by this audit.

All 256 saved auxiliary/original/combined objectives and their journal counterparts are finite and satisfy `training_objective = original_total + lambda1*auxiliary`; all 32 window-group means and saved preclip/applied-clip records agree. Original total remains `.5 H1 + .5 AR`. Exact auxiliary protocol is 99 adjacent edges, 90 within chunks plus nine recomputed boundaries with both endpoint gradients, unchanged training normalization and no dt division. The complete protocol passes the already-reviewed isolated consumer validator. This checks saved arithmetic/source wiring, not independent training-gradient reconstruction or new causal-action labels.

Training actual memory cap12 GiB/no swap; recorded peak `5007224832` bytes; sampled minimum MemAvailable `106.13467025756836 GiB`, above22 GiB. No data payload rehash or new HDF sampling was performed.

## Independent execution record

R2 audit unit `fluid-control-p064-temporal-increment-terminal-audit-r2-20261007.service`, invocation `caf1dfdd61ca421f9a1a9b749da1ea83`, exited0 under CPU1/2 GiB/no swap/120 s with CUDA hidden and absolute PYTHONPATH. Executed `/tmp/check_temporal_increment_terminal_r2.py`, SHA `8e07191ee4e777c9f6b17ba6a38781697c663afaa85ed508710087d14954dd31`; twelve CPU fixtures passed, followed by independent narrow review.

R1 invocation `40325a70cd0a43f2b5c0c322ebb217f5` and source SHA `82c5a55194142615c8f55a3472c707a1de253c0379c7c04bac2925eedcd31983` remain preserved: the reviewer incorrectly required the approval's abbreviated protocol dictionary to equal the producer's complete protocol. R2 applies explicit two-field b00 aliases, exact declared-field checks and the unchanged full consumer protocol validator. No scientific calculation, training or criterion was changed or repeated.
