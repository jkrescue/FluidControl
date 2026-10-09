# P026 matched history trainer — independent CPU review

Reviewed final staged trainer SHA256 `562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e` and tests SHA256 `fc32801e1c873285d4b2ef81df4e58664683744f74924592e34da88f84262332` against the full canonical `docs/FC_P026_HISTORY_COMPARISON_PLAN_20261005.md`. The final delta adds reporting-only warm/padded summaries; the original full review covered predecessor trainer `5fc4f6d35c3c65dede542df67a6dd1bd1fc44ab150a65859bd68887c823623b3`.

Verdict: no concrete blocking defect found in the reviewed training logic. This is CPU software review, not authorization or evidence of completed training. No GPU, model training, data modifications or canonical source edits were performed.

## Verified contracts

- Separate flow and aerodynamic parent arguments enforce distinct directories, exact checkpoint basename sets, confinement, pinned artifact bytes, official load epochs/metadata and tensor hashes. Flow is the P009 epoch-0 pair; aerodynamic initialization is the P018 epoch-1 pair. Official FNO/checkpoint implementations and immutable helper sources are pinned.
- K1/K4 differ in declared history representation and the minimal 288 additional lifting elements. Both expose the same 28 trainable names and two frozen biases. Warmstart preserves inherited columns and remaps official coordinates.
- Original dataset samples and target/action normalization are retained. Prefix reads use the original family/case/start metadata and only preceding frames, with trajectory-frame-0 padding. H1 histories end at the current true state; AR histories use the separately frozen flow. Aerodynamic field outputs are discarded.
- Original P013 force objective is unchanged. The reviewed shared objective uses ten mixed-20 chunks, with K1 exact fixture equivalence and K4 chunk/monolithic gradient coverage from prior tests.
- Both protocols specify fresh AdamW at 1.5625e-7, betas 0.9/0.999, epsilon 1e-8 and weight decay 1e-4. The pinned P015 helper averages eight raw gradients before one norm-1 clip/update. The trainer checks all 28 optimizer state step counts and finiteness at each update.
- Actual dataset inventory and sampler hash are checked before updates; completion requires all 1,368 consumed identities equal the original sampled order and exactly 171 updates. Family counts and K4 warm/padded partition are fixed at 720/408/240 and 1,300/68.
- New-history/inherited gradient norms are recorded after the eight-window average and before clipping, along with applied clip scale, update norm and cumulative displacement. CPU snapshots/reductions avoid an additional full-size GPU reduction buffer but still consume host memory.
- Diagnostic panels occur only at 0/456/912/1368 consumed windows. Model/gradient/optimizer/mode invariants are checked; RNG is restored. K4 terminal ablation zeros only added history columns, evaluates the fixed panel and restores the exact terminal tensor hash. It is explanatory, not checkpoint selection.
- Only terminal official checkpoints are saved. Flow bytes remain copied from the fixed parent. Fresh official aerodynamic and flow reloads check epochs and exact tensor hashes; aerodynamic metadata must also match exactly. New P026 kinds/history schemas and separate role architectures avoid relabeling K4 as a six-channel model.

The actual P018 manifest was inspected for producer-schema compatibility. Retained count/rate/config fields remain applicable; P026 status, experiment, protocol, semantics and aerodynamic initial-parent hashes are explicitly replaced. Consumers must use the explicit role architectures rather than assuming the inherited six-channel base architecture describes K4.

## CPU evidence

Independent command, from `/tmp/p026-review.ifqR2N`:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider test_train_fcp026_history.py test_probe_fcp026_history_resource.py test_p026_history_objective.py test_p026_state_history.py
```

Final result: **41 passed in 1.16 seconds**. Coverage includes synthetic inventory, prefix padding/immutability, parameter partitions, fresh optimizer/step/finite checks, pinned accumulation equivalence, parent path/hash rejection, K1 objective equivalence and K4 causal history gradients. The added reporting test verifies the same K4-availability partition in both arms: three warm windows (two nonzero) and three padded windows (three nonzero). Original objectives average all windows within a subgroup; force statistics exclude fixed zero-action window 160. The existing overall aggregate remains the unchanged P020 callback. Empty subgroup summaries use null rather than invented zero values.

## Boundaries before execution

These tests do not execute the full trainer or prove production DataLoader/resource stability. The canonical plan still requires actual preapproval inventory/order evidence and formal H1/AR/reset history-caller tests; their completion must be checked separately. Immutable launcher approval must bind the complete source/dependency closure, both parent pairs, data bytes and resource guards. The one-window resource result does not measure full optimizer/cache peak memory or provide an all-window runtime guarantee.

Warm/padded metadata and fixed-panel subgroup summaries are retained. These six-window summaries do not replace full formal warm/cold reporting or establish reset safety. No candidate acceptance or PPO is authorized by this review.
