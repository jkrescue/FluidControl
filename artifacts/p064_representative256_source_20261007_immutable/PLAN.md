# B continuation on a fixed representative 256-point H1 panel

Preparation only; no GPU training is authorized. Existing B real-CFD delivery remains valid independently of this predictor research.

## Hypothesis and history

The fixed40 B diagnostic lowered its H1 training loss96.49% but stopped at300 closures with all four normalized RMSEs above0.01 and a still-decreasing curve. This supports removable local fitting error, not sufficient convergence or generalization. P016 already trained full28 parameters on a fixed mixed H1/AR panel; F already tried pureH1 under Adam32 and harmed AR/H5. Therefore this is explicitly a **combined B-continuation, fixed representative panel and optimizer experiment**, not a first H1 experiment, an isolated LR cause, or a guaranteed AR repair.

## Fixed sampling and weights

Reuse the exact original B256 schedule and its pinned parent order. Keep192 `original44` rows and64 `controlled_b00` rows in their original order. Each of these TWO source labels has its own zero-based occurrence counter k; select lead `1+(37*k % 100)` within its original H100 window. Input is recorded state at window_start+lead−1 with stored current/next omega; target is force at window_start+lead. No future state enters the input, and no current force is an input.

Each schedule row has weight1/256, including repeated physical point identities. Do not deduplicate or reweight silently. Publish exact case/family/time/action coverage, unique physical point count and duplicates, plus per-row source HDF, current/target indices, actual time/actions, normalized input/mask/target and physical truth hashes. This panel is representative only by the fixed source schedule, not a claim to cover all44 original trajectories or the entire training distribution. No NEW b02/b04 controlled acquisition trajectories, development or frozen-test data enter training; the original training families already contain b02/b04 phases and are retained.

Project sampling adapter uses existing PhysicsNeMo HDF5Reader-backed project datasets; no new model or reader API. CPU preparation reads only the selected adjacent frame pairs, not all256 H100 state sequences. It verifies the original three-family H100 index sizes720/408/240 and controlled701 starts before resolving original global indices.

## Model, objective and optimization

Initialize from verified B. Unchanged official six-input/seven-output FNO, last four channels mask-pooled; train same28 aerodynamic tensors, freeze two lifting biases and separate flow model. Original normalized absolute four-force balanced H1 objective weights(.125,.125,.125,.625); no AR training, new auxiliary, decay or clipping. Original norm/action scale unchanged. All gradient microbatches accumulate SSE/256, including the last six points (25×10+6), never an equal average of26 batch means.

Reuse reviewed persistent LBFGS strong_wolfe/history5/lr1/max_iter1. Fixed highest/noTF32 profile after original high/TF32 official load. At most200 outer calls and300 total full-panel gradient closures. Total inner2400s/outer2430s, reserve60s for finalization (fit deadline2340s including setup). Stop early only on accepted-point remeasurement if every channel normalized RMSE≤0.01; that is a training fit target, not admission. Record every trial and every returned-point objective/channel errors/gradient norms/step/counters separately. Budget exception restores last accepted model+optimizer state. No best-checkpoint selection or LR/weight search.

40-point measured255s suggests approximately27–28min by fixed microbatch count for256 points, excluding variable line search and data/save overhead;40min is a hard bound, not a completion guarantee. Approved preparation resource profile24GiB/noSwap,16GiB allocator,CPU4,Available50GiB startup/22GiB runtime/20GiB reserve. Save only the terminal accepted candidate through existing official checkpoint APIs and perform an official fresh reload. Budget-stop may yield a terminal candidate but is never labelled converged; engineering failures do not authorize evaluation.

## Independent evaluation and decision

Original fixed-six H1/continuousAR100 components remain independent evaluation, with both B and candidate recomputed at the SAME highest/noTF32 profile (do not compare tiny deltas against historical high/TF32). Original fixed16 starts b01/b03 × H1–H5 development protocol and complete metrics remain unchanged, with matching B comparator. Consider promotion only if both pooled H1 rearCl and totalCd improve and both fixed-six components do not regress. Report all H5/phase/force/field metrics. No train-fit result alone authorizes PPO/CFD; B default and all physical thresholds remain unchanged.
