# Isolated K1 GPU inference precision override — preparation only

Hypothesis: reduced-precision GPU backend settings contribute to the observed approximately 0.013 rear-Cl difference from saved CPU inference. This probe cannot by itself attribute every residual difference to a particular kernel or establish equivalence. No equivalence tolerance is introduced.

Start from canonical `scripts/probe_k1_uma_inference.py` SHA256 `56c17443245c60f51b4e4c8f15b1851136a091c73441e5ecd76eebe0695f5e74`. Preserve its official checkpoint loading and historical precision identity validation unchanged. Only AFTER successful `load_bound_k1`, set float32 matmul precision to `highest`, CUDA matmul TF32 false, and cuDNN TF32 false. Persist observed before/effective flags before inference and in each of three saved decisions. This is explicitly a POST-LOAD INFERENCE PRECISION OVERRIDE, NOT execution under the original checkpoint precision protocol. Model tensors must remain unchanged and no gradients/optimizer/CFD are used.

Same K1, exact t148 sample, normalization, baseline, previous zero action, five held-action H2 candidates, and three inference calls. Compare saved outputs with the existing CPU and GPU records after a separately approved execution. Report raw force/cost differences and ranking, not a fabricated equivalence PASS. No GPU replacement in the independent CPU H5 experiment is authorized.

New exclusive output: `artifacts/k1_uma_gpu_no_tf32_probe_20261006`. Required separate approval status: `K1_UMA_GPU_NO_TF32_INFERENCE_APPROVED`, execution_authorized true, exact source SHA and absolute output path; prior probe approval is rejected.

All UMA guard/lifecycle functions are unchanged: MemAvailable >=50 GiB startup and >=22 GiB runtime, CUDA free observational, GPU0 allocator fraction0.06, cgroup memory cap <=12 GiB with swap0, 0.5s subprocess supervision, 180s internal deadline. Root must preserve the original 200s systemd control-group backstop and explicit 12GiB/no-swap cap. This preparation does not launch anything or change any older guard, source, result, or approval.

Tests are synthetic CPU-only (including mock precision settings, rejected old approval, ordering and unchanged guard AST checks); they do not verify actual GPU behavior or availability. CLI for later Lead approval only: the exact official Curator Python environment runs `probe_k1_uma_no_tf32.py --execute --approval <separately-approved-file> --approval-sha256 <actual-file-sha>` inside that bounded unit.
