# One real-frame CPU comparison approval

Approved by Lead on 2026-10-06, after independent source/API review. This is an engineering comparison, not scientific control admission.

Execute `/tmp/current-frame-adapter-review.cgILn5/verify_online_current_frame_cpu.py`, SHA256 `4386e8d9920ffe24cce1c43d2bff089ceecdabae4704b9a188519927c67599ff`, once on Spark using existing `.venv-curator-py312/bin/python`. Retain its exact bound VTU, HDF frame 0, normalization, adapter and official source pins. Output exclusively `artifacts/online_current_frame_cpu_20261006`.

Unit `fluid-control-current-frame-cpu-20261006`: CPU only, CUDA hidden, two CPU quota, 4 GiB MemoryMax, zero extra swap, 180-second RuntimeMaxSec, KillMode=control-group. Require host MemFree >=25 GiB and MemAvailable >=30 GiB immediately before launch. No concurrent new GPU job during this probe. Lead inspects unit and physical free memory during execution; stop this unit if either available/free memory falls below 20 GiB. Preserve failed output; no automatic retry or tolerance changes.

The immutable existing Curator sampler processes one byte-identical VTU copy; official HDF5Reader reads the corresponding existing frame. Report exact equality plus max-absolute and RMS differences in physical/normalized fields and packed input. No model, optimizer, CFD solver, new data generation, installation, policy or action execution. Differences require investigation; completion does not mean prediction or closed-loop success.
