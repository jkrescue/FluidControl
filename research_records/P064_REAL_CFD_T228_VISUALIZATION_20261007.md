# Saved real-CFD field comparison at t=228

Completed CPU-only postprocessing, not a new CFD solve or model evaluation. The three columns are paired zero, successful B policy seed 20261006, and B policy seed 20261007 (whose primary drag criterion failed). E083 is not included.

All paths below are repository-relative. Output: `artifacts/p064_real_cfd_t228_comparison_20261007/`.

- Figure: `real_cfd_t228_comparison.png`, SHA256 `124c012d473752383315eea5255846a8713bd7acb32e1a969db2985eff0b7af3`.
- Figure manifest: `figure_manifest.json`, SHA256 `afaf55e0dfa719f77138e1dc2088334582bdb697b25fcd22146632bd5062c55b`.
- Before/after source inventory: `source_inventory.json`, SHA256 `500023a834106bc92bda259b97e0c309cc12e6bc3c0ce5783b3e170dd50c6b1c`.
- Executed script, now archived unchanged as `scripts/render_p064_real_cfd_t228.py`: SHA256 `b405ffe8653e607e684bfdd5adf59929a546afdd67a4e2c37d47b25529dbdd41`. Canonical promotion occurred after execution; `render_source.py` retains the same bytes in the output.

Sources are `artifacts/p064_b_projected_ppo_long_cfd_20261006/case_zero`, its `case_mpc`, and `artifacts/p064_b_seed20261007_projected_ppo_long_cfd_20261006/case_mpc`. Only saved `228/U`, `228/p`, mesh and system files were copied into exclusive scratch. Both experiments' zero U/p were byte-identical. All inventoried source bytes were rechecked unchanged; no source chmod or writes occurred.

Pinned OpenFOAM image `sha256:24205c9677d39c95221eb903988094dd7a228fc41a2054df3eaa13f80e465fcb` supplied `foamToVTK` only. The reviewed R2 export helper and official Curator callable supplied three 128×256 samples on the same x=[8,25], y=[4,11] grid and exact common mask. Helper hashes, package versions, three NPZ hashes and render arguments are in the manifest. Each row has one shared color range: speed [0.001188167226752761, 1.3560177664605686]; pressure p′ [-0.6257286071777344, 0.641122579574585]. Pressure uses each frame's mean over the same valid ROI removed, not a common absolute pressure reference. Masked cylinder cells are white.

Actual unit `fluid-control-p064-real-cfd-t228-20261007.service`, invocation `7e01e4d03658422f836f91d66f59e0ba`, exited 0 with MainPID 0. Runtime was 10.8343 s; minimum observed physical MemAvailable was 122392158208 bytes. Host cap was 4 GiB/no swap/one CPU, with sequential separate 4 GiB exporter containers (not a combined 4 GiB cap). Startup/runtime physical-available guards were 50/22 GiB. All three exporters exited 0 without OOM and their owned containers were absent afterward. Root-owned export scratch is retained. The 300-second outer bound was not reached.

Three CPU tests check masking, shared ranges and degenerate-range rejection; they do not independently validate CFD physics. Root visually inspected and accepted the rendered figure. Instantaneous morphology does not establish drag reduction, force-window performance, FNO accuracy or net energy benefit. The existing terminal force reports remain authoritative. The PNG and manifest were not changed during archival, and no postprocessing rerun was performed.
