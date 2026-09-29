# CFD data generation and provenance

Status: **developmental numerical data, not a validated reproduction** (2026-09-27).

## 1. Physical scenario and source distinction

The target is a uniform incompressible incoming flow over one upstream circular cylinder and two smaller downstream rotating control cylinders. The developmental solver uses a two-dimensional nondimensional approximation inspired by the Yeung–Tang arrangement. It is **not** the published experiment or its exact geometry: the gap is `0.125D`, not `0.05D`; the modeled Reynolds number is 100, not 680; the flow is 2-D rather than a water-tunnel flow.

An independently published real-data benchmark is also being ingested: Rodríguez-Asensio et al., [paper](https://arxiv.org/abs/2602.17713), [Zenodo data](https://doi.org/10.5281/zenodo.20794709). It has three **equal-diameter** cylinders, a stationary front cylinder and counter-rotating rear pair, at Re=9100. Experimental PIV and force data are measured; companion URANS vorticity and force data are numerical. Neither is produced by this code. The PolyU [Yeung thesis](https://theses.lib.polyu.edu.hk/handle/200/14670) provides the research target but no public machine-readable flow-field archive was identified on its repository page.

### Published URANS provenance (third-party data, not regenerated here)

The paper describes a 2-D ANSYS Fluent URANS calculation matching the experimental cylinder arrangement and tunnel walls. Its reported configuration is: three `D=30 mm` cylinders with `1.5D` center spacing; water at `U∞=0.31 m/s`, `ν=1 mm²/s`, `Re=9100`; about 180,000 mesh elements, cylinder surface spacing `0.01D`, more than 300 cells per circumference, 28 inflation layers, `y+≈1`, and minimum orthogonality 0.58. It uses a k–ω SST model with curvature correction, 2% inlet turbulence intensity and turbulent viscosity ratio 4. Rear-cylinder rotation uses local rotating mesh zones and no-slip walls. The time step is set to one degree of cylinder rotation, so it varies by actuation; the simulated `p` range is approximately `−3` to `2.5` in `0.5` increments. The paper explicitly warns that 2-D URANS does not resolve 3-D effects and has limitations for separated flows. These details are recorded to interpret the downloaded numerical files; no Fluent case files or solver configuration are provided by the Zenodo record, and this project does not claim bit-for-bit CFD regeneration.

The downloaded URANS `AerodynamicForces.mat` is numerical (10 `p,Cd` values) and readable. The 60 `vorticity_*.hsf` files start with the `;; HSF V21.00` signature: these are [HOOPS Stream Format](https://docs.techsoft3d.com/3df/3021/general/hsf/HSF_architecture.html) visualization files, not HDF5 or simple scalar arrays. They must **not** be treated as machine-readable vorticity grids without a verified HOOPS/Fluent export path. The experimental `.h5` files, once integrity-checked, are the intended field-training source.

The experimental archive has 57 `.h5` files but only 28 nominal actuation values (`−2.8` to `2.6` in steps of `0.2`). Its `_zero` files are **unactuated reference acquisitions** immediately preceding corresponding controlled experiments. Critically, their HDF5 `p` attribute is the associated controlled case value, not the physical rotation during the reference acquisition. `scripts/prepare_piv.py` therefore explicitly excludes `_zero` files from labeled control training. The extra `p=-1.4` upward-branch file remains identifiable in the manifest; the first-stage single-branch model selects one file per `p` and documents this selection.

The experimental `AerodynamicForces.mat` has 28 numerical `p,CD` pairs but its `Uncertainty` array contains **NaN for every entry** in the released archive. The repository description promises uncertainties, yet there are no numerical per-case uncertainty values to use. The preparation manifest records them as `null`; model metrics must not invent error bars. The paper discusses approximate aggregate force uncertainty, which is not a substitute for missing casewise values.

## 2. Developmental Fourier/Brinkman CFD configuration

Source: `src/fluid_control/cfd.py`; launch: `scripts/generate_dataset.py`. Solver is an in-house, inspectable pseudo-spectral method, **not** SU2, OpenFOAM, DNS, or the paper's solver.

| Item | Current value / implementation |
| --- | --- |
| Equations | 2-D incompressible Navier–Stokes, nondimensional `U∞=D=1`; kinematic viscosity `1/Re` |
| Domain/grid | `[0,12D) × [-3D,3D)`; 256 × 128 uniform Cartesian points; `Δx=Δy=0.046875D` |
| Bodies | Main disk radius `0.5D` centered at `(3D,0)`; control disks radius `0.125D` at `(3D+0.75/√2 D, ±0.75/√2 D)`; surface gap `0.125D` |
| Actuation | Two independent signed surface-speed ratios `q₁,q₂`; local angular speed `ωᵢ=qᵢ/rᵢ`; nine fixed anchors plus seeded random values in `[-2,2]²` |
| Boundary model | Periodic FFT in both directions; streamwise sponge near `x<0.8D` and `x>10.5D` relaxes to `(1,0)`; **not** a true inlet/outlet |
| Solid model | Smooth disk masks and Brinkman velocity penalization, `η=0.02`; stationary main disk, prescribed rigid-body tangential velocities on controls |
| Time/numerics | Explicit nonlinear advection, implicit Fourier viscosity, projection onto divergence-free velocity, 2/3 spectral filter; `Δt=0.005D/U∞` |
| Existing run | 3600 steps (`18D/U∞`), mean from step 2000 (`8D/U∞` averaging); 48 action cases, seed 20260927 |
| GPU | PyTorch FFT, CUDA device selected by `CUDA_VISIBLE_DEVICES=0` |

The algorithm computes the convective term on the physical grid, applies body/sponge forcing, Fourier-projects velocity to zero divergence, and applies viscous damping and a 2/3 filter each time step. Its periodic/sponge boundary and coarse body representation can alter wake dynamics. No pressure or cylinder surface-force coefficient is currently exported, so the objective below is a **wake proxy, not drag**.

## 3. Generated data fields and current files

`artifacts/cfd_dataset.npz` holds 48 cases; `artifacts/cfd_dataset.json` holds the serialized configuration, seed, count and output grid. Pilot files `smoke.*` and `pilot_9.*` are also development diagnostics. The `.npz` arrays are:

The JSON metadata explicitly sets `solver=in_house_fourier_brinkman_2d` and `status=developmental_unvalidated`. The legacy CFD-surrogate training/evaluation scripts refuse this dataset unless `--allow-development-data` is passed, so it cannot silently become a formal result.

- `actions`: shape `[N,2]`, nondimensional signed control surface speeds.
- `fields`: shape `[N,3,64,128]`, area-downsampled time means of streamwise velocity `u`, transverse velocity `v`, and `(u−1)²+v²`.
- `features`: shape `[N,6,64,128]`, the three solid masks, prescribed solid `u,v`, and the normalized Reynolds-number plane. These are model inputs, not measured flow fields.
- `wake_error`: mean of `(u−1)²+v²` over `x∈[4.5D,8D], |y|≤1.5D` and averaging time.
- `control_cost`: `0.01(q₁²+q₂²)`; `objective = wake_error + control_cost`.

The reported baseline `objective=0.751853` and smallest value among sampled actions `0.667853` are **internal developmental numerical values**. They are not drag reductions, experimental results, or verified improvements.

The completed `artifacts/cfd_sensitivity.json` compares the baseline grid against a 384×192 grid and a half time step over the same `18D/U∞` horizon for `(q₁,q₂)=(0,0),(1,-1)`. The wake proxy changes by approximately `9.9%` and `8.7%` under grid refinement, versus approximately `0.04%` and `0.20%` under time-step halving. Thus grid independence is **not** established. The short averaging horizon is another unresolved issue.

## 4. How to regenerate the developmental set

From the project root, in its isolated environment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/generate_dataset.py \
  --out artifacts/cfd_dataset.npz --count 48 --batch 8 \
  --nx 256 --ny 128 --dt 0.005 --steps 3600 --average-start 2000
```

Never overwrite `data/raw/`. Record git commit, installed package versions, GPU, command, and SHA-256 of output for an externally reviewed run.

## 5. Release gate for newly generated CFD training data

Before any self-generated flow is admitted to the **formal** PhysicsNeMo training set, require all of:

1. Check finite fields, velocity-divergence norm, body-mask leakage and CFL history; archive failures.
2. Demonstrate late-time stationarity and at least several shedding cycles, using probe/force time series and a stated averaging window. The existing `18D/U∞` total time is too short to assume this.
3. Run at least three mesh resolutions and two time steps at the same physical time and action settings; publish relative changes in the primary observables.
4. For an unactuated isolated cylinder at a documented Re, compare Strouhal number and mean drag with a published benchmark. For the three-body case, compare baseline/actuated wake quantities against a published case at matching geometry and Re.
5. Separate solver verification from surrogate evaluation; hold out entire actuation values (and preferably Re values), and report uncertainty/error bars.

Until then, the Zenodo PIV is the defensible primary data source. An external solver such as OpenFOAM can be added as a separate case with its own version-pinned container, committed mesh/BC/numerics, raw run logs, and exported fields. The existing pseudo-spectral output must not be silently relabeled as OpenFOAM or SU2 output.

An open-source starting case was screened: [darshan315/fluidic_pinball](https://github.com/darshan315/fluidic_pinball), commit `211b7759c1afa8bd01e79b7ab921b8f8aeb3b40b`, MIT licensed, originally configured for OpenFOAM v2106 and `pimpleFoam`. Its baseline `blockMeshDict` groups all three cylinder surfaces into one `cylinders` patch with `noSlip`; `transportProperties` sets `ν=0.005` with `U∞=D=1` (Re≈200), and `controlDict` runs to time 2000 with `Δt≤0.01`. It is **not yet a controlled-rotation dataset**: independent wall rotations require three separate patches and correctly signed `rotatingWallVelocity` entries, followed by mesh, Courant, force and convergence checks. No OpenFOAM case from that repository has been run or reported here. The official [OpenFOAM rotating-wall boundary condition](https://doc.openfoam.com/2312/tools/processing/boundary-conditions/rtm/derived/wall/rotatingWallVelocity/) and [OpenCFD v2106 container](https://hub.docker.com/r/opencfd/openfoam-run/tags?name=2106) are candidate components for that separately validated route.
