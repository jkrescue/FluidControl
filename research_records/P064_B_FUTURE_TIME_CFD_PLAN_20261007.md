# Fixed B prospective future-time physical validation — preparation only

One workflow, no execution authorized by this draft. Same Re100/L/D5 fixed cylinders, no new model/training/geometry. Freeze retained E082 B canonical policy before generating any future observations. This tests a new deterministic time segment, not an independent physical condition or guaranteed statistical independence. Do not select a different start after viewing baseline forces; preserve failures and all six windows.

## Exact source and protocol

Parent: `artifacts/p064_b_symmetry_canonical_ppo_long_cfd_20261007/case_zero`, existing maximal saved time228. Terminal `228` contains U/p/phi/U_0/phi_0/uniform; mesh/system complete. Its parent result SHA is `165b78194f84676b5ea0091b1d160ccf25f913a9f49c74a9424a7d0f03cde7dc`. PENDING binds every file under228/constant/system, the parent result, all original source/runtime/training inputs. Metadata inspection confirmed248 absent in this parent; this is not a global OS read-history assertion.

Policy `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/ppo_final.zip`: `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`; Vec `1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145`; canonical adapter `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`. Same one policy call/sign restore/single physical filter, |omega|<=.75, delta<=.1 per .1D/U, CFD dt=.005. Official SB3 CPU inference; no online FNO/MPC.

1. Copy only parent228/constant/system into exclusive `baseline_zero`; do not modify historical source. Run200 zero-action .1 intervals, all original20-step/Courant/continuity guards, exactly228→248. Save baseline progress/health and all resulting probe/force files.
2. Read real exact t248 wakeProbes (32u/v), front/rear Cd/Cl, plus prescribed zero omega into69 channels. Missing/nonfinite/conflicting records abort, never zero-fill or copy old148 observations. Bind source-file SHA and observation in baseline_result. Copy generated248/constant/system into identical paired branches.
3. Solve new B/zero pair248→328 for800 intervals. Six original windows translated: (248,254.2],(254.2,260.4],(248,260.4],primary(268,328],companion[268,328],full(248,328]. Primary original drag>=2%, fluctuation RMS ratio<=1.05, bias<=.10; full six reported, no threshold change. New zero is actually solved, not reused historical results.

The clone only changes identity/time and adds fixed zero pre-roll/copy. Tests compare original numerical helper AST and paired-loop text (only148→248). Original exact-time observation reader/PairSolvers/force metrics remain pinned and unchanged. Total pre-roll200+paired1600 solver segments. Baseline and paired branches retain real saved fields; generated baseline unchanged after copying. Terminal source-tree hashes rechecked. Future248 restart hashes are generated and recorded, not invented in the prospective spec.

## Resource / decision boundary

Expected about25min from previous paired800 wall1109s plus200 single-zero intervals; not a guarantee. Global worker2280s / outer2400s / stop120s;8GiB/noSwap controller CPU400%, sequential baseline one8GiB CPU2 solver then paired two8GiB CPU2 solvers. Available>=50GiB startup, runtime22GiB,20GiB reserve; disk>=20GiB before and during. Estimated newstorage<=6GiB, parent paired output4.1GiB; measured disk available215GiB at preparation. No deletion or source chmod. Any failure aborts and retains owned evidence; no automatic retry/policy selection.

Pending plannedunit `fluid-control-p064-b-future-time-cfd-20261007.service`; output `artifacts/p064_b_future_time_248_328_cfd_20261007`. Freeze same reviewed driver bytes and rebind paths only after independent review; Lead can authorize this whole fixed pipeline once. Future baseline does not need a second outcome-dependent approval.

Launch template (NOT executed; substitute actual frozen path and approved JSON SHA only after approval):

```sh
systemd-run --user --unit=fluid-control-p064-b-future-time-cfd-20261007.service --property=Type=exec --property=RemainAfterExit=yes --property=MemoryMax=8G --property=MemorySwapMax=0 --property=CPUQuota=400% --property=RuntimeMaxSec=2400 --property=TimeoutStopSec=120 --property=KillMode=control-group --property=OOMPolicy=stop --working-directory=/workspace/fluid_control --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=4 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONNOUSERSITE=1 --setenv=PYTHONPATH=/workspace/fluid_control/.runtime/exploratory-h5-ppo-py312 /workspace/fluid_control/.venv-curator-py312/bin/python FROZEN_DRIVER --spec FINAL_APPROVAL --spec-sha256 FINAL_SHA --execute
```
