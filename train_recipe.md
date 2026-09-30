# 串列双圆柱第一阶段训练操作手册

更新时间：2026-09-30
远程项目：`~/workspace/fluid_control`

本文给出从 OpenFOAM 原始结果、PhysicsNeMo Curator、PhysicsNeMo Datapipe 到 FNO 训练和测试的完整可执行流程。每一步先检查预期产物，再进入下一步；脚本默认拒绝覆盖已有结果。

## 0. 长期系统目标与当前阶段位置

长期目标是建立面向不同来流速度的实时在线闭环控制系统：根据流动观测在线选择后圆柱转速，以降低阻力、抑制升力波动或尾流涡脱落，并在加入结构动力学后研究减振。当前 FNO 是动作条件动力学代理，只负责预测“给定状态和转速后，下一时刻会发生什么”，不负责选择动作。

当前完整链路在闭环系统中的位置为：

```text
传感器/流场 -> 状态估计 -> 控制器 -> omega
                              |
                              v
                    PhysicsNeMo FNO 代理
                              |
                              v
                     下一状态与 Cd/Cl
```

当前数据只覆盖 `U∞=1、Re=100、L/D=5`，且模型使用完整流场输入。跨来流闭环控制需要新增多 `U∞/Re` CFD 数据并把来流作为条件；真实在线应用还需要稀疏传感器状态估计。闭环策略必须回到独立 OpenFOAM 中回放验证。适用范围和扩展条件见 `cfd/tandem_cylinders/CASE_SPEC.md` 第 10 节。本阶段继续完成代理模型训练和独立 test rollout，不在当前步骤实现控制器。

现有动作域内的多步训练、独立测试和论文物理基线对照已经完成。高转速 CFD 数值门槛已验证到 `q=±2.5`；当前下一步是生成扩展动态数据并重训代理。扩展代理通过独立测试后，再进入论文闭环控制复现。已有控制草案保存在 `closed_loop_control_spec.md`。

## 1. 固定的软件和数据边界

### 1.1 软件

| 环节 | 工具 | 当前固定版本或来源 |
| --- | --- | --- |
| CFD | OpenCFD OpenFOAM v2512 | `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319` |
| 数据整理 | PhysicsNeMo Curator | NVIDIA 官方仓库提交 `86533e581b3550326d89e97cb4d4126e7061b416`；当前为 beta API；单独使用 Python 3.12 |
| Datapipe 与模型 | NVIDIA PhysicsNeMo | `HDF5Reader`、`DatasetBase`、`TensorDict`、PhysicsNeMo `DataLoader` 和 FNO；远程为 2.2.2 |
| 深度学习 | PyTorch/CUDA | 远程 `.venv` 当前为 `torch 2.14.0+cu130` |
| 模型 | PhysicsNeMo FNO | `physicsnemo.models.fno.FNO` |
| 配置 | Hydra / OmegaConf | `conf/tandem_fno.yaml`，运行时保存 resolved config |
| 设备与分布式 | PhysicsNeMo DistributedManager | 单 GPU 和 `torchrun` 多 GPU 使用同一入口 |
| 训练封装 | PhysicsNeMo StaticCapture | FP32 梯度更新和梯度裁剪；当前 FNO metadata 不支持 GPU AMP |
| 指标日志 | PhysicsNeMo LaunchLogger / PythonLogger | minibatch、epoch、validation 指标及文件日志 |
| Checkpoint | PhysicsNeMo checkpoint utilities | `save_checkpoint` / `load_checkpoint`，模型为 `.mdlus` |

上游代码核对基于 NVIDIA PhysicsNeMo 官方仓库提交 `426f7552da4b4fa675e404e8a4f437e27681b668`。项目实际训练使用上表所列已安装版本，版本检查结果要随训练日志保存。

本链路对照 NVIDIA 官方 `examples/cfd/darcy_fno` 的训练结构实现。2026-09-29 已在远程训练环境的 `.venv` 中只读核验以下 PhysicsNeMo 2.2.2 API 可导入且参数匹配：`FNO`、`DistributedManager`、`StaticCaptureTraining`、`StaticCaptureEvaluateNoGrad`、`LaunchLogger`、`PythonLogger`、`save_checkpoint` 和 `load_checkpoint`。核验只检查接口，没有启动模型训练。

完整数据和训练链路为：

```text
OpenFOAM 原始 U/p、力、探针和 omega(t)
  -> foamToVTK 数值格式转换
  -> PhysicsNeMo Curator Source -> Filter -> Sink
  -> 分轨迹 HDF5、manifest、train-only normalization
  -> PhysicsNeMo HDF5Reader -> DatasetBase -> TensorDict -> DataLoader
  -> Hydra 配置 + DistributedManager
  -> PhysicsNeMo FNO
  -> StaticCaptureTraining / StaticCaptureEvaluateNoGrad
  -> LaunchLogger + PhysicsNeMo checkpoints
  -> 独立 test 轨迹滚动评估
```

### 1.2 第一阶段数据

统一使用 `Re=100`、`L/D=5`、二维不可压缩串列双圆柱。前圆柱固定，后圆柱施加无量纲角速度 `omega`。每条轨迹使用 `t=80..160`，场输出间隔 `0.1`，因此有 801 帧和 800 个相邻时间步样本。

| 划分 | 动态轨迹 | 恒定转速轨迹 | 合计轨迹 | 一步样本 |
| --- | ---: | ---: | ---: | ---: |
| train | 12 | 3：`-1, 0, +1` | 15 | 12,000 |
| validation | 2 | 1：`-0.5` | 3 | 2,400 |
| test | 2 | 1：`+0.5` | 3 | 2,400 |

动态训练轨迹包含 8 条随机分段线性转速和 4 条多正弦转速；验证集使用训练中未出现的四分之一档位与独立多正弦；测试集使用独立四分之一档位与 chirp。整个轨迹只属于一个划分，禁止随机拆散相邻帧。

## 2. 登录、定位和版本记录

在本机终端执行：

```bash
ssh <SSH_HOST>
wsl.exe -d Ubuntu-24.04
cd ~/workspace/fluid_control
```

记录运行环境：

```bash
mkdir -p artifacts/tandem_fno
{
  date --iso-8601=seconds
  uname -a
  nvidia-smi
  .venv/bin/python --version
  .venv/bin/python - <<'PY'
import physicsnemo, torch
print("physicsnemo", physicsnemo.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import StaticCaptureTraining, load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger
print("physicsnemo_runtime_components", "ok")
PY
} |& tee artifacts/tandem_fno/environment.log
```

## 3. Gate A：确认 CFD 完整且网格检查合格

先查看求解器是否仍在运行：

```bash
cd cfd/tandem_cylinders
ps -eo pid,lstart,etimes,%cpu,%mem,cmd | grep pimpleFoam | grep -v grep || true
tail -n 8 dynamic_batch.log
```

所有动态轨迹结束后运行只读检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 validate_dynamic_dataset.py \
  --workers 4 \
  --write ../../artifacts/tandem_cylinders/dynamic_dataset_manifest.json \
  | tee ../../artifacts/tandem_cylinders/dynamic_dataset_validation.log
```

必须看到 16 条动态轨迹均通过，并满足：每条 16,000 个求解步、801 个 `U/p` 场、16,000 个前柱力、16,000 个后柱力和 16,000 个 32 点探针样本，日志含 `End`，且没有 NaN/Inf。

中等网格旋转算例完成后比较 `omega=+1` 的粗网格和中等网格：

```bash
python3 compare_convergence.py control_small_p100 control_grid_p100_medium \
  | tee ../../artifacts/tandem_cylinders/rotation_grid_comparison.json
cd ../..
```

训练前门槛：前后圆柱 `Cd_mean`、`Cl_rms`、主要频率的相对差异原则上均不超过 3%。如果超过，不进入 Curator，先补网格或时间步验证。

### 3.1 高转速动作域验证

以下命令创建并运行 `q=±2、±2.5` 稳定性试验，以及 `q=±2.5` 的时间步和网格试验。脚本拒绝覆盖已存在的算例或日志。

```bash
cd cfd/tandem_cylinders

python3 make_high_rotation_pilots.py

for name in \
  high_rotation_qm200_dt005 high_rotation_qp200_dt005 \
  high_rotation_qm250_dt005 high_rotation_qp250_dt005 \
  high_rotation_qm250_dt0025 high_rotation_qp250_dt0025 \
  high_rotation_qm250_medium_dt0025 high_rotation_qp250_medium_dt0025
do
  bash run_high_rotation_pilot.sh "$name"
done
```

将 `q=±2.5` 的粗、中网格 `Δt=0.0025` 算例续算至 `t=160`：

```bash
for name in \
  high_rotation_qm250_dt0025 high_rotation_qp250_dt0025 \
  high_rotation_qm250_medium_dt0025 high_rotation_qp250_medium_dt0025
do
  bash extend_high_rotation_pilot.sh "$name"
done
```

使用相同长统计窗生成粗、中网格报告并比较：

```bash
python3 analyze_high_rotation_pilots.py \
  high_rotation_qm250_dt0025 high_rotation_qp250_dt0025 \
  --window-start 120 --window-end 160 \
  --output ../../artifacts/tandem_cylinders/high_rotation_coarse_long_window.json

python3 analyze_high_rotation_pilots.py \
  high_rotation_qm250_medium_dt0025 high_rotation_qp250_medium_dt0025 \
  --window-start 120 --window-end 160 \
  --output ../../artifacts/tandem_cylinders/high_rotation_medium_long_window.json

python3 compare_high_rotation_pilots.py \
  ../../artifacts/tandem_cylinders/high_rotation_coarse_long_window.json \
  ../../artifacts/tandem_cylinders/high_rotation_medium_long_window.json \
  --output ../../artifacts/tandem_cylinders/high_rotation_grid_comparison.json
cd ../..
```

2026-09-29 实测结果：全部试验正常结束且数值有限；时间步减半后的 `Cl RMS` 变化为 0.33%–0.45%；`t=120..160` 的粗、中网格主频相同，`Cl RMS` 网格差为 2.723%–3.051%。扩展训练数据采用粗网格 `Δt=0.005`，中等网格 `Δt=0.0025` 仅用于独立数值验证和最终控制策略回放。

## 4. Gate B：把 OpenFOAM 场导出成 VTK

`foamToVTK` 只做格式转换，不生成或修改物理解。它读取每个原始时间目录的 `U` 和 `p`，输出二进制 `internal.vtu`，供 Curator 读取。

单轨迹冒烟测试已经完成：`dynamic_train_00` 在 36.36 秒内导出 801 个快照，目录约 2.2 GiB，日志正常结束且未检出 fatal/error。批量导出时跳过该轨迹，避免脚本的防覆盖检查中止任务。

完整批量导出也已完成：21/21 条轨迹均为 801 个 VTK 快照，自动检查退出码为 0，总计约 45 GiB。逐轨迹计数位于 `artifacts/tandem_cylinders/vtk_counts.txt`，批量日志位于 `artifacts/tandem_cylinders/vtk_export.log`。

创建固定的 21 条轨迹清单：

```bash
cat > /tmp/tandem_cases.txt <<'EOF'
dynamic_train_00
dynamic_train_01
dynamic_train_02
dynamic_train_03
dynamic_train_04
dynamic_train_05
dynamic_train_06
dynamic_train_07
dynamic_train_08
dynamic_train_09
dynamic_train_10
dynamic_train_11
dynamic_validation_00
dynamic_validation_01
dynamic_test_00
dynamic_test_01
control_small_m100
control_small_z000
control_small_p100
control_small_m050
control_small_p050
EOF

while read -r name; do
  vtk="cfd/tandem_cylinders/cases/$name/VTK_curator"
  if [[ -d "$vtk" ]]; then
    count=$(find "$vtk" -name internal.vtu -type f | wc -l)
    [[ "$count" == 801 ]] || { echo "$name existing VTK count is $count" >&2; exit 1; }
    echo "Reusing validated VTK: $name" >&2
  else
    echo "$name"
  fi
done < /tmp/tandem_cases.txt > /tmp/tandem_cases_to_export.txt

xargs -a /tmp/tandem_cases_to_export.txt -n1 -P2 \
  bash scripts/export_tandem_vtk.sh \
  |& tee artifacts/tandem_cylinders/vtk_export.log
```

逐条确认均为 801 个文件：

```bash
while read -r name; do
  count=$(find "cfd/tandem_cylinders/cases/$name/VTK_curator" \
    -name internal.vtu -type f | wc -l)
  printf '%-30s %s\n' "$name" "$count"
done < /tmp/tandem_cases.txt | tee artifacts/tandem_cylinders/vtk_counts.txt

awk '$2 != 801 {bad=1} END {exit bad}' \
  artifacts/tandem_cylinders/vtk_counts.txt
```

若某次导出中断，先阅读对应的 `cases/<name>/log.foamToVTK_curator`。确认输出无用后，手动删除该算例的 `VTK_curator` 和日志再重跑；脚本不会自动覆盖证据。

## 5. 安装独立的 PhysicsNeMo Curator 环境

Curator 当前未装在训练 `.venv` 中，并且从源码构建需要 Rust。以下命令把工具链、源码和虚拟环境都放在项目目录内，不修改训练环境：

2026-09-29 预检结果：Git 2.43.0、uv 0.12.5 和 Python 3.11 可用；系统起初没有 `rustc/cargo`、Curator 源码或 `.venv-curator`。官方仓库 HEAD 为固定提交 `86533e581b3550326d89e97cb4d4126e7061b416`，项目盘剩余约 397 GiB。首次按仓库元数据的 `Python >=3.11` 创建环境后，安装成功但导入失败：`core/base.py` 使用 `class Source[T](ABC)`，这是 Python 3.12 的 PEP 695 语法。为保持 NVIDIA 源码原样、不打私有补丁，Curator 环境改用 Python 3.12；PhysicsNeMo 训练 `.venv` 仍保持 Python 3.11。随后实测 `uv python find 3.12` 返回 `/usr/bin/python3.12`，版本为 Python 3.12.3。失败环境已保留为 `.venv-curator-py311-failed`（Python 3.11.16），新 `.venv-curator` 已用 Python 3.12.3 创建完成。

2026-09-29 基础安装实测：源码远端为 `https://github.com/NVIDIA/physicsnemo-curator.git`，工作树干净，HEAD 与固定提交完全一致；项目内工具链为 Rust/Cargo 1.98.1。Curator 0.1.0 基础包已成功构建并安装到 Python 3.12.3 的 `.venv-curator`，安装退出码为 0。依赖实测包括 NumPy 2.5.3、h5py 3.16.0、PyVista 0.49.0 和 VTK 9.7.1；完整安装日志保存为 `artifacts/tandem_cylinders/curator_install_py312.log`。采用官方 `VTKSource` 前继续安装仓库定义的 `mesh` extra，并将其中的 PhysicsNeMo 固定为与训练环境一致的 2.2.2。

导入验证也已通过：`Source`、`Filter`、`Sink` 和 `run_pipeline` 均从已安装的 `physicsnemo_curator` 成功加载，输出 `CURATOR_IMPORT_OK` 且退出码为 0。验证日志保存为 `artifacts/tandem_cylinders/curator_import_py312.log`。

```bash
cd ~/workspace/fluid_control
mkdir -p .tools
export RUSTUP_HOME="$PWD/.tools/rustup"
export CARGO_HOME="$PWD/.tools/cargo"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal
export PATH="$CARGO_HOME/bin:$PATH"

uv venv .venv-curator --python 3.12
git clone https://github.com/NVIDIA/physicsnemo-curator.git \
  .tools/physicsnemo-curator
git -C .tools/physicsnemo-curator checkout \
  86533e581b3550326d89e97cb4d4126e7061b416

uv pip install --python .venv-curator/bin/python \
  --no-sources-package nvidia-physicsnemo \
  "./.tools/physicsnemo-curator[mesh]" \
  "nvidia-physicsnemo==2.2.2" h5py numpy
```

首次补装 `mesh` extra 时未加 `--no-sources-package`，Curator 仓库的 `[tool.uv.sources]` 将 `nvidia-physicsnemo` 改指 GitHub HEAD，因而与显式版本 `2.2.2` 冲突，解析退出码为 1且未安装任何包。训练环境元数据确认其现有 `nvidia-physicsnemo 2.2.2` 来自软件包索引（`direct_url=None`）。修正命令仅忽略该包的 Git 源覆盖，保留官方 extra 和版本一致性；失败日志保存在 `artifacts/tandem_cylinders/curator_mesh_extra_install.log`。

随后使用 `--no-sources` 已成功绕过 Git 源覆盖，但官方 `mesh` extra 的传递依赖仍声明已弃用的 PyPI 占位包 `sklearn==0.0.post12`，该包默认拒绝构建，安装退出码为 1。按该包错误信息提供的兼容方式，仅在安装命令中设置 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True`；这不会修改 NVIDIA 或项目源码。失败日志保存在 `artifacts/tandem_cylinders/curator_mesh_extra_install_retry2.log`。

第三次安装已通过：`mesh` extra 共解析 134 个包、安装 97 个包，退出码为 0。核心实测版本为 `nvidia-physicsnemo 2.2.2`、Torch 2.14.0、TensorDict 0.14.2、PyArrow 25.0.1、Warp 1.17.0；Curator 仍为固定源码提交构建的 0.1.0。成功日志保存在 `artifacts/tandem_cylinders/curator_mesh_extra_install_retry3.log`。

验证导入和提交号：

```bash
git -C .tools/physicsnemo-curator rev-parse HEAD
.venv-curator/bin/python - <<'PY'
import h5py, numpy, pyvista
import physicsnemo_curator
from physicsnemo.mesh import Mesh
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.run import run_pipeline
print("curator_import_ok", physicsnemo_curator.__file__)
print("h5py", h5py.__version__)
print("numpy", numpy.__version__)
print("pyvista", pyvista.__version__)
print("Mesh/VTKSource", Mesh, VTKSource)
PY
```

若机器已经具有可用的 `cargo`，可跳过 rustup 安装，仅保留 `export PATH`、建环境、固定提交和安装步骤。

## 6. Gate C：运行 Curator

项目 Curator 脚本以 NVIDIA 已有能力为主体，只为本实验补充轨迹分组、标签对齐和质量门槛：

```text
TandemTrajectorySource
  └─ 官方 Curator VTKSource
       └─ 官方 PhysicsNeMo Mesh.sample_data_at_points
  -> NumericalQualityFilter
  -> TrajectoryHDF5Sink
  -> 官方 Curator run_pipeline
```

- VTK 读取：使用 Curator 官方 `physicsnemo_curator.domains.mesh.sources.vtk.VTKSource`，只加载 `U/p`。
- 网格采样：使用官方 `physicsnemo.mesh.Mesh.sample_data_at_points`，把非结构场采样到固定 `256×128` 网格；范围为 `x=[8,25]`、`y=[4,11]`。
- 项目 Source 适配器：只负责把 801 个官方 Mesh 帧组成一条轨迹，并对齐本工况的动作、力和时间。
- Filter：检查帧数、数组形状、有限数、有效区域覆盖率和动作范围。
- Sink：每条轨迹写一个压缩 HDF5；训练集、验证集和测试集分别落盘。
- 压力：每帧在有效流体区域减去空间均值，消除不可压压力的任意常数。
- 归一化：只用 train 划分计算均值和标准差，避免验证集和测试集泄漏。

2026-09-29 接口核验：代表性 `internal.vtu` 含 39,336 个点、19,290 个单元，point/cell 数据都含 `p/U`，field data 含 `TimeValue=80.0`；远端 PhysicsNeMo 2.2.2 已确认提供 `Mesh.sample_data_at_points`。因此脚本不再直接调用 `pyvista.read/grid.sample`。

官方单帧链路实测已通过：`VTKSource` 发现 801 个文件；首帧转换为 `physicsnemo.mesh.Mesh` 后为 39,336 个点和 115,740 个四面体单元，point/cell `p/U` 与 global `TimeValue` 均存在。在 `64×32` 规则网格上调用 `Mesh.sample_data_at_points` 得到 `U=(2048,3)`、`p=(2048,)`，有效点 2,024/2,048，覆盖率 0.98828125；输出 `OFFICIAL_VTK_MESH_API_OK` 且退出码为 0。日志为 `artifacts/tandem_cylinders/curator_official_vtk_api_check.log`。

先对一条轨迹做 smoke test：

```bash
rm -rf data/curated/tandem_smoke
.venv-curator/bin/python scripts/curate_tandem_cfd.py \
  --cases-root cfd/tandem_cylinders/cases \
  --output data/curated/tandem_smoke \
  --nx 256 --ny 128 --limit 1 \
  |& tee artifacts/tandem_cylinders/curator_smoke.log
```

2026-09-29 冒烟运行实测已完成：官方链路处理 `dynamic_test_00` 的 801/801 帧，Curator 写出 1 条轨迹，退出状态为 0。墙钟时间 4 分 21.06 秒，CPU 利用率 1919%，峰值常驻内存 2,332,712 KiB（约 2.22 GiB），swap 为 0。Curator 的 `0/1` 表示整条轨迹尚未交给 Sink；逐帧日志从 1 推进到 801，Sink 完成后变为 `1/1 (100%)`。日志位于 `artifacts/tandem_cylinders/curator_smoke.log`。

检查 smoke HDF5：

```bash
.venv-curator/bin/python - <<'PY'
from pathlib import Path
import h5py
p = next(Path("data/curated/tandem_smoke").rglob("*.h5"))
with h5py.File(p, "r") as f:
    print(p)
    for key in f:
        print(key, f[key].shape, f[key].dtype)
PY
```

HDF5 冒烟质量检查已通过：`state=(801,3,128,256)`、`mask=(801,1,128,256)`、`omega=(801,1)`、`force=(801,4)`，字段类型与压缩设置符合定义；时间从 80.0 严格递增到 160.0。mask 仅含 0/1，有效覆盖率为 0.9869384765625；动作范围为 `[-0.75,0.75]`，力数据全部有限；逐帧最大有效区域压力均值绝对值为 `1.6456821227265347e-09`，无效区域状态最大绝对值为 0。输出 `SMOKE_HDF5_OK`，退出码为 0；日志为 `artifacts/tandem_cylinders/curator_smoke_hdf5_check.log`。

预期 `state=(801,3,128,256)`、`mask=(801,1,128,256)`、`omega=(801,1)`、`force=(801,4)`、`time=(801,1)`。通过后运行完整 21 条轨迹：

```bash
rm -rf data/curated/tandem_cylinders
.venv-curator/bin/python scripts/curate_tandem_cfd.py \
  --cases-root cfd/tandem_cylinders/cases \
  --output data/curated/tandem_cylinders \
  --nx 256 --ny 128 \
  |& tee artifacts/tandem_cylinders/curator_full.log
```

确认 manifest：

```bash
.venv-curator/bin/python -m json.tool \
  data/curated/tandem_cylinders/manifest.json
.venv-curator/bin/python -m json.tool \
  data/curated/tandem_cylinders/normalization.json
find data/curated/tandem_cylinders -name '*.h5' -type f | sort
du -sh data/curated/tandem_cylinders
```

预期计数为 train 15、validation 3、test 3。

2026-09-29 完整执行结果：Curator 21/21 完成，退出码 0，墙钟时间 1:18:48，峰值常驻内存 2,620,552 KiB，swap 为 0。输出 21 个 HDF5、总计 5.5 GiB，划分为 train/validation/test = 15/3/3。独立脚本 `scripts/validate_tandem_curated.py` 随后逐文件检查形状、时间轴、有限数、掩码、压力零均值和无效区，并重新计算 train-only normalization；结果 `CURATED_DATASET_OK`，退出码 0。证据为 `curator_full.log`、`curated_validation.log` 和 `curated_validation.json`。

## 7. Gate D：验证 PhysicsNeMo Datapipe

安装本项目本身，使训练脚本能导入 `fluid_control.tandem_datapipe`：

```bash
uv pip install --python .venv/bin/python -e .
```

`TandemWindowDataset` 继承官方 `physicsnemo.datapipes.DatasetBase`，内部使用 `HDF5Reader` 懒读取当前帧和下一帧，并返回 `TensorDict` 与样本 metadata。训练使用官方 `physicsnemo.datapipes.DataLoader` 完成采样、批处理和预取。随后使用 PhysicsNeMo `DistributedManager` 选择设备、`StaticCaptureTraining` 和 `StaticCaptureEvaluateNoGrad` 封装训练与验证、`LaunchLogger` 记录指标，并使用 `save_checkpoint/load_checkpoint` 保存和恢复模型及优化器状态。每个样本为：

- 输入 `x[6,128,256]`：标准化 `u,v,p`、有效掩码、当前 `omega_t`、下一时刻 `omega_t+1`；
- 场标签 `delta[3,128,256]`：下一帧减当前帧；
- 力标签 `force[2]`：下一时刻后圆柱 `Cd,Cl`；
- 划分边界：窗口不跨轨迹。

执行检查：

```bash
.venv/bin/python - <<'PY'
from fluid_control.tandem_datapipe import TandemWindowDataset
for split, expected in (("train", 12000), ("validation", 2400), ("test", 2400)):
    ds = TandemWindowDataset("data/curated/tandem_cylinders", split)
    sample, metadata = ds[0]
    print(split, len(ds), {k: tuple(v.shape) for k, v in sample.items()})
    print("metadata", metadata)
    assert len(ds) == expected
    assert sample["x"].shape == (6, 128, 256)
    assert sample["delta"].shape == (3, 128, 256)
    assert sample["force"].shape == (2,)
    assert all(v.isfinite().all() for v in sample.values())
    ds.close()
PY
```

2026-09-29 实测结果：官方 `DatasetBase`、`HDF5Reader`、`DataLoader` 链路通过。数据长度为 train 12,000、validation 2,400、test 2,400；连续三个 batch 均为 `x=(4,6,128,256)`、`delta=(4,3,128,256)`、`force=(4,2)`、`mask=(4,1,128,256)`、`time=(4,1)`，全部为有限数。输出 `PHYSICSNEMO_DATAPIPE_OK`，退出码 0；日志为 `artifacts/tandem_cylinders/datapipe_validation.log`。

## 8. Gate E：训练前 smoke test

使用 GPU 0/1 跑 1 个 epoch 和抽稀样本。每个进程的 CUDA allocator 上限为显卡总显存的 0.75，并每 5 秒记录实际显存，确保每卡至少保留 15 GiB：

```bash
bash scripts/run_tandem_fno_smoke.sh
```

首次尝试通过断开式 SSH 后台启动，`torchrun` 收到 SIGHUP，尚未进入有效训练便以退出码 1 结束；GPU 回到空闲，无数据或 checkpoint 损坏。该问题属于启动方式，不是模型或数据错误；失败日志保留在 `artifacts/tandem_fno_smoke_sighup_20260929_092040/`。后续在用户 tmux 前台执行上述脚本。

第二次由用户在 tmux 前台执行，两个 rank 均在 `DistributedManager.initialize()` 的 NCCL process-group 初始化阶段失败：NCCL 2.30.7 报 `Cuda failure 999 'unknown error'`，退出码 1。该次仍未进入模型构建或训练；随后出现的 `Process group cannot be None` 是初始化失败后的清理异常。先运行 `scripts/diagnose_tandem_gpus.py` 核验两卡独立 CUDA 和 peer access，再决定 NCCL transport 参数；不直接降级成单 GPU。

GPU 诊断结果：驱动 596.72，PyTorch 2.14.0+cu130，CUDA 13.0，NCCL 2.30.7；GPU 0/1 均可独立完成 CUDA 矩阵乘法。但 WSL 的 `nvidia-smi topo -m` 无法生成拓扑矩阵，且 `torch.cuda.can_device_access_peer` 对 0→1 和 1→0 均返回 false。输出 `CUDA_DEVICE_CHECK_OK`、退出码 0；日志为 `artifacts/tandem_cylinders/gpu_diagnostic.log`。因此下一步先用 `NCCL_P2P_DISABLE=1` 做最小双 rank all-reduce 测试。

仅设置 `NCCL_P2P_DISABLE=1` 仍出现 CUDA 999。第一组可工作的回退配置同时关闭 P2P、SHM、IB 和 cuMem device/host allocation，并指定 `NCCL_SOCKET_IFNAME=lo`；NCCL 两个 rank 经内置 `NET/Socket` 完成 all-reduce，两侧结果均为 3.0。日志为 `nccl_socket_check.log` 与 `nccl_socket.*.log`。

为避免长期使用较慢的 loopback Socket，随后进行逐项消融测试：保持 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、`NCCL_CUMEM_ENABLE=0` 和 `NCCL_CUMEM_HOST_ENABLE=0`，把 `NCCL_SHM_DISABLE` 改为 `0`。2026-09-29 实测两个 rank 初始化成功，4 个通信通道均明确记录为 `via SHM/direct`，all-reduce 两侧结果均为 3.0，输出 `NCCL_ALL_REDUCE_OK`，退出码 0。由此确认 CUDA 999 的关键触发条件不是传统 SHM，而是当前 WSL 映射下不可用的 GPU P2P 和 NCCL cuMem 路径。训练启动脚本现采用 SHM 配置；Socket 仅保留为已验证的故障回退。验证日志为 `artifacts/tandem_cylinders/nccl_shm_check.log`。

当前训练环境变量为：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SOCKET_IFNAME=lo
```

正式多步训练前曾出现 WSL `getpwuid` 和 `/etc/default/locale` I/O 错误。检查确认 `ext4.vhdx` 所在宿主盘仅剩约 30 MiB；释放宿主空间后 WSL 正常启动，工程、HDF5 和 checkpoint 均可读取，内核日志未出现新的 ext4 I/O 错误。为避免训练检查点继续扩展 VHDX，`artifacts/` 已逐文件核对后迁至 `<WINDOWS_DATA_DRIVE>:\WSLData\fluid_control\artifacts`，Linux 原路径保留符号链接；HDF5 训练数据继续位于 ext4。该事件属于宿主存储容量问题，不是模型或 PhysicsNeMo 故障。

修复后的双卡训练冒烟测试已于 2026-09-29 通过。1 个 epoch 的训练 loss 为 `0.2025793`；验证集物理单位场 MAE 为 `0.00336758`、RMSE 为 `0.00516719`，归一化力系数 MAE 为 `0.746278`；训练 epoch 用时 `8.36 s`、`88.90 ms/iter`，退出码 0。PhysicsNeMo 模型、训练状态、最佳 checkpoint、完整配置、runtime metadata 和 history 均已生成并通过 JSON/非空文件审计，输出 `SMOKE_TRAINING_ARTIFACTS_OK`。checkpoint 保存已限制为 rank 0，避免两个进程并发覆盖同一文件。GPU 0/1 实测最大占用分别为 1,726/1,858 MiB，最低剩余分别为 70,714/70,582 MiB，满足每卡至少保留 15 GiB 的约束。

同配置单卡对照使用 batch size 8，训练为 `69.71 ms/iter`、约 `114.8 samples/s`；双卡每 rank batch size 8、全局 batch size 16，约 `180.0 samples/s`。双卡 SHM 相对单卡吞吐提升约 1.57 倍，并行效率约 78.4%，因此正式训练继续使用 GPU 0/1。该结果只说明 SHM 对本任务有实际收益，不能等同于原生 Linux 中可用 PCIe/NVLink P2P 时的 NCCL 性能。单卡对照日志位于 `artifacts/tandem_fno_smoke_single/train.log`，最终双卡冒烟日志位于 `artifacts/tandem_fno_smoke/train.log`。

正式训练最初以每 rank batch size 8 启动并稳定收敛。根据项目负责人对完成速度和最终精度的取舍，后续从现有 checkpoint 恢复时改为每 rank batch size 64、全局 batch size 128，并把每进程显存比例上限改为 0.85，理论上每张 72 GB GPU 至少保留约 10.8 GiB。恢复时沿用 checkpoint 中的优化器和余弦学习率状态，不按 batch 比例放大学习率，以降低中途切换 batch 带来的不稳定风险。最终精度必须以独立 test rollout 为准。

验收：进程返回码为 0，损失为有限数，并生成：

```bash
find artifacts/tandem_fno_smoke -maxdepth 2 -type f -printf '%P\n' | sort
python3 -m json.tool artifacts/tandem_fno_smoke/training_history.json
```

应能看到 `resolved_config.yaml`、`physicsnemo.log`、`checkpoints/*.mdlus`、`best/*.mdlus` 和相应的 `checkpoint.*.pt`。`.mdlus` 是 PhysicsNeMo 模型 checkpoint；`.pt` 保存 epoch、优化器、scheduler、AMP capture 状态和 metadata。重复相同命令时，`load_checkpoint` 会从最新 epoch 恢复。

如显存不足，首先把 `--batch-size 8` 改为 `4`；如 DataLoader worker 报错，把 `--workers 4` 改为 `0`。将改动记入日志。

## 9. 正式训练

默认模型配置保存在 `conf/tandem_fno.yaml`：二维 FNO，输入 6 通道，输出 5 通道，4 个 Fourier 层，隐通道 32，模态数 `24×24`。前三个输出学习标准化场增量，后两个输出经流体掩码空间平均后学习后圆柱标准化 `Cd/Cl`。损失为场增量 MSE 加 `0.2 ×` 力系数 MSE；优化器 AdamW，学习率 `2e-4`，50 epochs，余弦退火。训练由 PhysicsNeMo `StaticCaptureTraining` 管理并使用梯度裁剪。PhysicsNeMo 2.2.2 的 FNO metadata 将 GPU AMP 标记为不支持，冒烟测试中框架自动关闭了 bf16；因此正式配置显式使用 FP32，避免请求无效 AMP，同时保持框架行为可复现。

模型选择依据：本阶段数据是同一几何上的固定二维规则网格，目标是学习带旋转动作条件的时空流场算子，因此采用 PhysicsNeMo FNO。MeshGraphNet 更适合保留非结构网格的任务，SFNO 针对球面，均不如 FNO 直接匹配当前表示。FNO 是第一阶段基线选择；是否优于其他候选必须以后续同划分对照实验判断。

正式执行入口为：

```bash
bash scripts/run_tandem_fno_train.sh
```

脚本固定使用 GPU 0/1 和经验证的 NCCL SHM 配置，调用 `torchrun --nproc_per_node=2`。当前每 rank batch size 为 64，全局 batch size 为 128；重复执行时通过 PhysicsNeMo `load_checkpoint` 从最新 epoch 恢复，并在覆盖 runtime 配置前自动归档上一阶段日志。

另开终端查看运行：

```bash
watch -n 2 nvidia-smi
tail -f artifacts/tandem_fno_train.log
```

每个 epoch 都由 `LaunchLogger` 输出 train loss、验证集场 MAE/RMSE 和后柱力系数的标准化 MAE。完整恢复点位于 `checkpoints/`；验证集物理单位场 MAE 改善时，使用 PhysicsNeMo checkpoint 格式写入 `best/`。

**正式训练结果：已通过。**50/50 epochs 完成，`formal_training_exit=0`，无残留训练进程。Epoch 50 同时是验证场 MAE 最佳 checkpoint：train loss `3.0743371e-05`、场 MAE `5.3888804e-04`、场 RMSE `7.9748279e-04`、归一化后柱力 MAE `8.5806989e-03`。后柱力 MAE 的单独最低值出现在 Epoch 42，为 `8.4354119e-03`；当前 `best/` 按主要指标场 MAE选择 Epoch 50。恢复后的 batch 64 阶段用时 14:56.58，峰值 CPU 常驻内存约 3.30 GiB、swap 为 0。runtime metadata 记录双 GPU、显存比例上限 0.85 和理论保留 10.754 GiB/卡。

### 9.1 多步 rollout 微调

单步模型在长 rollout 中存在累积误差。`TandemRolloutDataset` 继续使用 PhysicsNeMo `DatasetBase` 和 `HDF5Reader`，但每个样本返回连续 10 步状态、动作和后柱受力。`scripts/train_tandem_fno_rollout.py` 从 Epoch 50 模型初始化，以 10 步自回归损失进行微调；teacher forcing 在前 10 Epoch 从 0.5 线性降为 0，随后完全使用模型自身状态。模型、分布式管理、训练封装、日志与 checkpoint 均继续使用 PhysicsNeMo API。

完整 10 步显存探测得到：每 rank batch 64 时，GPU0/1 峰值约为 51.4/51.6 GiB，每卡剩余约 20.8 GiB。因此正式运行采用全局 batch 128：

```bash
OUTPUT_DIR=artifacts/tandem_fno_rollout \
  bash scripts/run_tandem_fno_rollout.sh \
  training.batch_size=64 training.workers=4
```

配置位于 `conf/tandem_fno_rollout.yaml`。最佳模型按“10 步末端流场 MAE + `0.1 ×` 末端后柱受力 MAE”选择，原单步模型和多步模型保存在不同目录。

**多步微调结果：已通过。**20/20 Epoch 完成，退出码为 0，最佳 checkpoint 为 Epoch 20。验证集 10 步平均/末端流场 MAE 为 `0.00225494/0.00366929`，平均/末端后柱受力 MAE 为 `0.00582447/0.00749916`。墙钟时间为 1:13:36，GPU0/1 峰值显存为 52,002/52,178 MiB，最低剩余显存为 20,438/20,262 MiB。

## 10. 测试集滚动预测

以下命令保留了最初冻结单步模型时采用的 1/10/50 步评估，用于追溯第一版验收结果：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_tandem_fno.py \
  --data data/curated/tandem_cylinders \
  --config conf/tandem_fno.yaml \
  --checkpoint-dir artifacts/tandem_fno/best \
  --output artifacts/tandem_fno/evaluation.json \
  --visualization-dir artifacts/tandem_fno/rollout_visualizations \
  --visualizations-per-horizon 3 \
  --split test --horizons 1 10 50 \
  |& tee artifacts/tandem_fno_evaluate.log

python3 -m json.tool artifacts/tandem_fno/evaluation.json | less
find artifacts/tandem_fno/rollout_visualizations -name '*.png' -type f | sort
```

该初始评估逐条测试轨迹给出 1、10、50 步滚动场 MAE、后圆柱 `Cd/Cl` MAE、最大预测场幅值和非有限值检查，并同时给出 persistence 基线。每条轨迹、每个 horizon 选择三个 rollout，共生成 27 张 PNG。最终模型比较使用下述固定 5 帧起点间隔和 100 步 horizon 的严格评估。

**独立 test rollout：已通过。**Epoch 50 checkpoint 完成全部 3 条 test 轨迹、1/10/50 步评估，`evaluation_exit=0`；共检查 2,400/240/48 个片段，所有 horizon 均 `stable=true`、`failed_segments=0`。汇总结果如下：

| horizon | 场 MAE | persistence 场 MAE | 场误差降低 | 后柱力 MAE | persistence 力 MAE | 力误差降低 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0006102 | 0.0087487 | 93.03% | 0.0052400 | 0.0664161 | 92.11% |
| 10 | 0.0047165 | 0.0827404 | 94.30% | 0.0131261 | 0.6244206 | 97.90% |
| 50 | 0.0160685 | 0.1038332 | 84.52% | 0.0489211 | 0.8106566 | 93.97% |

27/27 张 PNG 已生成。视觉抽查显示恒定转速轨迹在 50 步仍较好复现尾流；动态动作轨迹保留主要涡结构和相位，但出现累积的高频纹理误差。最难的 `dynamic_test_01` 在 50 步的场 MAE 为 0.0218156、后柱力 MAE 为 0.0760545，其中 `Cl` MAE 为 0.0949183。结果满足第一阶段代理模型门槛，但未来闭环不应把长时间纯开环代理 rollout 当作真实系统；应采用有限预测窗口并用新观测持续校正状态。

为提高统计覆盖率，模型优化对照使用固定 5 帧起点间隔，并增加 100 步评估。基线和多步模型必须使用完全相同的参数：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_tandem_fno.py \
  --data data/curated/tandem_cylinders \
  --config conf/tandem_fno.yaml \
  --checkpoint-dir artifacts/tandem_fno/best \
  --output artifacts/tandem_fno/evaluation_dense.json \
  --visualization-dir artifacts/tandem_fno/rollout_visualizations_dense \
  --horizons 1 10 50 100 \
  --segment-stride 5 \
  --evaluation-batch-size 8 \
  --visualizations-per-horizon 0

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_tandem_fno.py \
  --data data/curated/tandem_cylinders \
  --config conf/tandem_fno.yaml \
  --checkpoint-dir artifacts/tandem_fno_rollout/best \
  --output artifacts/tandem_fno_rollout/evaluation_dense.json \
  --visualization-dir artifacts/tandem_fno_rollout/rollout_visualizations \
  --horizons 1 10 50 100 \
  --segment-stride 5 \
  --evaluation-batch-size 8 \
  --visualizations-per-horizon 3

python3 scripts/compare_rollout_evaluations.py \
  artifacts/tandem_fno/evaluation_dense.json \
  artifacts/tandem_fno_rollout/evaluation_dense.json \
  --output artifacts/tandem_fno_rollout/baseline_comparison.json \
  --markdown artifacts/tandem_fno_rollout/baseline_comparison.md
```

基线检查了 480/477/453/423 个窗口，1/10/50/100 步流场 MAE 分别为 `0.00061045`、`0.00471497`、`0.01596130`、`0.03894886`，全部保持数值稳定。多步微调模型使用完全相同的命令参数评估，并通过 `scripts/compare_rollout_evaluations.py` 生成逐 horizon 和逐工况对照。

多步模型采用相同测试窗口完成评估，全部 1,833 个窗口稳定。相对单步基线，10/50/100 步流场 MAE 分别降低 9.47%/16.80%/24.97%，后柱受力 MAE 分别降低 38.37%/29.35%/26.68%。1 步受力 MAE 增加 0.81%；恒定 `omega=+0.5` 工况的 100 步受力 MAE 增加 17.34%，但两条动态轨迹在同一指标上改善 30.56% 和 32.45%。最终评估生成 36 张 `u/v/p` Ground Truth、Prediction 和 Absolute Error 对比图。

参考论文的物理设置、无控制统计、控制目标和当前复现边界见 `docs/PAPER_REPRODUCTION.md`。

## 11. ParaView VTK 导出

### 11.1 OpenFOAM 原生流场

下列命令从已有 `foamToVTK` 输出中选择 7 个工况，每个工况保留 `t=80、100、120、140、160` 五个时刻。恒定转速覆盖 `omega=-1、-0.5、0、0.5、1`，另包含两条动态测试轨迹。

```bash
.venv-curator/bin/python scripts/package_tandem_cfd_vtk.py \
  --output artifacts/tandem_paraview/cfd
```

每个工况目录生成一个 `.pvd` 时间序列和五个 OpenFOAM 原生 `.vtu`；字段为 `U` 和 `p`。

### 11.2 PhysicsNeMo 预测流场

预测导出加载多步微调 Epoch 20 checkpoint，从 3 条独立 test 轨迹的 frame 100 开始执行 100 步自回归，在 step 1 和每 10 步写出一个二进制 VTK RectilinearGrid：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  .venv-curator/bin/python scripts/export_tandem_prediction_vtk.py \
  --data data/curated/tandem_cylinders \
  --config conf/tandem_fno.yaml \
  --checkpoint-dir artifacts/tandem_fno_rollout/best \
  --output artifacts/tandem_paraview/prediction \
  --split test --start 100 --steps 100 --write-every 10 --include-steps 1

.venv-curator/bin/python scripts/validate_tandem_vtk_export.py \
  artifacts/tandem_paraview
```

每个 `.vtr` 同时包含 `ground_truth_U/p`、`prediction_U/p`、`absolute_error_U/p` 和 `valid_mask`。物理时间、转速及真实/预测后柱 `Cd/Cl` 保存在 Field Data；每个轨迹目录的 `.pvd` 可直接在 ParaView 中播放。

2026-09-29 的实际导出包含 35 个 CFD `.vtu` 和 33 个预测 `.vtr`。全部 68 个数据文件已由 PyVista 回读，字段、点数和有限值检查通过，状态为 `PARAVIEW_VTK_EXPORT_OK`。Mac 副本位于 `results/tandem_paraview_20260929/`，压缩包为 `results/tandem_paraview_20260929.tar.gz`，SHA-256 为 `95fd50e5c1b9451fb19c7a01e056274c6392448c5efffa71effd2d56cecedc3a`。

连续时间序列另存于 Mac 的 `results/paraview_continuous_20260929/`。训练入口 `dynamic_train_00.vtm.series` 直接引用 OpenFOAM 生成的 801 个 `.vtm/.vtu`，时间范围为 `80..160`、间隔为 `0.1`。PhysicsNeMo Epoch 20 模型对三条独立测试轨迹分别连续 rollout 100 步，并转换为三个 `.pvd + 100 VTU` 时间序列。四个入口均已使用 ParaView 6.1.0 实际打开，时间轴和字段检查通过。

## 12. 产物与审计位置

| 产物 | 路径 |
| --- | --- |
| 原始 OpenFOAM 场、力和探针 | `cfd/tandem_cylinders/cases/<case>/` |
| CFD 动态数据校验 | `artifacts/tandem_cylinders/dynamic_dataset_manifest.json` |
| 旋转工况网格比较 | `artifacts/tandem_cylinders/rotation_grid_comparison.json` |
| 高转速短时稳定性 | `artifacts/tandem_cylinders/high_rotation_pilots_dt005.json` |
| 高转速时间步检查 | `artifacts/tandem_cylinders/high_rotation_pilots_dt0025.json` |
| 高转速长窗网格比较 | `artifacts/tandem_cylinders/high_rotation_grid_comparison.json` |
| VTK 转换日志 | `artifacts/tandem_cylinders/vtk_export.log` |
| Curator 日志 | `artifacts/tandem_cylinders/curator_*.log` |
| Curator HDF5、manifest、归一化 | `data/curated/tandem_cylinders/` |
| 环境记录 | `artifacts/tandem_fno/environment.log` |
| 最佳 checkpoint 与训练历史 | `artifacts/tandem_fno/best/`、`training_history.json` |
| 测试结果 | `artifacts/tandem_fno/evaluation.json` |
| 多步最佳 checkpoint 与训练历史 | `artifacts/tandem_fno_rollout/best/`、`training_history.json` |
| 严格基线与多步评估 | `artifacts/tandem_fno/evaluation_dense.json`、`artifacts/tandem_fno_rollout/evaluation_dense.json` |
| 模型误差对照 | `artifacts/tandem_fno_rollout/baseline_comparison.json` |
| 多步流场可视化 | `artifacts/tandem_fno_rollout/rollout_visualizations/` |
| 多步结果校验和 | `artifacts/tandem_fno_rollout/EVIDENCE_SHA256SUMS` |
| ParaView CFD 与预测结果 | `artifacts/tandem_paraview/` |
| ParaView 导出检查 | `artifacts/tandem_paraview/validation.json` |

所有命令都从原始数值文件生成可复查产物。不要只保留终端截图；保留日志、JSON、固定提交号、原始算例配置和 checkpoint。

## 13. 扩展动作域数据与重新训练

扩展数据用于提高论文相关转速范围内的代理精度，物理条件仍为 `Re=100、L/D=5、U∞=1`。后圆柱角速度扩展为 `omega∈[-5,5]`，对应表面速度比 `q∈[-2.5,2.5]`。32 条轨迹均从无控制解 `t=80` 重启，使用粗网格 19,290 单元、`Δt=0.005`，保存 `t=80..160` 的 801 个流场快照。

| 划分 | 轨迹数 | 动作类型 | 一步样本数 |
| --- | ---: | --- | ---: |
| train | 24 | 随机 ramp、multisine、chirp、边界保持 | 19,200 |
| validation | 4 | 独立参数与相位组合 | 3,200 |
| test | 4 | 独立参数与相位组合 | 3,200 |

全部 32 条 OpenFOAM 轨迹正常到达 `t=160`，Curator 生成 24/4/4 个 HDF5，完整数据审计返回 `CURATED_DATASET_OK`。训练集最大 `|domega/dt|=6.666668`，validation/test 最大值均为 `6.666668`，没有动作变化率超出训练覆盖。原始时间目录和 VTK 采用四工况流式处理，在 HDF5 校验通过后清理可再生文件，避免 WSL 虚拟磁盘再次占满宿主 D 盘。

扩展模型配置位于 `conf/tandem_fno_expanded.yaml`：输入仍为当前标准化 `u/v/p`、有效域 mask、当前与下一时刻动作，共 6 通道；输出为三通道场增量和后柱 `Cd/Cl`，共 5 通道。PhysicsNeMo FNO 使用 48 个隐通道、5 个 Fourier 层、`32×32` 模态及 128 宽度解码器，共 47,222,525 个可训练参数。动作在 DataPipe 中除以数据 manifest 的 `max_abs_omega=5`，物理单位动作仍保存在 HDF5 和评估报告中。

双 GPU 冒烟从每 rank batch 224 开始并一次通过，因此正式训练采用每 rank 224、全局 batch 448、FP32、AdamW、初始学习率 `2e-4`、余弦退火和 80 Epoch。GPU0/1 截至 Epoch 27 的峰值显存为 43,350/43,526 MiB，最低剩余为 29,090/28,914 MiB，利用率均达到 100%。该设置满足显存保留约束；显存低于上限是模型实际计算图所需，并不通过无意义缓存强行占满。

训练在 Epoch 19 checkpoint 完整写出后被终端中断，随后通过 PhysicsNeMo `load_checkpoint` 恢复模型、优化器和 scheduler，并从 Epoch 20 继续。中断未造成数据或 checkpoint 损坏。

截至 2026-09-30 10:20 的 Epoch 27 快照如下；这些是训练进行中的诊断值，最终模型以完成 80 Epoch 后的最佳验证 checkpoint 和独立 test rollout 为准。

| 指标 | Epoch 1 | Epoch 27 | 相对下降 |
| --- | ---: | ---: | ---: |
| train loss | 0.203610 | 0.0004880 | 99.76% |
| 验证场 MAE | 0.0071696 | 0.0028186 | 60.69% |
| 验证场 RMSE | 0.0119896 | 0.0043028 | 64.11% |
| 验证受力 MAE（标准化） | 0.799182 | 0.0268695 | 96.64% |

![扩展动作域 FNO 训练曲线（Epoch 27 快照）](docs/assets/expanded-training-curves-progress.png)

当前曲线没有出现发散：train loss 连续下降；验证场 MAE 仅在早期出现两次小幅回升，Epoch 5 后持续下降；验证场 RMSE 仅一次早期回升；受力 MAE 连续下降。训练误差和验证误差同时改善，尚未出现持续性的训练下降而验证恶化，因此当前没有过拟合证据。验证场误差在后半段下降速度减慢，继续执行余弦退火仍有必要；是否得到更好的长期模型必须由 1/10/50/100 步独立 test rollout 判断，不能只根据一步曲线决定。

实时曲线由以下命令提供，浏览器每 3 秒重新读取 `training_history.json`：

```bash
.venv/bin/python scripts/serve_tandem_training_dashboard.py \
  --root . --host 0.0.0.0 --port 8765
```

完整自动入口为 `scripts/run_expanded_training_pipeline.sh`。它依次执行全数据校验、PhysicsNeMo DataPipe 检查、batch 冒烟、80 Epoch 单步训练、独立测试、10 步 rollout 微调和最终模型对照。扩展训练证据位于 `artifacts/tandem_fno_expanded_v1/`、`artifacts/tandem_fno_rollout_expanded_v1/` 和 `artifacts/tandem_cylinders/expanded_*`。

## 14. 无 teacher forcing 与动作条件审计

为避免课程式 teacher forcing 在训练早期使用真实中间场造成部署差距，使用相同数据、初始模型、优化器、随机种子和 10 步损失执行完全自由 rollout 消融：

```bash
bash scripts/run_no_tf_ablation_pipeline.sh
```

该流程训练 30 Epoch，只在 Epoch 10、20、30 保存 checkpoint，然后自动执行 1/10/50/100 步独立 test 和模型对照。最终状态为 `NO_TF_ABLATION_PIPELINE_OK`，最佳模型是 `artifacts/tandem_fno_rollout_no_tf_v1/best/FNO.0.30.mdlus`。

无 teacher forcing 相对课程式 teacher forcing 的结果为：

| Rollout | 流场 MAE | 流场变化 | 后柱受力 MAE | 受力变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00108605 | -6.41% | 0.0342814 | +2.95% |
| 10 | 0.00811312 | -10.04% | 0.0428153 | -5.85% |
| 50 | 0.0202451 | -15.03% | 0.110876 | -21.43% |
| 100 | 0.0307753 | -19.40% | 0.176541 | -18.72% |

随后执行动作条件审计：

```bash
bash scripts/run_tandem_action_sensitivity.sh
```

脚本使用同一个冻结 checkpoint 和 test 轨迹，仅替换模型收到的 `omega`：真实动作、全零、反号和固定种子时间打乱。主要结果位于：

```text
artifacts/tandem_fno_rollout_no_tf_v1/action_sensitivity/summary.json
artifacts/tandem_fno_rollout_no_tf_v1/action_sensitivity/summary.md
```

在 100 步测试中，转速置零、反号和打乱使流场 MAE 分别增加 190.52%、386.74% 和 211.85%，使后柱受力 MAE 分别增加 746.57%、1433.23% 和 789.35%。这表明模型的动作通道对预测有实质影响。该结论不等价于闭环控制有效；控制策略仍需回到未参与训练的 OpenFOAM 环境验证。

动作范围分桶使用统一的 5 帧起点间隔，避免长 horizon 样本过少：

```bash
bash scripts/run_tandem_action_error_bins.sh
```

结果保存在 `artifacts/tandem_fno_rollout_no_tf_v1/action_error_bins/`。100 步高幅值分桶 `max |omega|∈[4,5)` 包含 224 个窗口，流场和受力 MAE 相对 `[2,3)` 分桶分别增加 64.61% 和 166.67%。50 步快速动作分桶 `max |domega/dt|∈[4,6)` 包含 59 个窗口，相对 `[0,1)` 分桶分别增加 122.86% 和 239.73%。后续控制器需要对这些区域施加不确定性检查，不能仅依靠动作硬边界。
