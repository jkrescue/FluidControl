# FNO ROI 与气动力可观测性只读说明（D015 辅证）

日期：2026-10-05  
范围：现有 Re=100、D=1、U=1、L/D=5 串列双圆柱数据与代码；未运行推理、训练或新 CFD。  
代码基线：`7fb060d397da73473639350e3864e0cad3b56e71`。本说明绑定下列逐文件 SHA，因此不假设工作树其余文件状态。

## 核心结论

现有 FNO ROI **完整包住两个圆柱及其近壁外侧流体和尾迹**，但 HDF 中圆柱内部被 mask，且不保存 OpenFOAM wall-face 值、patch 拓扑、壁面法向或显式壁面剪切梯度。因此，该 ROI 可用于学习流场与气动力统计映射，也可能支持近似重构；但仅凭当前 HDF 不能直接、严格复现 OpenFOAM 在圆柱 patch 上的 Cd/Cl 壁面积分。

气动力真值来自 OpenFOAM `forceCoeffs` 对 `frontCylinder`/`rearCylinder` patch 的计算。Curator 将其 `coefficient.dat` 中 Cd、Cl 对齐到场时间。当前 FNO 并不对预测的压力和黏性应力做表面积分：它另输出四个空间通道，并在全部有效网格上取平均，得到 `[front_cd, front_cl, rear_cd, rear_cl]`。因此，“总体场误差尚可”与“rear-Cl 误差较大”并不矛盾；二者不是同一个物理或数值约束。

## 实证范围与分辨率

- CFD 圆柱中心为前柱 `(10, 7.5)`、后柱 `(15, 7.5)`，半径 `0.5`。FNO 规则网格为 `x=[8,25]`、`y=[4,11]`、`256×128`，所以两柱及后柱下游 10D 均位于 ROI 内。
- 对真实 train HDF `matched_start_acquisition_train_b00_zero.h5` 检查：有效点 `32,340`，无效点 `428`；每个圆柱内部各有 `214` 个无效点，最近有效点到柱心半径为 `0.501896`。mask 在所查首末帧相同。
- 每个圆柱 `r≤0.6D` 内只有 `92` 个有效流体点，占全部有效点 `0.284%`；两柱合计约 `0.57%`。所以按全域有效点平均的 field loss/relative-L2 可能被 bulk/wake 主导，不能单独证明近壁不对称压力与速度梯度准确。
- 输入为当前归一化 `(u,v,gauge_pressure)`、mask、`omega_now` 与 `omega_next`。pressure 在每帧、全部有效 ROI 上减去空间均值。

pressure 去均值本身不能被认定为力误差原因：对理想闭合表面，均匀压力常数的净合力应抵消。不过，当前 HDF 缺少严格壁面积分所需的 wall pressure、法向和黏性剪切信息；这与 pressure gauge 是两个不同问题。

## 对 rear-Cl 的限制与待检验解释

以下仅为可检验解释，不是因果结论：

1. 全域场指标可能稀释后柱窄近壁区域的误差，而 rear-Cl 对上下表面压力/剪切的不对称尤其敏感。
2. 后柱处于前柱尾迹中，作用相位和转速改变可能放大局部非对称误差；需要按相位、动作与近壁距离分层验证，不能由几何直接推出。
3. 四个 force 通道是学习 readout，并未被约束为预测场的压力/黏性应力积分；场与力可出现不同误差趋势。
4. ROI 上游边界距前柱仅 2D，且模型使用单帧状态与相邻动作，不显式提供更长历史；这可能影响动态可观测性，但当前证据不足以定因。

不新增推理即可在后续已有输出上检验：

- 将 H1 的 `u/v/gauge-p` 误差按后柱壁面距离分带，并与 bulk/wake 及 rear-Cl 误差关联；
- 按 action、phase、正负旋转分层，避免只看宏平均；
- 使用既有 `diagnose_fno_force_window.py` 同时报告 raw pressure 与逐帧 demeaned pressure 误差；
- 若原始 `forceCoeffs` 分量可用，比较压力/黏性贡献与总 rear-Cl 误差，但不得把 FNO 总力头伪装成分量预测。

这些检查只用于解释 D015 或后续候选的误差，不授权改模型、加入新 physics loss、访问 frozen split 或启动新实验。

## 证据与 SHA-256

- `data/curated/tandem_cylinders_matched_start_full40_dev30_v1/train/matched_start_acquisition_train_b00_zero.h5`: `243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01`
- `scripts/curate_tandem_cfd.py`: `d182c5c8093ca6a303aca34d3b4c0b4d40c55ed598f18b2d97d356f057a0926e`
- `scripts/train_tandem_fno.py`: `9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a`
- `src/fluid_control/tandem_datapipe.py`: `c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae`
- `scripts/diagnose_fno_force_window.py`: `4ef0a878ac6f3ab3a8e0a957b7b16882731b2aff8093d0efa46cb4e29a954df8`
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/system/blockMeshDict`: `a3faf63c69548315b5346ec4d0969ea52fefe3b18e8aad811a972a10d9408ee6`
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/system/controlDict`: `8c8bbd6413517d497387067562316c9ac2996a56bd7ebb67c6965939611d29eb`

