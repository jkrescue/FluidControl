# FC-P064：受控闭环数据对气动力读出修复的固定预算 A/B 方案

状态：仅 CPU 工程准备；未批准 GPU、训练、策略重训或新 CFD。

## 单一问题与假设

已保存的投影 PPO 闭环轨迹显示：在真实当前流场和已执行动作上，K1 的 H1 力预测仍有明显误差；P028 只更新 flow、冻结 aero 时 H1 力与 K1 完全相同，不能修复这一项；P029 通过冻结 aero 的损失反传改善过原训练分布 H10，但没有解决正式长时门，也没有证明受控分布 H1 力已修复。

本实验只检验一个因素：在相同 K1 终态、相同损失和相同更新预算下，用固定比例的已保存 b00 投影策略闭环窗口替换原 44 条训练轨迹中的训练窗口，是否改善受控分布气动力读出。它不检验新架构，不更新 flow，也不自动授权新策略或 CFD。

## 精确父模型与不变训练合同

- 父 manifest：`artifacts/fcp026_history_training_k1_20261005/candidate/dual_model_manifest.json`，SHA-256 `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`。
- 冻结 flow：P009 model/state `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` / `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`。
- 两臂 aero 都从 K1 终态 model/state `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5` / `ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3` 开始。
- 两臂均新建 AdamW，不续接父 optimizer moment：LR `1.5625e-7`，betas `(0.9,0.999)`，eps `1e-8`，weight decay `1e-4`，8 窗原始梯度平均后 clip norm 1。
- flow 参数、权重字节、模式均不变；aero 保持 P026 的精确训练范围：28 个参数可训练、两个 lift-bias 参数冻结，不把“aero 更新”误写成所有 aero 参数都更新。
- 架构、K1 history 输入、H100/chunk10、四力归一化、动作打包、原 train-only normalization、精度和 objective 均与 P026 K1 相同：每域 `0.5 × 四通道等权 normalized MSE + 0.5 × rear-Cl normalized MSE`，H1 与冻结-flow free-AR 两域等权。不得增加 field loss 或重标损失。
- 继续使用官方 PhysicsNeMo `HDF5Reader`、DataLoader 和 checkpoint save/reload API；`TandemRolloutDataset` 与本次固定调度器均为项目 adapter，前者调用官方 Reader，后者只编译固定样本表并路由已有 dataset。

## 固定预算和机械序列

P026 K1 的实际 resource watch 跨度约 5357 秒，1368 窗/171 更新约 89.3 分钟；完整两臂约 3 小时。因此第一轮预声明为有限预算数据效应试验：

- 每臂恰好 256 个 H100 窗、32 次 optimizer update、每次 8 窗；跑完固定终态，不早停、不选最好 checkpoint、不临时追加更新。
- A：实际 P026 K1 官方 shuffle 顺序的前 256 项，完整 sampler SHA `177ebd...f9f`，前缀 SHA `06c922...6691`；该前缀机械产生 base/train8/train16 = 140/64/52 窗，不做重平衡。
- B：与 A 相同的 256 个消费位置、相同 update 边界；每个 8 窗 update 固定把位置 0 和 4 替换为 b00 窗，合计 64/256 = 25%。被替换位置的 A 原始 global index 仍写入 B 元数据，便于逐位置核对。
- b00 是 801 帧，H100 合法 start 为 0..700。64 个 start 固定为 `floor(i×700/63), i=0..63`，包含 0、700，间隔仅 11 或 12；不按误差、力或控制收益选窗。
- 每个消费窗权重均为 1/256，每个 update 内为 1/8；B 的 b00 总权重恒为 25%，不是在 1368 窗之外追加数据。
- 被替换的 64 个 A 位置来自 base/train8/train16 = 31/19/14，因此 B 的原数据为 109/45/38，加 64 个 b00；这些数目由固定 sampler 前缀和每四位替换规则唯一决定，不是按结果选择。
- 估计每臂约 16.7 分钟仅由 P026 线性比例得到，不是运行时保证。Spark 是 UMA：正式 launcher 采用已审口径，启动前 `MemAvailable ≥ 50 GiB`，运行期操作前 `MemAvailable ≥ 22 GiB`，并始终保留 20 GiB；记录 MemFree 仅作观测，不把 MemFree 或 CUDA free 当成缓存门，不执行 `drop_caches`。allocator fraction、cgroup 和 deadline 另审。

## 数据身份和泄漏边界

- recovery 准备的 b00 全 801 帧转换必须与原 CFD、动作、时间、mask、四力、train-only normalization 和 H100 start 表逐项绑定后才能进入 B。单个转换 HDF 所在 artifact 目录不能直接冒充 dataset root：执行准备须创建独占、只读的 dataset 视图，`train/` 内仅以同文件系统 hard link（若受控权限允许）或审阅过的只读引用暴露该 HDF，并复制原 normalization 的精确字节、写入绑定转换 receipt/HDF SHA 的专用 manifest；不得复制大型数组或改写源 artifact。
- b00 一旦用于 B，即明确属于训练/开发数据，不能再称作 test 或独立验证；b01/b03 已被开发流程打开，也不是新鲜最终测试。
- b00 只能报告 train-fit，不能单独作为数据假设支持。主描述比较固定使用已打开的 b01/b03 两个 development phase，在完全相同精度、起点和 H1–H5 协议下比较 K1/A/B；逐 phase 与两 phase pooled 都报告，不得用它选 checkpoint、追加 epoch 或调比例。
- 每个有效终态都应按另行审批跑相同的原正式协议；不能因有限预算 A/B 阴性结果就否定 data-coverage 假设，也不能因训练损失或受控 H1 改善直接宣称可闭环。

## 结果解释与后续边界

描述性支持条件预声明为：在 b01+b03 pooled development 上，B 相对 A 的 H1 rear-Cl MAE 与 total-Cd MAE 均严格下降；同时逐 phase 报告相同指标、H2–H5、固定起点符号和 persistence，不设置事后百分比门。b00 仅作为 train-fit 报告。无论支持、混合或不支持，都不改变原正式科学门。

现有成功投影策略及其真实 CFD 结果保持不变。任何采用的新 surrogate 都必须另行训练兼容策略，并在独立审批下重新做真实 CFD；本方案不批准策略训练、新 CFD 或模型训练执行。
