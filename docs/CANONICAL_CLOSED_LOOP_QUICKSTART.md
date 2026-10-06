# 已验证 canonical 基本闭环：最短复现入口

本入口默认复现 **E085 b01**：冻结 E082 策略/VecNormalize，真实 OpenFOAM 130→210，共800个反馈周期，每周期0.1 D/U；先20 D/U过渡、主窗(150,210]及原六个统计窗。这是已打开开发相位的工程复现，不是新的独立泛化试验。

实际链：官方 Curator/HDF5Reader 数据 → 官方 PhysicsNeMo FNO 代理 → HydroGym/SB3 PPO 训练 → CPU canonical观测、一次策略预测、物理方向恢复、一次安全过滤 → OpenFOAM推进 → 真实观测反馈。部署没有在线 FNO/MPC；它们不是基本闭环演示的必需新增项。此入口不重训模型或策略。

## 1. 默认只预检（不创建输出、不启动CFD）

在保存全部原始产物的 Spark 仓库运行；Git单独克隆不包含大数据、策略、镜像或环境：

```sh
cd /workspace/fluid_control
python3 scripts/reproduce_canonical_closed_loop.py
```

核验原批准、不可变 driver、source/input/runtime/restart SHA、已固定训练结果绑定、镜像、MemAvailable≥50GiB、磁盘≥20GiB。错误即停止；没有自动补数据、修环境或重试。

## 2. 仅在另行批准一次新复现后执行

审批人从 `docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json` 创建新的批准JSON：只能改变 `output`（不存在的 artifacts 子目录）、`lead_statement`、可选 `interpretation`，并新增：

```json
{"reproduction":{"base_approval_sha256":"ee010bbe1932e0b48b2f2e90a1b9dd77d463f86dad6367e0c0dd49a46f0b45f1","unit":"fluid-control-canonical-reproduce-demo1.service","execution_authorized":true}}
```

所有科学字段、策略、源、资源合同必须与原批准完全一致。原批准只授权历史一次运行，不可直接再次执行。

准备阶段将新增 `reproduction.execution_authorized` 设为 `false`，允许只读预检、拒绝执行；只有另行明确批准后才设为 `true` 并重新计算批准SHA。顶层授权字段保留历史driver兼容值，不代表新增运行已获授权。

```sh
python3 scripts/reproduce_canonical_closed_loop.py --approval docs/NEW_APPROVAL.json --approval-sha256 NEW_SHA --unit fluid-control-canonical-reproduce-demo1.service
# 先审上述只读预检与打印的完整argv，再显式执行同一批准：
python3 scripts/reproduce_canonical_closed_loop.py --approval docs/NEW_APPROVAL.json --approval-sha256 NEW_SHA --unit fluid-control-canonical-reproduce-demo1.service --execute
```

禁止复用已有 unit 或输出；不自动 restart。原 driver 保留8GiB/noSwap控制进程、两个8GiB/noSwap求解器、CPU资源、3600s内部/3750s外部超时和精确owned-container cleanup。运行MemAvailable22GiB阈值留20GiB物理余量；不使用GPU。历史约18–19分钟，不是运行时保证。

## 3. 验收与显示

launcher返回真实 InvocationID/MainPID；这只是启动，不是训练或物理成功。新输出的 `progress.json` 给800周期进度、真实观测/动作；`result.json` 与原始forces/solver日志才是终态证据。dashboard须绑定这次新unit/invocation/output，不能把旧卡冒充新运行。此三文件交付未修改dashboard，也未启动新运行。

基本演示验收：入口预检通过、800次真实反馈完成、真实U/p图与请求/施加ω、配对Cd及后柱Cl曲线可追溯、资源/cleanup/原六窗统计核验。主物理阈值仍为减阻≥2%、升力波动RMS比≤1.05、均值偏置≤10%；早期窗口单列，不改为15%/20%。已有E085六窗通过；E083/E086两指定seed主窗通过但早期失败保留。

整体研究目标仍未完成：完整代理预测精度FAIL、控制相关预测误差和有限泛化证据不能由基本演示替代。C50/D25/H25负结果及旧非canonical第二seed失败均保留。真实流场瞬时图不能计算平均减阻，也不能称FNO预测；未核验转矩功率换算，动作平方成本不是净节能。

证据：[E085终态](P064_B_SYMMETRY_CANONICAL_B01_CFD_TERMINAL_REVIEW_20261007.md)、[E082训练](P064_B_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md)、[完整预测评估](P064_B_FORMAL_TERMINAL_REVIEW_20261006.md)、[历史分阶段指南](CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md)。
