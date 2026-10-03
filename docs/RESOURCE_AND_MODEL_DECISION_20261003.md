# 资源利用与模型选择：2026-10-03 决策记录

研究目标保持不变：Re=100、L/D=5 串列双圆柱，通过后柱旋转降低前后柱总平均阻力，同时控制后柱升力波动和平均侧向升力；最终在真实 CFD 中验证在线闭环。PolyU Tang 团队 2024 年的后柱自旋 PPO 研究以减轻下游圆柱升力为主要目标，本项目增加了总阻力与平均升力偏置两项约束，不能把两者的数值成果直接等同。

## 已测资源，而非猜测

在 Spark GB10 上使用既有真实 H20 训练数据和官方 PhysicsNeMo FNO 2.2.2 作单步前后向资源试验（无优化器更新、无新 checkpoint、无验证/测试集接触）。宽度 48、32 模态、batch 4 为约 55.6 transitions/s、11.75 GiB 峰值；batch 8 约 60.0 transitions/s、23.25 GiB 峰值，吞吐仅提高约 7.8%；宽度 64、batch 4 降为约 42.2 transitions/s；48 模态、宽度 48、batch 4 约 46.9 transitions/s、13.44 GiB 峰值，参数由约 4722 万增至 1.062 亿。以上只说明可运行和吞吐，不说明预测更准。原 H100 终点总阻力误差约 16%–18%，未达预定 10% 验收线，尤其不能可靠识别约 2% 的真实阻力收益。

## 并行优先级

1. 首先用 Worker CPU 并行生成预声明的中间转速、不同起始相位真实 OpenFOAM 数据；Spark CPU 同时做已验收九案的 VTK 和 PhysicsNeMo Curator 整理。只要 Worker 开始剩余内存至少 64 GiB、运行中至少 40 GiB、Spark 磁盘至少 250 GiB，最多 4 个 Worker 求解案并发。原始数据统一回 Spark，Worker 只作临时计算节点。由于九案已显示 ±0.75 固定旋转存在很大平均升力偏置，增加 ±0.375 比盲目扩 FNO 更直接地填补控制动作覆盖缺口。
2. 九案仅为流程验收；后续完整 40 案按预声明的相位分组。训练 20、验证 10、冻结测试 10；冻结测试不得用于模型或奖励选择。同一基准周期的不同相位不等于独立物理工况泛化，应在论文中明确限制。
3. 在训练/验证两组具有足够动作覆盖后，再用同一数据划分、公平训练预算比较官方 FNO 的 batch 4 与 8、32 与 48 模态。主要选型指标是多步总阻力和后柱升力误差、动作差分误差、rollout 稳定性与 CFD 重验收益，不是 GPU 占用率。保持至少 20 GiB 可用统一内存。
4. FNO3D 在这里意味着对二维空间加时间块做时空算子，而不是凭空获得三维物理流场；用于在线 PPO 还必须保证因果性、可变转速输入和每步反馈接口。现阶段不把它作为可直接替换的基础模型。局部卷积混合架构虽有研究价值，但会突破当前“只用 PhysicsNeMo 官方模型/API”的实现原则；先不引入自创网络。若标准 FNO 在足够真实数据上仍失败，再预注册架构消融。
5. HydroGym/PPO 只能在代理模型通过独立多步误差、动作排序和 CFD 复验后用于策略训练；需要定期真实 CFD 校正，不能只在代理上宣称闭环成功。完整 40 案全是给定转速分支，仍不能覆盖闭环策略频繁改变转速时的状态分布；后续必须另外预声明有时间变化动作的 CFD 轨迹，并评估动作切换瞬态。主动学习可在训练/验证误差指出高不确定动作或相位时补充 CFD，但不能动冻结测试定义。

相关原始研究及官方文档：

- [PolyU：2024 下游圆柱自旋 PPO 论文档案](https://ira.lib.polyu.edu.hk/handle/10397/107766)
- [2024–2025 surrogate + CFD 交替强化学习研究](https://arxiv.org/abs/2408.14232)：报告需定期用 CFD 修正代理误差，而非无限递推。
- [NVIDIA PhysicsNeMo FNO 官方 API](https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/api/models/fnos.html)
- [NVIDIA PhysicsNeMo Active Learning 官方示例](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/active_learning_aero/README.html)：是外部气动案例，不能原样声称适用于本项目或当前固定版本。
- [HydroGym 官方仓库](https://github.com/dynamicslab/hydrogym)：后续用于 CFD 环境中的强化学习与复验，而不是替代真实 CFD 数据。
- [2025 U-Net/FNO 混合模型研究](https://arxiv.org/abs/2504.13126)：仅作为未来消融动机，并非本项目已经验证的架构。

本项目可复算资源数据见 `artifacts/benchmarks/tandem_fno_h20_resource_modes48_20261003/benchmark.json` 和 `docs/FNO_H20_MODES48_RESOURCE_BENCHMARK_20261003.md`。以上文献的效果属于各自几何、雷诺数、控制方式；不能替代本项目 CFD 验证。
