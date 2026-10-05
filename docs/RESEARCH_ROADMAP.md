# Research roadmap and prioritized backlog

## 课题定位

面向后圆柱旋转的串联双圆柱主动流动控制，研究控制相关的神经算子代理预测，并通过真实CFD验证兼顾整体减阻与升力波动约束的反馈控制。实时性能、跨Re稳健性、CFD样本效率均需测量后才可成为结论。

当前采用有限场景先验证、再扩展的路线；不是同时开展所有可能方向。

## 四层架构、七类工作

CFD真值 → 代理预测 → 控制决策 → 在线CFD反馈。分开管理数据质量、字段精度、递推稳定性、分布外泛化、气动力预测、控制设计和闭环验证；分开并不意味着七路同时训练。

| 阶段 | 任务 | 退出条件 |
|---|---|---|
| P0 baseline reconstruction | 重建代码/数据/模型/协议/结果证据 | 每项主要结论能追溯artifact；无伪造缺失指标 |
| P1 data-space analysis | Re、alpha、L/D、相位和动作时间历程覆盖 | 区分已见参数、未见相位、未见动作历程、参数外推；同初态配对成立 |
| P2 surrogate accuracy | u/v/p及前后受力固定协议基线 | 明确各动作/相位/时间范围误差，不以单个平均数掩盖失败 |
| P3 rollout stability | H1/10/50/100时序误差、相位/幅值、必要物理诊断 | 有限输出与准确预测分别判定；不把H100外推为800步稳定 |
| P4 control-oriented prediction | FC-P011两臂均formal失败，FC-P012也未触发预声明梯度解释；FC-P013独立受力FNO仅完成单窗无更新资源探针 | 完成trainer保存/fresh-reload、六窗诊断和双模型契约审查后才可另批正式训练；原gate不变，PPO继续阻断 |
| P5 interpretable control baseline | 条件性MPC/有限动作搜索对照 | 代理准入后才执行；相同物理目标、动作约束和真实CFD对照；不强制推翻PPO |
| P6 RL integration | 绑定合格新FNO重新训练PPO | 环境/模型/归一化/策略SHA对应，训练及独立评估无数值异常 |
| P7 real-CFD closed loop | 冻结策略、配对OpenFOAM反馈验证 | 满足原物理指标；再逐步增加未参与开发的测试与多seed证据 |

P5不是“已有PPO必须作废”的依赖；P4不通过时MPC和surrogate PPO均不能用代理收益声称物理成功。

## 优先级队列（不是全部开跑）

| ID/优先级 | 假设 | 实验/指标 | 决策和下一步 |
|---|---|---|---|
| FC-P001 / COMPLETE — 干预未获支持 | 配对时间统计监督改善动作引起的受力变化 | λ0/10同协议后评估已完成；端点通过，但两支力窗口均仅2/6零动作分支通过，旋转分支失败 | 保留FC-E006/007负结果；不启动代理PPO，不放松验收要求 |
| FC-P002 / COMPLETE — 诊断范围内 | 固定Re100下，失败与动作历程、相位和预测时域有关 | 覆盖审计和失败图已完成；参数幅度覆盖不等于动作历程覆盖；同物理时刻H1/H100对齐进一步表明非零动作的一步升力误差已明显存在 | 不把全部误差归因递推累积或网络容量；跨Re仍未评估 |
| FC-P003 / COMPLETE — 干预未获支持 | 将16次配对监督均匀分布到epoch，减少早期监督被后续更新冲淡的可能 | 两epoch和完整后评估已完成；validation10与dynamic6组件通过，但force-window仍仅2/6 zero分支通过，四个旋转分支Cl′ RMS误差与旧λ10基本不变，development FAIL | 保留FC-E011负结果；不启动代理PPO；其后FC-P003B/C单因素实验也已完成并分别记账 |
| FC-P003B / COMPLETE — 干预未获支持 | 动态动作/零动作同初态配对监督，比常值动作配对更适合控制相关预测 | 两epoch、SHA回传和完整同协议后评估已完成。端点组件通过，但force-window仅2/6 zero分支通过；四个旋转分支rear-Cl′ RMS误差比FC-P003仅降2.53--3.16%，仍约为限值5.9倍。true-state H1旋转分支rear-Cl MAE仍为0.156--0.192 | 保留FC-E012负结果；不启动代理PPO，不放松门槛 |
| FC-P003C / COMPLETE — 干预未获支持 | 将配对窗口统计项替换为true-state每端点action-minus-zero四力监督，可直接改善一步动作-受力映射 | 固定两epoch训练和完整同协议后评估已完成；endpoint组件通过，但force-window仍仅2/6 zero分支通过，四个旋转分支Cl′ RMS误差相对FC-P003B均值约恶化1.37%，development FAIL | 保留FC-E013负结果；PPO继续阻断；不以技术probe或endpoint PASS替代完整准入 |
| D015 / COMPLETE — FC-P008与FC-P009 DEVELOPMENT FAIL | FC-P008原formal联合window仅1/6；FC-P009固定50/50共享头正式候选恢复端点Cd/动作与多数mean-Cl，但force-window仍仅两个zero分支通过（2/6），均不准入PPO | FC-P009 receipt `ac5c0dd0…e231c`；四旋转rear-Cl′ RMS误差`0.06750/0.12501/0.07045/0.07987`，固定上限约`0.0294`。field与C不变，frozen/PPO均未访问。后续P010尾窗监督在free-AR小幅改善但H1多数相位退化，且五次fit均未在200次预算内达到梯度容差 | 拒绝继续固定线性头扫参；保留原门槛，不把endpoint或train-cache改善当闭环准入 |
| FC-P011 / COMPLETE — BOTH DEVELOPMENT FAIL | 在相同继续训练和受力loss下，额外解冻现有decoder最后hidden linear层比只调rear-Cl row更能改善控制相关表征 | A/B均完成1368更新及同协议formal；两臂force-window仅2/6 zero通过。A旋转RMS误差`0.0703/0.1222/0.0681/0.0832`，B为`0.0626/0.1192/0.0752/0.0762`；B validation delta-Cd `0.024318>0.023` | 拒绝两臂并禁止PPO。只准备no-optimizer/no-save的六窗梯度分解，区分尺度与方向冲突；该诊断不构成新gate或训练批准 |
| FC-P012 / COMPLETE — 预声明梯度解释未获支持 | FC-P011B的权衡伴随field与weighted-force的强尺度失衡或反向梯度冲突 | P009与P011B各6个train窗口、共12行；五个nonzero窗中两模型ratio>10与cos<-.2均为0/5；zero窗另列，参数前后不变 | 不据此扫loss权重或改clip；representation-capacity后续须另批并继续接受原formal裁决 |
| FC-P013 / ENGINEERING PROBE PASS — TRAINING NOT RUN | 将受力预测放入独立官方FNO可避免与冻结flow表征共享训练，同时保持原flow递推 | 单个真实H100双域forward/backward在4.877秒完成，28个可训练tensor梯度有限，双模型参数未变，峰值reserved 4.326 GB、guard最低内存107.673 GiB | 只证明工程可行；正式训练须闭合保存/fresh-reload、六窗诊断及dual manifest契约并另获批准，之后仍由原formal裁决 |
| FC-P004 / GATED | 合格代理上的短时域显式动作规划可提供解释性控制对照 | 可选MPC候选序列受相同动作约束，在真实CFD配对评价；预先规定时域和计算预算 | 用于区分代理/控制器问题，不因为MPC可解释就视为安全可靠 |
| FC-P005 / GATED | 合格新代理支持学到有物理收益的PPO策略 | 新policy仅在准确绑定的FNO中训练，随后真实CFD配对；指标沿用原标准 | 通过后才能主张surrogate-assisted闭环，而不是CFD-only成果 |
| FC-P006 / LATER | 收益在未参与开发的样本/随机种子上可重复 | 固定候选后开展冻结集/新独立样本、多seed，报告区间及失效 | 扩大或收缩稳健性结论；反复开发用的validation不当最终test |
| FC-P007 / DEFERRED | 校准不确定性可选择更有用CFD数据 | 先校准误差/不确定性，再与随机采样同CFD预算比较；参数空间扩展单独预声明 | 有可测样本效率收益才采用active learning；不能只展示循环图 |

## 实验批准单

每次批准填：ID；待检验假设及反证条件；父模型；代码与resolved config SHA；数据/归一化/划分；唯一改变及对照；指标公式/单位/窗口/聚合；独立评价负责人；训练/评估预算；数值/资源停止条件；完成后的接受、拒绝或不确定判定；下一步。单次实验停止条件与项目目标完成是两回事。

## 最高风险与应对

1. 受力来自学习输出，不是壁面物理积分：字段漂亮不能证明受力正确，单独检查action-minus-zero响应。
2. 代理可能被控制器利用其误差：surrogate reward不是CFD收益，真实反馈验证不可省略。
3. 短窗端点和长窗统计不同：固定两套协议，不能交换数值或把短窗频谱当长期收敛。
4. 多次查看开发验证会适应验证集：冻结集不参与训练/选择，独立终验必须后置。
5. ROI是开放子域，当前一帧条件可能不充分：是待测假设，不能伪造ROI进出口边界条件。
6. 初始相位相近不代表独立物理工况；波动减小不等于已计算结构疲劳或涡激振动。
7. omega²/变化率惩罚不是电机电能；未测真实执行功耗不得声称净节能。
8. 训练结束/文件存在不代表完整评估：以独立receipt、完整性复核和任务清单保持持续推进。
9. 旧复现文档与后续原文核查存在信息确定性差异：保留历史但标明适用阶段，未经原文核实的奖励/传感器/动作参数不成为当前“论文复现”的依据。

## 持久状态职责

`PROJECT_STATE.md`只描述当前已核事实、阻碍与下一优先级；`DECISIONS.md`回答为何取舍；`EXPERIMENTS.md`解释实验与协议；`experiments/results.csv`承载机器可读指标与证据。这些文件是项目索引，不搬迁现有cfd/src/conf/data目录，不复制大型数据。
