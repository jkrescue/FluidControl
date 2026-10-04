# DECISIONS

本文件于2026-10-04重建。旧决定的记载是事后整理，不伪装成历史预注册。新决定先登记再实验。原始失败与证据不覆盖。

| ID | 决定/状态 | 原因和证据 | 后续检验/改变条件 |
|---|---|---|---|
| D001 | 保持固定Re100、L/D5、后圆柱旋转；accepted | 当前数据和已验证CFD控制均为该场景；不能把用户举例的多U当已有数据 | 当前闭环验证完成后再单独批准跨Re数据设计 |
| D002 | 以整体减阻及升力约束为目标；accepted, reaffirmed | 真实CFD既定>=2%减阻、RMS<=1.05、偏置<=0.10 | 不因候选失败而放松；目标变更需独立记录 |
| D003 | 保留官方二维FNO和官方数据组件；accepted | 当前存在可复现数据/模型/控制接口，未证明结构容量是根因 | 仅在误差分布和受控对照支持时提新架构实验 |
| D004 | 当前优先控制相关幅值/动作响应，不只看field loss；accepted | train16动态端点Cd通过但力窗口失败；加权分支幅值动作方向错误 | 完整后评估验证paired λ0/10是否真正改善 |
| D005 | 拒绝当前λ10 paired-stat干预；rejected by FC-P001 | 同亲本/同数据/同协议下，λ0与λ10均只有2/6 zero分支通过force-window；λ10将四个旋转分支Cl′ RMS误差均值仅降1.68%，个别phase有改善也有退化，仍约为限值的6倍 | 保留两支失败证据；完成FC-P002失败图后，只提交一个Lead批准的FC-P003单因素假设，不启动代理PPO |
| D006 | 不推倒已有CFD-only PPO，不强制从MPC重开项目；accepted | 已有真实反馈减阻约4.2%的有效基线 | 可在代理通过验证后加入有限时域MPC诊断对照；MPC也受代理误差影响，不能绕过精度要求 |
| D007 | 不新增统一加权reward替代既定物理约束；accepted | 权重混合目标可能掩盖升力超限；已有动作与物理验收合同 | MPC/RL比较必须同观测/动作/起点/预算/物理约束，reward改变另做消融 |
| D008 | 跨Re/OOD与主动采样暂列后续，不声称现已具备；deferred | 当前不足是固定Re下动作/时序预测；未知不等于已定位跨Re泛化问题 | 先明确训练参数支持、误差图及不确定性校准，再设计新CFD采样 |
| D009 | 论文价值待实验支持；accepted | CFD-only收益不能归因FNO；b00/b01非独立工况；尚无样本效率/净电能收益证据 | 增加冻结测试、多seed、CFD预算与收敛/稳健性证据后再扩大主张 |
| D010 | 以持久状态和实验台账协调代理；accepted now | 多次流程结束、后处理失败与旧状态导致进展混淆 | 每关键节点更新并独立复核；阶段完成不得改project_goal_complete |
| D011 | 区分港理工相关研究背景、独立实现与严格论文复现；accepted now | `docs/POLYU_ZHAO_2024_SOURCE_AUDIT_20261003.md`记录部分原文参数尚未核实，而旧`docs/PAPER_REPRODUCTION.md`含较具体及较早阶段描述 | 不把旧文档更肯定的说法当证据；建立原文页码/公式对照后才宣称一致。当前整体减阻+升力约束是项目目标，不能冒称论文原奖励 |

## D012 — 补全 canonical 代理验证的数值生成程序

2026-10-05 Asia/Shanghai，accepted before FC-P003/P003B完整后评估。
独立源码和历史审计确认：旧window/dynamic receipt仅有consumer字段合同，
没有历史数值producer。采用2026-10-04已经固定的development误差标准补全，
不降低阈值、不将今日定义追认成历史预注册，也不把共享证据的receipt描述为
独立科学实验。实现须从校验后的原始评估证据重算，而非改名PASS布尔值。
详见`docs/CANONICAL_SURROGATE_PROTOCOL_COMPLETION_20261005.md`。
只授权CPU producer/诊断；PPO及最终真实CFD物理验收要求不变。

## D013 — 先恢复已完成的后评估阶段，技术探针不代替科学实验

2026-10-05 Asia/Shanghai，accepted operational decision。FC-P003B首次后评估在validation10 GPU推理完成后因audit挂载路径合同不一致失败。保留原失败，只允许不可变恢复程序在重算SHA、checkpoint、有限性、计数和数据合同后复用完整推理pair；成对文件缺失、哈希不符或未知故障必须停止，不得为赶进度重跑或改阈值。端点组件PASS不得越过dynamic6、force-window或development gate。

true-state paired-force GPU工程探针只被批准检查固定真实train-only pair上的causal索引、chunked gradient等价性和峰值内存；无optimizer、无权重保存、无validation/frozen数据。v1因原容器身份下mode-600输入不可读而在forward前失败，不对更底层rootless/user-namespace原因作未验证声称。Lead后续明确批准的v2只修复容器UID/挂载可读性并使用新独占输出，数值合同不变；它已通过梯度等价与资源检查，且模型parameters/buffers未变。该技术PASS不是候选代理、PPO准入或闭环成果；正式损失实验仍须等FC-P003B完整结果并单独批准。

## D014 — 拒绝FC-P003B并仅批准FC-P003C工程实现

2026-10-05 Asia/Shanghai。FC-P003B在与FC-P003相同的亲本、regular顺序、λ10、update数、epoch数和评价协议下，只将static16配对监督内容改为dynamic8×2。四个旋转分支rear Cl′ RMS误差较FC-P003均值仅降2.94%，force-window仍2/6、development FAIL，因此该干预不足以支持代理PPO。true-state H1的旋转rear-Cl MAE仍为0.156--0.192，说明失败不能只归因于自回归累积。

Lead以`docs/FC-P003C_APPROVAL.md`批准单因素下一步的工程实现和CPU测试：保持所有数据、顺序、模型、归一化、超参数及验收不变，只将配对损失替换为true-state每端点action-minus-zero四力误差。固定`w=[1,1,4,1]/7`和λ10不意味新旧损失梯度等强；必须报告通道损失/梯度贡献，不得看到validation结果后再改权重。完整GPU训练需另行批准，原准入门槛不变。

## 新决策格式

ID、记录时间、状态、待检验假设、对应实验ID、所依据证据/协议、可选方案、取舍原因、保留的不确定性、撤销/调整条件。只有读取过的产物可作为事实；代理口头报告是待核信息。
