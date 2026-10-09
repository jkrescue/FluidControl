# 闭环论文验证缺项与基线公平性审计

## 本次预先计划（不是历史实验预注册）

Owner: closed_loop_readiness_review；Lead于本轮明确批准一次只读CPU证据审计。先前查阅论文清单、源码和历史报告属于准备，不追认为历史预注册。本文件先落盘，再执行下述独立receipt生成。

- 资源：CPU1、MemoryMax=2GiB、MemorySwapMax=0、CUDA隐藏，最长600秒；无模型加载、训练、求解器或大数据复制。
- 假设：旧周期/恒转速与当前B在约束、起点或窗口上不等价，现有结果不足以支持公平优越结论。
- 比较维度：工况、初态来源、统计窗、动作幅值与变化率、网格/时间步验证范围、原三项门限、独立性。
- 输入：历史periodic result及action audit；constant_p100_grid_comparison；周期generator源码；B continuation approval/result/独审report与receipt；independent restart audit；PAPER_REPRODUCTION与PAPER_VALUE_AUDIT。脚本记录各实际SHA与字节数。
- E114既有独审已覆盖全部800单filter、799反馈、1600日志时钟、有限raw forces及动作变化率；本次只核报告/receipt身份，不重复全量动作/CFD审计。
- 输出仅是已保存证据的公平性判断与缺项清单，不产生新物理结果、不改变B默认或任何门限，不执行新的CFD验证。

## 实际结果

实际于2026-10-07T11:05:33Z完成。unit `fluid-control-paper-baseline-fairness-audit-20261007.service`，invocation `4beeb5c43c1c42bc800dd7dc73bb780a`，PID0/Resultsuccess/ExecMainStatus0；实际资源按上述限制。12项输入在读取前后SHA一致；这是元数据/已保存审计证据一致性检查，不是重新解析原始CFD或独立求解。

- receipt原执行输出为`artifacts/p064_paper_baseline_fairness_audit_20261007/receipt.json`；仓库内原字节归档：[receipt](report_20261007/evidence/paper_baseline_fairness_receipt.json)，SHA `7f3bbbd77310913288f9db0052cb9b9a93ddf5316e57219b131fde78560b6c5b`。
- 实际执行脚本为`/tmp/audit_paper_baseline_fairness.py`；随后原字节归档至[scripts/audit_paper_baseline_fairness_20261007.py](../scripts/audit_paper_baseline_fairness_20261007.py)，SHA均为`309cb8acc29f62e3fa33c6dc9127a5e593aec5e36f9066c36849cd7ae8d1b8c7`。归档未重跑，不将归档路径冒充原执行路径。
- 预先计划原字节保存在同artifact目录`pre_execution_plan.md`，另原字节归档为[预先计划](report_20261007/evidence/paper_baseline_fairness_pre_execution_plan.md)，SHA `e083cf28637d6032c8f00a30e86b938ef9f7968e771da5a0177110766f60e035`；当前文档随后补终态，不将历史实验追认成预注册。receipt中pre_execution_plan输入SHA指执行前版本，而非本终态报告。

## 公平性结论

| 证据 | 已知范围 | 尚不能声称 |
|---|---|---|
| 周期P10/P20 | 已完成同历史zero配对；t80起步、统计120–160；峰值ω=1，最大速率分别1.870999/0.944213 | 不是当前B的幅值.75、速率≤1、相同起点/统计窗公平对照；P20速率虽不超B限制，幅值仍不同 |
| 周期物理结果 | P10减阻−0.58607%、RMS比1.10322；P20减阻0.91860%、RMS比1.13798；两者原AND均FAIL | 不能由历史失败推断B胜过约束匹配、冻结选择的最佳周期策略 |
| 恒转速网格对 | `control_small_p100`与`control_grid_p100_medium`，窗口80–160；后Cd均值相差0.15624%、后去均值Cl RMS相差1.06153% | 验证的是恒ω=+1，不是当前B反馈策略；不外推B的4%效益数值不确定度 |
| 其他数值历史 | PAPER_REPRODUCTION记录zero及高转速dt/网格检查 | 本次未复算其raw；不能以该文档替代最终策略中网格/半dt闭环验证 |
| E114 | 同冻结B策略续跑328–408及与E109连接；800新反馈、所有原门通过，既有独审覆盖完整单filter/反馈/solver时钟 | 不是新Re、几何或独立相位；1600周期经历已验证restart，不是单进程无中断 |
| seed/phase | canonical多seed、既定phase上的历史结果存在；8个已选phase均已打开 | 不是预先封存的独立测试集，也不能将时间步/20D/U块当独立样本 |

E114原receipt已核800cycles、799feedback links、1600solver logs及max|Δω|=`0.10000000000000003`；原报告还核所有有限raw force、.005时钟及单次物理filter。故不再为相同事项新建重复raw审计。原seed结果须逐项保留，不能选择性只报告最好seed。

## 下轮精确定义（仅建议，未执行授权）

1. **最终控制器数值验证**：固定B policy `5c05699e…`及Vec `1d250051…`、canonical/filter、Re100/L/D5。先在同粗网格将dt .005→.0025，反馈间隔仍.1；zero与B配对、预先固定初态生成方法及过渡/统计窗，不复用旧zero作新离散对照。随后独立中网格；网格变化必须明确初始化/映射和再稳定程序，不将粗网格state字节直接视为中网格同初态。报告配对三指标随离散变化，不承诺门限必过。
2. **公平开环基线**：在开发集选择后冻结一个恒转速和一个周期波形，再与冻结B共享同一预声明初态、网格、dt、时长、过渡与统计窗。共同约束|ω|≤.75、|Δω|≤.1/反馈周期；分别披露动作RMS及速率，若宣称等执行器代价则需在测试前固定匹配规则，不能用测试B动作事后调幅/频率。原2%/1.05/10%不变。
3. **独立相位重复**：冻结策略后生成并预登记新的评估初态，禁止看结果换相位或挑策略；旧b03/b07虽曾标frozen、现已打开。restart124至多是未记录使用的同极限环邻近状态，不是现成封存holdout。新相位仍不是新物理工况；统计按轨迹/相位而非时间点重采样。
4. **物理解释边界**：补扭矩和机械功后才讨论净节能；参考论文约98%是不同目标/动作域下文献结果，当前约4%总减阻不可冒充其完整复现。完整FNO预测准入与有效MPC仍未完成。

剩余收尾窗口不启动旧恒转速网格脚本：其需300GiB并拒绝现存case，且会重复历史恒转速而非验证B。没有本轮已冻结、无需改配置的公平新CFD入口；本次只补齐证据界限，未增加新物理PASS。

## 关键来源身份

所有完整路径/字节数/SHA见receipt；关键项如下。

- 周期result：`artifacts/tandem_cylinders/periodic_rotation_benchmark_result_20261003.json`，`46b573daf70399c514c1422579c255d6ba32b8bdcee2936ddfe91e8de39a0d7f`。
- 周期action audit：`6819521baa978b2fb5616b445be40545c1aa20dfe0ba3791bec03f0a12b488c7`；generator：`883cf6c0b4b0073dfa77b17f475a0383eb7a6e45d516864330e4dad7f2469f59`。
- 恒转速网格result：`artifacts/tandem_cylinders/constant_p100_grid_comparison.json`，`d98b8d96f478ebeeffe6f6bb415b5641bef322d48da502f2bf7f27bbf5bfb987`。
- [E114终态独审](P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md)：`09e89416e4dc59494fdab362f7316b213762850c742a752a91a3a9b3856fd2e5`；result `752b92d1063e51a8fb6a45ea539b173c3c5ffbd24c6a83255e0fa649f392063e`；原receipt `a26368b60007b53a422351f1d086fcda095cd823a98ded331e513c324de07af1`。
- [独立起点审计](P064_INDEPENDENT_CONTROL_RESTART_AUDIT_20261007.md)：`59596b685b6603b41818fa2416b9fed347443a7c78c68d2de3010f9ad0b465e4`。
- [PAPER_REPRODUCTION](PAPER_REPRODUCTION.md)：`1d0a47cd34c1a18ee874ffcea16a3b2c7c411aea866d953028a826c4640c1a38`；[PAPER_VALUE_AUDIT](PAPER_VALUE_AUDIT.md)：`d994ebe3e0ae1b6759e0915eb5d55e5326aafbb25b575c7feda896cbb1e11d5b`。这些是历史文档，当前进展以最新主报告为准。
