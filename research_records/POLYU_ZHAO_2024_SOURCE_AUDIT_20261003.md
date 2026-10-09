# Zhao et al. (2024) 串列圆柱自旋控制：原始来源核查

核查日期：2026-10-03
对象：F. Zhao, Y. Zhou, F. Ren, H. Tang, Z. Wang, *Mitigating the lift of a circular cylinder in wake flow using deep reinforcement learning guided self-rotation*, **Ocean Engineering 306** (2024), 118138. DOI: [10.1016/j.oceaneng.2024.118138](https://doi.org/10.1016/j.oceaneng.2024.118138)。

## 来源与可访问性结论

- [PolyU Scholars Hub 正式条目](https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/)和[PolyU Institutional Research Archive 条目](https://ira.lib.polyu.edu.hk/handle/10397/107766)确认作者、期刊、卷号、文章号、DOI 和摘要。
- PolyU IRA 元数据说明其保存的是 final accepted manuscript，并给出文件名 `Zhao_Mitigating_Lift_Circular.pdf`；但本次核查时，条目页仍显示 `embargoed access`（Embargo End Date: 2026-08-15），页面未提供可工作的 PDF 下载链接。直接访问官方 bitstream 返回服务器错误。因此，本次没有获得一份**可重复公开下载并逐页核验**的完整稿。
- Elsevier 原论文页面可由检索索引读取摘要/结论片段，但正文访问受限。以下只把官方摘要或出版者原论文页面明确显示的内容记为“已核实”；正文参数绝不由二手论文或项目现有假设补齐。

## 逐项核查

| 项目 | 可由原始/官方来源核实的内容 | 状态与边界 |
|---|---|---|
| Reynolds 数 | 出版者原论文结论片段明确为 **Re = 100**。 | 已核实。 |
| 圆柱布置与间距 | 两圆柱零错列串列；出版者结论片段称无量纲中心距 `L*` 从 **1.1 到 9.0**。官方摘要明确把 **L* = 5.0** 称为最具挑战的间距。 | 已核实范围；当前可见文本没有显示 `L*` 的完整公式，本文档不额外改写其定义。 |
| 控制对象/算法 | PPO 根据传感器反馈控制圆柱的 self-rotation/rotation velocity；高保真 CFD 由 GPU 加速。研究目标是削弱尾流中圆柱的升力波动。 | 已核实。可见摘要没有明确说前、后两圆柱是否都可转；因此不在此处外推“仅后圆柱可动”。 |
| 观测传感器 | 使用传感器反馈；随后基于 POD 优化传感器分布以改善跨间距泛化。 | **数量、物理量、坐标、采样频率和状态向量组成无法由当前可访问官方文本核实。** |
| 动作频率/更新时间 | 当前可访问官方文本未给出。 | **未核实。** 不以 CFD 时间步、涡脱落频率或本项目控制间隔代替。 |
| 动作幅值/约束 | 动作为受控自旋速度，但当前可访问官方文本未给出速度的无量纲定义、上下界或变化率约束。 | **未核实。** |
| 奖励 | 摘要只说明目标和结果是降低升力波动；未展示 reward 公式、时间平均窗或动作惩罚。 | **未核实。** 不能据此声称论文奖励包含 drag、能耗或 `omega^2`。 |
| lift/drag 定义 | 当前可访问官方文本未展示 `C_L`、`C_D` 的归一化公式、正负号、前/后圆柱力的合成方式，也未展示 fluctuation 指 mean-subtracted RMS、标准差或其他量。 | **未核实。** 摘要报告“lift fluctuation reduction”不等于公布了力系数定义。 |
| 主要公开结果 | 在 L* = 5.0，800 episodes 内得到 **98%** 升力波动削弱；线性/原传感器布置跨其他间距泛化时削弱效果为 **75%–80%**；POD 优化传感器分布后跨间距均达到 **>88%**。 | 已由 PolyU 官方摘要核实。它们是论文结果，不是可直接复用的数据集。 |
| 动作时序 | 官方记录只列文章稿件，没有独立 action time-series、策略 checkpoint 或补充数据文件。摘要没有给动作序列。 | **没有发现可下载的数值动作时序。** 这不排除完整论文图中展示了示意曲线；因全文不可访问，本次不作进一步断言。 |
| CFD 数据/代码 | PolyU IRA 记录未列 CFD 场、力时序、网格、求解器输入、代码仓库或数据集；只记录文章稿件。 | **没有发现官方公开 CFD 数据或代码。** |
| 精确 action–lift 相位差 | 官方摘要和当前可访问的原论文片段没有给出角度、时间延迟、互相关峰值或明确的相位计算方法。 | **不能确认论文给出了精确相位差。** 即使完整稿的曲线视觉上呈现同相/反相，也不能据此制造一个精确数值；在取得全文并核对正文/图注前，应记为“未核实”，不能作为 Stage-C 的文献冻结参数。 |

## 对当前复现的直接含义

1. 可安全对齐的文献条件只有：Re=100、零错列串列、L* 覆盖 1.1–9.0、以 PPO 控制自旋、以升力波动削弱为核心目标；L*=5.0 是论文重点困难工况。
2. 当前项目的传感器排列、action bound/update interval、奖励权重、`C_L/C_D` 统计和 action–lift phase gate 都必须标为**本项目预先声明的复现设计**，不能声称是 Zhao et al. (2024) 的精确参数。
3. 若要做逐参数忠实复现，下一步的必要条件是取得可逐页引用的作者接受稿/出版版，并保存页码、公式号、图号；在此之前不应把二手引文或由图估读的相位差写进冻结协议。

## 官方来源

- PolyU Scholars Hub: <https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/>
- PolyU Institutional Research Archive: <https://ira.lib.polyu.edu.hk/handle/10397/107766>
- Publisher DOI record: <https://doi.org/10.1016/j.oceaneng.2024.118138>
