# 模型与数据索引

默认交付为 **B 双 FNO + E082 canonical PPO + 配套 VecNormalize**。Representative256 是未采用的候选，不替换默认模型。

## Spark 工件

路径相对 `/workspace/fluid_control`；大数据与权重不包含在 GitHub clone 中。

| 工件 | 路径 |
|---|---|
| B 模型清单 | `artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json` |
| 流场权重 | 同目录 `flow/FNO.0.0.mdlus` |
| 气动力权重 | 同目录 `aerodynamic/FNO.0.1.mdlus` |
| 训练状态 | 两个分支各自的 `checkpoint.0.*.pt`，不以模型权重替代 |
| PPO 与归一化 | `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/{ppo_final.zip,vecnormalize.pkl}` |
| 固定训练统计 | `artifacts/b00_controlled_train_dataset_view_20261006/normalization.json` |

PPO SHA256 为 `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`。其余完整路径、字节数与 SHA 见[原始交付清单](../reproducibility/FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md)及[机器依赖清单](../reproducibility/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json)。

## 图片来源

`assets/` 保留 6 张实验图片和 4 张结构示意图，均来自原报告，未修改科学数据。

- [E114 闭环曲线](assets/e114_closed_loop.png)；[t*=228 CFD 流场](assets/canonical_real_cfd_t228.png)，后者不是 E114 时段。
- [B 单步预测](assets/retained_b_fields_h1.png)与[五步预测](assets/retained_b_fields_h5.png)样例。
- [PPO 训练曲线](assets/e082_ppo_learning.png)，不是 CFD 收益；[候选固定窗口比较](assets/rep256_fixed_six.png)。
- 4 张 SVG：FNO 结构、HydroGym 分工、代理训练与 CFD 验证关系；不是仿真结果。

[训练配置](../guide/TRAINING.md) · [控制实现](../guide/CONTROL.md) · [评估结果](RESULTS.md) · [运行指南](../guide/REPRODUCTION.md)
