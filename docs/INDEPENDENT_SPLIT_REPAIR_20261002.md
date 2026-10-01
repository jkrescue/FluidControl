# 扩展 CFD 数据划分独立性修复（2026-10-02）

## 发现与影响

原始 `expanded_v1` 的 32 条 OpenFOAM 轨迹已经完成 CFD 求解，但动作日程中的 `edge_hold(seed)` 只使用 `seed % 4` 选择四条固定序列。对 32 份 `case_config.json` 的动作表、初始重启场、几何和求解配置做 SHA-256 审计，`expanded_train_23`、`expanded_validation_03`、`expanded_test_03` 的签名完全一致：`8734694e9145bde613be0e5b587cbb6d47d0d0751dfaa5299f879b9679c71aba`。它们不能作为相互独立的训练、验证、测试轨迹，原始 24/4/4 划分不用于最终 FNO 留出性能声明。

## 修复方案与可复现代码

`cfd/tandem_cylinders/make_expanded_edge_replacements.py` 生成两个新的真实 CFD 工况：`expanded_validation_04`、`expanded_test_04`。它们仍从同一个已验证的 `t=80` OpenFOAM 重启场出发，保持 Re=100、L/D=5、`dt=0.005`、`t=80..160`、801 帧和后柱 `|omega|<=5`；但各自使用不同且未在原始 32 条中出现的确定性置换 `edge_hold` 动作表。预仿真审计还要求其最大 `|domega/dt|` 不超过训练集的 6.6667。求解仍使用隔离 OpenFOAM v2512 容器，并要求常规数值 QC、原始力/动作标签对齐和 Curator 全场 QC。

`scripts/build_independent_expanded_v2_spark.sh` 等待原始 Curator 与两条替代 CFD 均完成，然后通过同一文件系统的硬链接，从原始 HDF5 复用 30 条非重复轨迹；通过官方 PhysicsNeMo Curator 路径导入两条新轨迹，形成独立目录 `data/curated/tandem_cylinders_expanded_independent_v2/`。原始 `expanded_v1` 保留不变，不删除原始 HDF5/CFD。新目录完成后重新计算仅训练集归一化、验证 32 条 HDF5 与原始 OpenFOAM 标签、全场质量、物理响应、跨划分唯一性，再启动容器中的官方 PhysicsNeMo FNO/Datapipe 训练与测试。训练脚本仍有 20 GiB 可用内存下限和 20% CUDA 分配上限。

可复核的主要命令：

```bash
python3 cfd/tandem_cylinders/make_expanded_edge_replacements.py --list
python3 scripts/audit_tandem_split_integrity.py \
  --data data/curated/tandem_cylinders_expanded_independent_v2 \
  --cases-root cfd/tandem_cylinders/cases
bash scripts/build_independent_expanded_v2_spark.sh
```

截至本文初稿，新 CFD 与原始 Curator 正在运行，独立 v2 数据集和 FNO 结果尚未完成。只有日志出现 `EXPANDED_INDEPENDENT_V2_OK` 且审计输出 `SPLIT_INTEGRITY_OK`，才把新划分视为可训练；只有后续真实测试集结果通过阶段门禁，才报告代理性能。动作条件代理的反事实动作消融仍只是模型诊断，不能代替 OpenFOAM 闭环验证。
