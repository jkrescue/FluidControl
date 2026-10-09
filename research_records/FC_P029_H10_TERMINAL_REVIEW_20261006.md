# FC-P029 H10 同协议终态复核（2026-10-06）

## 结论

FC-P029 的 train-only、origin=51、H10、44 轨迹记录动作诊断成功终止，但结果状态仍为 `P029_P027_MATCHED_SHORT_HORIZON_COMPARISON_COMPLETE_NOT_ADMISSION`。它不开启科学准入、PPO 或新候选保存。

在完全相同的 44 轨迹、origin=51、H10 和记录动作下，P029 相对 K1 父模型将 AR 场 mean-case normalized RMSE 从 `0.03890924` 降至 `0.03793724`（-2.50%），同时将 rear-Cl MAE 从 `0.03869322` 降至 `0.03335771`（-13.79%），总 Cd MAE 从 `0.01690930` 降至 `0.01538605`（-9.01%）。但 P029 的场误差仍高于 P028 的 `0.03354390`（+13.10%），且 rear-Cl 波动 RMS 改善不具有跨 family/phase 一致性。因此它支持“固定控制相关力损失能改变短时 AR 误差分布”，但不支持“已修复原 formal 失败”。

## 实际执行与来源证据

- user unit：`fluid-control-fcp029-h10-comparison-20261006.service`
- InvocationID：`f7291c51f1f1423fa4093b08e71371fd`
- 终态：`active/exited`，`ExecMainStatus=0`，容器 exit code `0`
- 容器 ID：`daeefd16d57c58d28aeeef82a288ab09c69c724f6db60e8969cf8fd447b937d1`
- 镜像：`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`
- 执行批准：`docs/FC_P029_H10_COMPARISON_EXECUTION_APPROVAL_20261006.json`，SHA-256 `6a3aaa1aa22b02c24d470992017b28bc446e009211291e550a3ee81b1a49b771`
- immutable source manifest：`artifacts/fcp029_h10_source_20261006_immutable/source_manifest.json`，SHA-256 `44edc4acd9a490402407319d9c0bf023b239e130f54ee3860dd7f4fe911fc44b`
- 结果：`artifacts/fcp029_h10_comparison_20261006/result.json`，SHA-256 `aed040eb22766f8f47b1f6093f50bc1aa753dd748cd90b8d68ac9db1095b329d`
- container-created 证据 SHA-256：`c1bda061190aff6316a61cd8afdc1f3acb7eab3b467640892852abf00f24a140`
- container-terminal 证据 SHA-256：`9144e0e78c3b35733f8bce26ae48a710c6d281cd292724ecc97663f76becd93c`
- resource watch SHA-256：`868f5d0e0818397f6197c782576824acdac2576aa4313b31f0007827b6ca91f4`，34 条记录；最小 MemFree `24.90 GiB`，最小 MemAvailable `110.91 GiB`，guard 记录的最小 CUDA free `24.91 GiB`，均高于 `20 GiB` 底线。
- run log SHA-256：`feb6084c23647324111193ff05493205e2d826de7a27a20b067dbd818f81857e`。
- 容器以 `runc`、`network=none`、只读 rootfs、`12 GiB` memory limit、`.06` allocator fraction、`900 s` deadline 运行；输入源、K1/K4/P029 候选和三个 train 数据叶均是只读 mount，仅专用 output mount 可写。

## 同协议数值对照

下表中场误差是 44 个 case 的 `normalized_all_rmse` 算术平均；rear-Cl 是物理系数 MAE；总 Cd MAE 是 10 个 lead 的 `per_lead_total_drag.mae` 算术平均。

| 模型 | H10 AR 场 RMSE | H10 AR rear-Cl MAE | H10 AR 总 Cd MAE | rear-Cl RMS 绝对误差 |
|---|---:|---:|---:|---:|
| K1 父模型 | 0.03890924 | 0.03869322 | 0.01690930 | 0.01431606 |
| P028 flow repair | 0.03354390 | 0.03950744 | 0.01728356 | 0.01568219 |
| P029 control-aware flow | 0.03793724 | 0.03335771 | 0.01538605 | 0.01413810 |

H1 场 RMSE 分别为 K1 `0.00848216`、P028 `0.00735675`、P029 `0.00818601`。P029 的 44 个 `k1_h1_force_byte_equal_parent_vs_p029` 均为 true；因为 aero 模型固定且 H1 输入状态相同，三者的 H1 四力总体指标完全一致：rear-Cl MAE `0.02654604`，总 Cd MAE `0.01410799`。这是测得的 H1 force 对照，不是对场输出相同的声明。

### family 拆分（AR）

| family | 指标 | K1 | P028 | P029 |
|---|---|---:|---:|---:|
| base | rear-Cl MAE | 0.04110407 | 0.04418596 | 0.03626858 |
| base | 总 Cd MAE | 0.01792348 | 0.01776672 | 0.01664588 |
| base | rear-Cl RMS 绝对误差 | 0.01677250 | 0.01867560 | 0.01885180 |
| train8 | rear-Cl MAE | 0.06202245 | 0.06021030 | 0.05236709 |
| train8 | 总 Cd MAE | 0.02148517 | 0.02429849 | 0.01770367 |
| train8 | rear-Cl RMS 绝对误差 | 0.02015646 | 0.02107411 | 0.01710684 |
| train16 | rear-Cl MAE | 0.02401504 | 0.02330788 | 0.02021443 |
| train16 | 总 Cd MAE | 0.01335364 | 0.01317215 | 0.01265246 |
| train16 | rear-Cl RMS 绝对误差 | 0.00832532 | 0.00924446 | 0.00676160 |

P029 的 rear-Cl MAE 和总 Cd MAE 在三个 family 中都低于 K1，但 base 的 rear-Cl RMS 绝对误差从 K1 `0.01677250` 恶化到 `0.01885180`。按 source phase 看，P029 的 RMS 绝对误差仅 b04 明显改善（`0.02573240 → 0.02289433`）；b00、b02、b06 分别从 `0.01596112`、`0.00707816`、`0.01488441` 变为 `0.01657551`、`0.00716779`、`0.01509522`。这是不能用总体 MAE 掩盖的 trade-off。

## 44 轨迹复现与边界

- 实际结果有 44 行，每行 origin 固定为 51，H10 时间和记录动作均完整。
- 将新结果的 case/family/phase/origin/times/actions，K1/K4 H1/AR/persistence 全量 metrics，以及 parent H1/AR 场 metrics 与已保存 P028 结果 `6146ea9276570981cc72c949e3e6fac46737c43aa83c561a37eaa0e54a4ab793` 逐项比较，差异数为 `0`。这证明旧 P027 K1/K4 对照臂被原样复现。
- 上述保留子集的 canonical compact JSON SHA-256 为 `00bcc730243fb15489113fcb0fbf362a3a37e9e135de52b52131bdbfbbbbe165`。该哈希是本次独立复核生成的对照摘要，不是原始 artifact 的签名或新准入凭据。
- 结果明确记录 `optimizer_created=false`、`model_saved=false`、`validation_accessed=false`、`frozen_test_accessed=false`、`scientific_admission=false`。该诊断只读 train-only 证据，没有改写候选、没有做参数选择，也不能代替原 formal suite 或真实 CFD 闭环验证。

