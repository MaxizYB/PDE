# 多尺度椭圆方程：CDMC 典型例子与机制实验

## 当前主线｜2026-11-04

**本轮方法简化与通用性验证：** [统一响应字典的方法说明](refinement/README.md)与[三类材料、两种NO基底及强数值对照结果](results/refinement_20261103/RESULTS.md)。同一组128维校正权重仅在48个六尺度材料上训练，冻结后迁移到129²两相材料和FNO/MgNO；去掉Cartesian区域帽函数和固定正弦拼接，保留原方程校准。257²独立后32材料K2场/导数/通量误差0.01333%/0.04955%/0.05027%，完整17.96ms。下列72/144维方案为本轮完整对照与历史证据，结果不覆盖。

**补充验证：** [DCNO/M2NO强基底、匹配消融、非规则网格与对比度外推](results/extension_20261103/RESULTS.md)。新增固定小系统去偏选项，复用分解、不增加全网格算子调用；统一训练入口为 `refinement/baseline_train.py`。所有原始主结果与新验证分开保存。

**完整文章：** [Nature风格中文论文初稿](../../reports/CDMC_NATURE_MANUSCRIPT_EXTENDED_20261027.md)，已统一为当前128维非线性字典，纳入四基底对照、成熟数值成本、匹配消融与三张源数据图。

以下为前期72/144维响应原型的机制与对照索引。研究对象保持为**材料变化的非均匀标量椭圆问题**；当前128维方法入口为 `refinement/study.py`。

- [本轮完整研究报告、结果与命令](DEPTH_STUDY_20261103.md)
- [当前完整论文初稿](../../reports/CDMC_NATURE_MANUSCRIPT_EXTENDED_20261027.md)
- [结构化结果总表](results/depth_20261103/SUMMARY.csv)、[精度—时间曲线](results/depth_20261103/ACCURACY_TIME.svg)
- 代码入口：`depth_study.py`；原图模型与物理求解在 `coupled_coarse.py`、`polynomial_joint.py` 复用；`depth_report.py` 汇总与核验原始结果。

前期32留出材料上，学习72 K2场／导数／通量为0.03946%／0.12216%／0.11931%、16.24ms；联合144 K4为0.00170%／0.00325%／0.00566%、28.61ms。完整条件的同预算训练控制、等维固定空间、零初值MG、材料迁移与多载荷复用均已完成。下方收纳历史路线，当前方法以本页顶部128维方法链接为准。

<details>
<summary>历史路线与原始证据索引</summary>

**2026-10-28 跨 PDE 扩展：** [二维非均匀线性弹性的方法与完整结果](ELASTICITY_EXTENSION_20261028.md)。沿用当前“物理滤波＋图响应空间＋真实方程校准”骨架，完成 65² 训练及独立材料 129² 迁移；32 个测试样本覆盖同分布与材料刚度外推。65² 同分布参照锚定 K2 的位移／应变／应力误差为 0.00736%／0.03958%／0.06364%，总在线 12.635 ms；报告保留更强的固定正弦空间与成熟弹性 AMG、LU 对照，定位学习空间的跨尺度缺口。入口为 `elasticity_preview.py`，最终总表为 `results/elasticity_20261028/ELASTICITY_SUMMARY_20261028.csv`；原 Darcy 代码及结果不变。

创建日期：2026-10-23；研究定位更新：2026-10-24。

**2026-10-28 数值工作替代实验已完成：** [完整方法与结果](REPLACEMENT_METHOD_AND_RESULTS_20261028.md)。在原 257×257、32 个留出样本上，固定物理滤波+区域图响应移除在线递归 MG：K2 场/导数/通量误差 0.04817%/0.14292%/0.14013%，总时间 17.291 ms；K4 为 0.00751%/0.02171%/0.02021%、25.764 ms。报告包含细层 CSR 优化 MG、从零 MG、固定正弦/POD、旧版的强对照，以及单独的冷启动成本。另有[19 工作点总表](results/REPLACEMENT_SUMMARY_20261028.csv)和[三指标曲线](results/REPLACEMENT_ACCURACY_TIME_20261028.svg)。旧路线与原始结果如下保留。

当前方法实验：[强弱分离的平衡保持传递：方法、推导与首轮实验](BALANCE_TRANSFER_METHOD_AND_RESULTS.md)。已完成 258 条新记录：四种结构的主组 12/12 载荷一次固定流程达到场与通量目标；明确比较 AMG、完整静态凝聚与 LU，并记录参数复用、递归压缩和大块规模检查。主版本为数值结构原型，无 PCG、无新网络训练。

当前重点与新实验见：[复杂耦合中的计算困难：物理平衡结构在离散与粗化中的丢失](COUPLING_DIFFICULTY_FINDINGS.md)。已完成 180 条零初值数值求解记录和 6 条离散核验，定位到局部平滑与粗空间同时遗漏闭合方向近平衡自由度的机制。研究问题从复杂耦合本身出发，NO 作为可选初值与方法信息来源；下文保留此前 NO/CDMC 研究的代码、结果与历史目标。

初始实验目标：**针对优秀 NO 实际遗留任务学习设置，比通用任务训练的相同校正结构更有效；并比较完整误差—时间关系。**

2026-10-24 研究评估：[MG、CDMC 与联合空间的机制、成本及研究方向](MG_MECHANISM_AND_RESEARCH_ASSESSMENT.md)。汇总当前 32 样本对照、工业 MG 文献及最接近的神经求解工作；新增 8 样本逐阶段与固定模式空间诊断，明确两轮 MG 的物理修复来源、当前 94 次细层算子调用和下一步实际替代粗层工作的研究目标。新增[机制诊断入口](audit_mg_mechanism.py)，本轮未重新训练。

最新结构改进：[CDMC 联合粗空间：完整方法与结果](COUPLED_COARSE_METHOD_AND_RESULTS.md)。保留 MG 骨架，将神经输出由点门控改为 8 个材料相容联合模式；相同 32 个测试样本上，场误差从原门控版 0.24219% 降至 0.07553%，在线时间由 38.137 ms 变为 41.898 ms。已有旧版研究结论继续保留如下。

新增消融：[FNO 后直接联合模式校正：去掉数值 MG](NO_MG_COMPARISON.md)。独立训练无 MG 分支并重新计时所有对照；32 样本场误差为 1.16017%、总时间 7.422 ms，对 FNO 降误差 41.14%。同次复测的 MG K2 + 联合模式为 0.07554%、44.681 ms。完整报告记录导数、通量、能量、残差及阶段时间；原高精度版本保留。

本轮已完成。结论与完整表格见 [实验报告](RESULTS.md)：任务匹配优于 Generic，但训练集标量校准已能达到相近或更好的结果；当前空间图设置的独特收益尚未成立。报告保留了强数值对照及所有异常样本。

## 数据与方法

- 系数分布取自 [HANO §3.1.2](https://arxiv.org/html/2210.10890)：六尺度三角乘积。物理方程为 $-\nabla\cdot(a\nabla u)=1$、零 Dirichlet。
- 从原域 $(-1,1)^2$ 映到单位方形，解缩放为原解的 $1/4$。用**存储后的节点系数**重建相同 P1-FEM 系统生成真值，不直接使用细网格下采样标签。
- 初始配置为 257×257、512/64/64 样本、单 seed。三个测试参数做 513×513 细化审计，分别记录代数误差和离散误差。
- NO 为标准四层 FNO，width=32、modes=16、padding=9，训练集统计归一化。普通 $L^2$ 与 $L^2+0.1H^1$ 训练并行，150 epochs，同数据同种子。两者是本实验新权重，不是 HANO 官方 checkpoint。
- CDMC **直接复用** `../v2/solver/correctors.py` 的两层图消息 setup，以及 `matrixfree_cdmc.py` 的完整 V-cycle。只学习平滑门控；粗层门控固定；当前样本末层直接 Cholesky；主方法无 PCG。
- Actual / Generic 两种训练均 48 个训练样本、12 个验证样本、20 epochs、两轮完整 V-cycle；同参数量、初始权重、样本次序及优化预算。Generic 方向混合低/中/高频正弦误差，并逐样本匹配实际 NO 误差的质量范数幅值。两种训练各在自身验证任务选模，测试都从同一个真实 NO 开始。
- 强数值对照使用环境已有的成熟 PyAMG smoothed aggregation、Ruge–Stüben 和 adaptive smoothed aggregation：零初值、相同 NO 初值、固定 1/2/4 V-cycle；另记录 AMG-CG 收敛至 $10^{-10}$ 的精度端点。**该 CG 仅属于数值基线。** CPU 与 GPU 时间分别标注；设置、组装与求解单独报告。

## 文件与入口

| 文件 | 职责 |
| --- | --- |
| `../../../baselines/data/multiscale_elliptic_v1/generate.py`（实际位于项目根 baselines/data） | 系数、P1 真值、独立划分及细化审计 |
| `models.py` | 标准 FNO；不重写 CDMC |
| `run.py` | train-no / train-cdmc / evaluate / numerics / report |
| `run_study.sh` | 服务器两张空闲 GPU 顺序编排；每个科学配置只跑一个 seed |
| `audit.py` | 粗层平滑稳定性、异常样本更新方向、训练集校准标量门控对照 |
| `results/n257_s20261023/` | 权重、训练记录、逐样本指标、表格与真实样本图 |

计时为 batch=1、同步墙钟、一次预热、三次重复取中位数；重复计时不等于重复训练。NO+校正总时间真实重跑 NO、样本数值构建、一次图设置和 K 轮校正；指标计算与 CPU 绘图不在求解计时内。固定网格计划另列。

输出包括质量加权场误差、P1 导数误差、能量误差、通量误差、离散残差、分段时间；`summary.csv` 为结构化表，`evaluation.json` 保留逐样本详情。`fields_example.png` 与 `accuracy_time.png` 由真实留出数据生成。

## 运行

在服务器 `/data1/caohuilin/Graph`、conda `gtmsmw` 中：

```bash
PY=/home/caohuilin/.conda/envs/gtmsmw/bin/python
$PY baselines/data/multiscale_elliptic_v1/generate.py \
  --out baselines/data/multiscale_elliptic_v1/generated/n257_s20261023 \
  --resolution 257 --train 512 --val 64 --test 64
bash CDMC/experiments/multiscale_elliptic_v1/run_study.sh
```

单独阶段可查看 `run.py --help`。`evaluation_h1_transfer` 是**冻结 L2-NO 所配校正器的 16 样本跨基底测试**；`evaluation_h1_matched` 是导数监督 NO 配套训练后的 32 样本测试。脚本为本轮从头复现入口，GPU 索引应按当时空闲情况调整；已有结果时不要直接重跑覆盖。

## 研究判断

首先检查强 NO 的误差结构及细化审计，再判断固定数值校正消除了哪些分量。任务匹配收益以 Actual 对 Generic 的相同测试比较衡量，而非仅以 NO 到 CDMC 的下降衡量。若困难主要来自当前粗空间遗漏，应记录为结构边界并定位；不把平滑门控反复调参当作粗空间改进。所有收益由留出结果决定，未预设新例子一定胜出。

## 当前记录

1. 本地无 ML 依赖的数据烟测、服务器小网格 FNO→完整 CDMC 训练→数值对照→评测/出图已通过。
2. 257×257 数据已生成：512/64/64 样本；测试材料节点对比度约 10244–64451。float64 参考求解相对残差最大约 $1.01\times10^{-11}$。
3. 三个测试样本的 257 解与 513 解限制值相差 9.43%、8.96%、9.51%。因此本轮首先是**离散系统上的互补任务机制实验**。升为连续多尺度 PDE 的主物理例子之前，还要提升/核验空间分辨率，不能让校正后很小的代数误差替代这一精度审计。
4. 普通与导数监督 FNO 均完成 150 epochs；Actual、Generic、H1-Actual 校正均完成 20 epochs。主评测各 32 样本，三种成熟 CPU 数值对照各 16 样本。
5. 普通 FNO 场误差 1.9711%，Actual K2 为 0.24221%，Generic K2 为 0.34151%。训练集平均标量门控 K2 为 0.23391%。已完成逐层稳定性与停滞方向诊断，详见报告，不将 Actual-vs-Generic 的差异单独当作空间图设置的证据。

</details>
