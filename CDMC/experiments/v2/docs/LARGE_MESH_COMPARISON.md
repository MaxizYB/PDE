# CDMC v2 大网格同离散对比：CDMC、固定 MG 与稀疏直接 FEM

日期：2026-09-28。本测试补上此前 66,049 节点只完成网格计划构建、没有解场评测的缺口。测试对象是**合成规则 Darcy 大网格规模代理**，不是来源于工业场景的真实复杂几何。NO 和 CDMC 均复用 33×33 规则 Darcy pilot 已训练权重，不进行大网格微调。

针对大网格上 CDMC 部分样本不如固定 MG 的原因，后续已经完成同后端门控拆分、层级深度及精确数值层级排查，见[大网格门控诊断](LARGE_MESH_GATE_DIAGNOSIS.md)。

## 可比条件

- 方程、系数采样、零 Dirichlet 边界及 P1 三角有限元与 v2 pilot 相同；网格为 129×129 或 257×257，分别有 16,641 / 66,049 节点。大网格使用独立随机种子 20271001，与小网格训练数据的种子 20260927 不同。标作 train/val/test 的三组仅是本次生成器的三个**独立系数样本**，没有用于训练模型。
- 真值由同网格 SciPy 稀疏直接 FEM 求得。FEM 对照时间含 CPU 刚度装配、边界消元、稀疏直接求解；重新以 float64 计算的相对离散残差在 $2.0\times10^{-12}$ 至 $4.2\times10^{-12}$。保存真值与复算解的质量加权差低于 $3\times10^{-6}\%$。
- NO+CDMC K2 与 NO+固定 MG K2：完全相同的 NO 初值、网格计划、矩阵无关 Galerkin 作用、两轮 V-cycle、每层两次预/后平滑、末层固定两步 PCG 和能量步长；**唯一差别是 CDMC 学习门控或全部固定为 1**。MG-zero K2 从零初值开始，无 NO。FEM 为独立 CPU 直接解。
- 神经/多级方法在 GPU 7（RTX 4090）上；FEM 在同服务器 CPU 上。GPU 进程限用该卡显存的 15%。热态时间含 CPU 输入与 NO（适用时）及校正，不含一次性网格计划、NO 几何准备和 CUDA 图捕获。每个样本重复运行 5 次计时；三样本 mean 是先对各样本 5 次取均值，再对 3 个样本取均值。

## 66,049 节点主结果

场误差为相对同网格 FEM 真值的质量加权 $L^2$，单位百分比。下表各行是不同系数样本，不是同一样本的重复 seed。

| 样本 | NO 误差 | NO+CDMC K2 误差 / 热态 ms | NO+固定 MG K2 误差 / 热态 ms | 零初值 MG K2 误差 / 热态 ms | FEM 装配+直接解 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.868% | 0.10863% / 27.405 | **0.05189%** / 26.759 | 0.20689% / 25.108 | 494.659 |
| 1 | 6.571% | 0.04564% / 27.029 | **0.02945%** / 26.390 | 0.11378% / 24.963 | 492.593 |
| 2 | 5.548% | **0.02594%** / 27.334 | 0.03393% / 26.679 | 0.12548% / 25.180 | 506.692 |
| **三样本 mean** | **10.662%** | **0.06007% / 27.256** | **0.03842% / 26.609** | **0.14872% / 25.084** | **497.981** |

相对于 CPU 稀疏直接 FEM，NO+CDMC 和 NO+固定 MG 的**热态**时间比分别约为 18.3× 和 18.7×。这不是同硬件速度比较：GPU 与 CPU 的硬件差异必须一同报告。更重要的是，本测试中固定 MG 在三样本中的两组精度优于 CDMC，mean 误差也更低；当前 CDMC 图门控**没有证明在大网格上的独立精度优势**。从零初值出发的固定 MG 也更快，但仅两轮时误差明显更大。NO 初值对两条校正路径都很重要。

原 33×33 训练网格有 4 个层级，257×257 测试网格有 10 个层级；初值缺陷的分布也会随尺度变化。这些是学习门控失去优势的**可能原因**，本轮没有做因果消融，不能把它们当成已证实的解释。

## 16,641 节点规模对照

独立随机种子的一个样本：NO 误差 4.868%；NO+CDMC K2 为 **0.02230% / 8.573 ms**，NO+固定 MG K2 为 **0.02315% / 8.368 ms**，零初值 MG K2 为 0.13901% / 7.531 ms；CPU FEM 为约 **80.492 ms**，残差 $4.95\times10^{-13}$。由 16,641 增至 66,049 节点，在这组测试口径下 FEM 耗时约增 6.2×，CDMC 热态约增 3.2×；只有两个规则尺度，不能据此推出一般复杂度或工业网格结论。

## 冷启动、安全与结论边界

在三组 66,049 节点独立进程中，固定网格计划平均 **36.54 秒**；NO 准备约 0.30–0.37 秒，CDMC 图捕获约 0.14–0.16 秒。GPU PyTorch 峰值已分配约 **797 MiB**，进程峰值 RSS 约 **1.74 GiB**，均未触及预设的 15% GPU 显存上限。此前单独计划测试出现过 46.5 秒和 63.3 秒，说明冷构建时间存在明显波动。

因此，对**仅求一个新网格样本**，目前方法总时间远慢于约 0.5 秒的 FEM；只有固定网格反复更换系数/边界并复用计划时，热态优势才可能抵消准备成本。用本轮均值粗算，约需 79 次同网格查询才能摊销 CDMC 的构建与图准备；这是口径下的估算，不是实测批量吞吐或对所有 FEM 求解器的结论。此次测试也没有覆盖不规则工业拓扑、不同 PDE、不同边界类型或训练分布以外的系数对比度。

## 实现和原始证据

评测脚本位于 checks/large_mesh_comparison.py；数据由 baselines/data/darcy_v2/generate.py 按种子 20271001 生成到服务器 Graph/baselines/data/darcy_v2/generated/scale_proxy_129_independent 和 scale_proxy_257_independent。原始 JSON 是 results/current/large_mesh_129_independent.json，以及 results/current/large_mesh_257_independent_train.json、large_mesh_257_independent_val.json、large_mesh_257_independent.json；最后一个是 test 样本。旧种子 20260927 的同规模样本与小网格训练样本编号重合，仅作为管线烟测保留，不参与本页结论。

服务器 Graph 工作目录下复现一个 66,049 节点样本：

    OPENBLAS_NUM_THREADS=1 /home/caohuilin/.conda/envs/gtmsmw/bin/python baselines/data/darcy_v2/generate.py --kind regular --out baselines/data/darcy_v2/generated/scale_proxy_257_independent --resolution 257 --train 1 --val 1 --test 1 --seed 20271001

    CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 180 /home/caohuilin/.conda/envs/gtmsmw/bin/python CDMC/experiments/v2/checks/large_mesh_comparison.py --data baselines/data/darcy_v2/generated/scale_proxy_257_independent --split test --proposal-dir CDMC/experiments/v2/results/pilot_regular --checkpoint-dir CDMC/experiments/v2/results/final128_regular --out CDMC/experiments/v2/results/current/large_mesh_257_independent.json --repeats 5 --gpu-fraction 0.15
