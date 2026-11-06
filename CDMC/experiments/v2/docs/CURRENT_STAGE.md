# CDMC v2 当前阶段：正式接入矩阵无关执行与可扩展网格计划

日期：2026-09-28。当前保留同一套已训练 Geo-FNO 与 CDMC 图设置权重、同一 P1-FEM 方程和同一 Galerkin 多级校正思想。v2 现在有两条可直接调用的评测路径：原逐样本装配路径 evaluate，用于完整数值基线；新 evaluate-compiled，用于固定拓扑多次查询的矩阵无关 CDMC 推理。后者已从 experimental/ 移入 adapters/darcy_matrixfree.py、solver/matrixfree_cdmc.py 和 run.py 的正式命令。

## 当前计算流程

1. **一次性网格计划**：从网格建立 P1 单元几何、自由节点、固定稀疏传递 $P_\ell,R_\ell$、参考系数的 Galerkin 层级、参考对角线、平滑阻尼与底层参考逆。计划不生成稠密累计基 $Q_\ell$，也不编译逐单元到每层矩阵的稠密映射。NO 与完整 CDMC 求解各捕获一个 CUDA Graph。
2. **每个样本的 NO 初值**：复用已训练 Geo-FNO，预测全场 $u_0$；边界值由样本给定。
3. **每个样本的物理信息**：只根据 P1 单元与当前系数计算细层精确算子作用、右端和细层对角线。中间层 Jacobi 对角线由参考层级和传递矩阵的平方权重估计。这个估计只用于平滑尺度，不替代求解所用的 Galerkin 作用。
4. **CDMC 多级校正**：图网络读取 NO 初值对应的真实离散残差，输出平滑与粗层门控。两轮 V-cycle 中，每层算子作用沿稀疏传递链回到细层计算，即 $A_\ell(a)v=R_0\cdots R_{\ell-1}A_0(a)P_{\ell-1}\cdots P_0v$。底层当前样本的精确算子作用配合参考逆执行固定两步 PCG，外层用精确细层作用确定能量步长，得到最终场。

变化集中在数值执行表示和计划构建，**没有重新训练** NO 或 CDMC，也没有把 CDMC 改成固定粗空间求解器。中间层对角线是近似量，所以新旧后端误差不要求逐位相同；底层当前样本作用、真实残差和外层能量步长仍是精确的 P1 离散作用。当前正式快路径仅实现 Darcy P1、SPD 多级内核和 CUDA 固定拓扑 batch=1 推理；其他 PDE 尚未接入这一后端。

## 同 GPU、同权重、同 16 样本复核

服务器 conda gtmsmw、GPU 7（RTX 4090）、CPU 新样本输入、逐样本同步墙钟。下表调用 evaluate-compiled --compare-assembled，在**同一进程、同一初值、同一 16 个样本**中分别计时捕获的矩阵无关求解和原 gpu_hybrid 装配求解。热态总时间含共享的输入与 NO，以及各自的在线系统/校正，不含一次性计划和图捕获。误差为质量加权相对场 $L^2$ 百分比。两边均为两轮 CDMC；新路径底层使用固定两步 PCG。

| 数据 | 旧场误差 mean / P95 | 新场误差 mean / P95 | 旧热态 ms | 新热态 ms | 本轮时间比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 规则 Darcy，1089 节点 | 0.01020% / 0.01667% | **0.01028% / 0.01595%** | 5.146 | **2.132** | 2.41× |
| 不规则 Darcy ID，938 节点 | 0.00899% / 0.01472% | **0.00822% / 0.01479%** | 4.252 | **2.017** | 2.11× |
| 不规则几何 OOD，909 节点 | 0.05711% / 0.10177% | **0.05131% / 0.09150%** | 4.379 | **2.023** | 2.16× |

同进程对照的逐样本解向量最大绝对差 P95 分别为 $1.29\times10^{-6}$、$4.64\times10^{-7}$、$3.22\times10^{-6}$。这个对照仍是固定小网格的 pilot；时间比仅按这轮同进程记录计算。单独运行新入口、不加载旧求解器时，单样本均值拆分为：

正式实现与先前最终探针在相同 16 个样本上的逐样本场误差，最大绝对差分别仅为 0.0000056 / 0.0000053 / 0.0000084 个百分点（规则 / 不规则 / OOD）；这核对了代码搬迁未改变已验证的数值路径。K=1 的 CDMC 和固定 MG K=2 各完成了 2 样本入口烟测；主表仅使用完整 16 样本的默认 CDMC K=2。

| 数据 | 输入拷贝 ms | NO ms | 捕获的校正求解 ms | 总计 ms | 网格计划仅一次 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 规则 | 0.076 | 0.464 | 1.608 | 2.149 | 202 |
| 不规则 ID | 0.075 | 0.425 | 1.593 | 2.093 | 187 |
| 几何 OOD | 0.071 | 0.417 | 1.587 | 2.075 | 202 |

另一次各自独立进程运行的旧入口结果为 7.860 / 5.815 / 5.922 ms；其完整评测还依次运行多种 MG、CDMC 和 FEM 基线，因此**不拿这组较慢数字计算加速比**。这轮旧入口网格计划为 292 / 333 / 332 ms；新入口的独立运行计划缩至约 187–202 ms，但**完整冷启动还要加 NO 编译和校正图捕获**。新路径这两项分别约 406–411 ms 和 102–119 ms。因此不能仅凭网格计划缩短宣称单次冷启动必然更快；该设计的主要收益是固定网格上的重复求解。新路径本轮 PyTorch 峰值 allocated 为 32.0–32.8 MiB，不代表整个 GPU 进程占用。

## 新网格规模测试

下表调用正式 MeshCompiledCDMC 模块，通过 guarded scaling probe **仅构建网格计划**；规则三角网格、GPU 显存使用上限 15%、不加载 NO/CDMC 权重。其后已补做大网格完整求解与 FEM 对比，单独见[大网格对比](LARGE_MESH_COMPARISON.md)。这里的计划是进程内首个主要 CUDA 任务，而上表小网格计划是在模型加载后测量，两者的千节点绝对值不能直接相减。数值是单次墙钟，波动较大。

| 规则网格 | 节点 / 三角形 | 正式模块计划时间 | PyTorch GPU 峰值 | 进程 RSS 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 33×33 | 1089 / 2048 | 0.710 s | 2.7 MiB | 705 MiB |
| 65×65 | 4225 / 8192 | 0.688 s | 14.0 MiB | 715 MiB |
| 129×129 | 16641 / 32768 | 3.729 s | 72.6 MiB | 779 MiB |
| 257×257 | 66049 / 131072 | **63.267 s** | **426.5 MiB** | **1146 MiB** |

同一算法的先前探针在 66,049 节点记录 46.537 s；这轮正式模块为 63.267 s，说明冷构建时间尚不稳定。两轮都成功绕过了原稠密累计基在该规模约 15.75 GiB 单基内存的问题。之后的独立样本完整求解显示：该规模 CDMC 热态约 27 ms，但数值固定 MG 的 mean 场误差更低；冷启动对单次新网格仍远慢于 FEM。详见[大网格对比](LARGE_MESH_COMPARISON.md)，不能把本节仅构建网格的结果当作工业级端到端证明。

## 复现入口与证据

从服务器 Graph 工作目录执行：

    CUDA_VISIBLE_DEVICES=7 /home/caohuilin/.conda/envs/gtmsmw/bin/python CDMC/experiments/v2/run.py evaluate-compiled --data baselines/data/darcy_v2/generated/pilot_regular --proposal-dir CDMC/experiments/v2/results/pilot_regular --checkpoint-dir CDMC/experiments/v2/results/final128_regular --out CDMC/experiments/v2/results/current/compiled_regular --eval-count 16

同进程 A/B 在该命令末尾加入 --compare-assembled，并将输出目录改为 results/current/ab_regular。

    CUDA_VISIBLE_DEVICES=7 /home/caohuilin/.conda/envs/gtmsmw/bin/python CDMC/experiments/v2/experimental/mesh_plan_scaling_probe.py --resolution 257 --mode compiled-v2 --gpu-fraction 0.15 --out CDMC/experiments/v2/results/current/compiled_plan_257.json

不规则 ID 用 pilot_irregular、pilot_irregular、final128_irregular；几何 OOD 仅将数据改为 pilot_irregular_ood_notch，仍复用不规则 ID 权重。旧路径把命令换成 evaluate、输出目录换成 assembled_*_gpu7、加 --system-backend gpu_hybrid。其他设置可用 --compiled-cycles、--compiled-coarse-steps 和 --compiled-corrector 调节；默认值是本页验证的配置。

主表同进程 A/B 的原始逐样本 JSON 位于 results/current/ab_{regular,irregular,ood}/compiled_result.json；单独运行新路径位于 results/current/compiled_{regular,irregular,ood}/compiled_result.json，旧完整评测位于 results/current/assembled_{regular,irregular,ood}_gpu7/pilot_result.json。规模测试位于 results/current/compiled_plan_{33,65,129,257}.json。早先设计排查与失败分支仍见 MESH_PLAN_OPTIMIZATION.md、MESH_PLAN_SCALING_AUDIT.md 和 CDMC_MATRIXFREE_VCYCLE_PREVIEW.md；它们是探索记录，不应与本轮正式结果混算。
