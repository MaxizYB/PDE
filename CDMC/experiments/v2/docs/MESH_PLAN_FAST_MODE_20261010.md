# CDMC v2：局部传递的快速网格计划

日期：2026-10-10。本文记录新网格计划构建的优化、误差代价及正式入口。采用已训练模型做冻结权重对照，不生成新真值、不重训 NO，也不增加外层迭代。原默认配置保留；快速配置显式指定 `--interpolation-max-entries 64`。

## 1. 最终结果

257×257 规则 Darcy：66,049 节点、65,025 自由未知量、131,072 三角形、10 层、末层 128 维。同一 GPU/进程、同一 12 个留出样本、同一冻结 Geo-FNO 与 CDMC 权重的对照如下。

| 设置 | 完整新网格计划 | CDMC K2 mean | CDMC K2 P95 | 同设置固定 MG K2 | 在线系统+CDMC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原无限制传递 | 42.973 s | 0.019294% | 0.027153% | 0.028210% | 35.759 ms |
| **每行最多 64 项** | **5.435 s** | **0.020327%** | **0.032583%** | 0.032317% | 38.724 ms |
| 每行最多 32 项 | 3.194 s | 0.052709% | 0.110609% | 0.066879% | 36.510 ms |

64 项配置将完整计划时间降低 **87.4%（7.91 倍）**。mean 误差仅增加 **0.001033 个百分点**，相对增加约 5.4%；在线时间增加约 8.3%。它没有把计算挪到未计时的准备阶段：计划中仍包括参考装配、全部层级、GPU 传递、参考辅助量和末层投影准备。

两种配置都固定两轮 V-cycle、每层两次前/后平滑、当前样本末层直接 Cholesky、一次图网络设置，无 PCG。快速配置的 mean 仍比其同条件固定 MG 低约 **37.1%**。原始记录：`../results/performance/plan_speed_20261010/paired_local3264.json`。

## 2. 瓶颈定位

本轮补齐了完整计划的分段计时。

| 构建阶段 | 原计划 | 64 项计划 | 工作内容 |
| --- | ---: | ---: | --- |
| 参考几何与细层 FEM 装配 | 87.91 ms | 70.60 ms | 单元梯度、质量、参考刚度及边界消元 |
| **CPU 参考层级** | **42,288.78 ms** | **5,142.96 ms** | 聚合、平滑插值、逐层稀疏 Galerkin 乘法 |
| 传递、图和设备准备 | 283.84 ms | 62.23 ms | 稀疏传递、门控限制及细层图 |
| 参考辅助量 | 252.38 ms | 127.22 ms | 各层阻尼、对角线、平方传递与尺度 |
| 末层投影几何 | 48.10 ms | 30.62 ms | 128 个末层基传播及单元投影梯度 |
| 其他 | 12.06 ms | 0.95 ms | 对象及小层准备等 |
| **完整计划** | **42,973.07 ms** | **5,434.58 ms** | 上述全部项目 |

原构建时间约 **98.4%** 在参考层级，并非网络预测、末层求逆或模型编译。之前的 34.33 s 是另一轮同规模墙钟记录；本页加速比使用本次成对复测的 42.973 s，不拼接历史时间。

## 3. 改变的数学对象

原始每层先产生二节点聚合的暂定插值 $Q_\ell$，再做一次 Jacobi 平滑：

$$P_\ell=(I-\omega_\ell D_\ell^{-1}A_\ell^{\mathrm{ref}})Q_\ell.$$

聚合越深，$P_\ell$ 的每行支撑越大。下一层参考算子

$$A_{\ell+1}^{\mathrm{ref}}=P_\ell^\top A_\ell^{\mathrm{ref}}P_\ell$$

随之填充，稀疏乘法的大量成本来自中间非零项，而不是层数本身。

快速路径对插值支撑局部化。令 $g(i)$ 为节点 $i$ 所属聚合列，$S_i$ 为保留集合：始终包含 $g(i)$，其余取绝对值最大的项，总数最多 $m=64$。被删除项的代数和为

$$\delta_i=\sum_{j\notin S_i}(P_\ell)_{ij}.$$

新的传递为

$$\widehat P_{ij}=
\begin{cases}
P_{ij}+\delta_i,&j=g(i),\\
P_{ij},&j\in S_i\setminus\{g(i)\},\\
0,&j\notin S_i.
\end{cases}$$

因此保留原插值对常数向量的作用：

$$\widehat P_\ell\mathbf 1=P_\ell\mathbf 1.$$

这里不强行将行和归一化为 1；原来的 Dirichlet 处理和带符号插值仍被保留。最终使用

$$\widehat R_\ell=\widehat P_\ell^\top,\qquad
\widehat A_{\ell+1}^{\mathrm{ref}}=
\widehat R_\ell\widehat A_\ell^{\mathrm{ref}}\widehat P_\ell.$$

**不再截断这些 Galerkin 矩阵。** 对当前样本仍精确施加

$$\widehat A_\ell(a)v=
\widehat R_{\ell-1}\cdots\widehat R_0 A_0(a)
\widehat P_0\cdots\widehat P_{\ell-1}v.$$

因此变化的是粗空间，不是细层 PDE 或真值。参考层级、训练/推理传递和末层矩阵共用同一套局部传递；当前细层残差及外层能量步长继续用真实 $A_0(a)$。257 对照中的末层矩阵与隐式作用差异为约 $10^{-6}$ 量级。

### 复杂度与时间收益

传递存储和传递作用的规模现在受 $m\sum_\ell N_\ell$ 约束，不再允许插值支撑不受限地扩展。测试网格的参考算子复杂度

$$C_A=\frac{\sum_\ell\operatorname{nnz}(A_\ell^{\mathrm{ref}})}
{\operatorname{nnz}(A_0^{\mathrm{ref}})}$$

从 **36.32** 降至 **25.69**；填充中间乘法的工作量下降更显著，因而建网格时间下降近 8 倍。只限制 $P$ 尚不能对任意图保证所有粗矩阵严格有界度，也没有消除在线隐式作用反复返回细层的成本。故本改动解决的是冷网格计划，在线求解仍在几十毫秒。

对于每网格仅一个查询，计入计划的数值部分约从 **43.009 s 降至 5.473 s**；两者均另需相同 NO/输入准备。固定网格大量查询时，原精确路径的在线延迟略低；按本次约 37.54 s 的建网格收益和约 2.97 ms 的单查询差异粗估，需约 1.27 万次查询后原路径才追回这次冷启动差值。

## 4. 小规模与不规则核对

| 数据与权重 | 测试数 | 原 K2 mean | 64 项 K2 mean | 原/新参考层级构建 |
| --- | ---: | ---: | ---: | ---: |
| 129×129，多尺度平滑-only权重 | 8 | 0.018301% | 0.018722% | 2.254 / 0.870 s |
| 凹口不规则 Darcy，原不规则 pilot 权重 | 8 | 0.009584% | 0.009584% | 15.81 / 15.30 ms |

129 的完整计划为 **2.413 → 0.937 s**，两轮误差仅增加 0.00042 个百分点。不规则 pilot 的支撑原本少于 64 项，优化基本不触发，其层级大小与算子非零数完全相同。该小网格完整构建的首轮设备初始化比后续轮次更重，因此不把其完整墙钟差异当作算法加速。

不规则核对复用的是该几何的历史权重，不是最新规则多尺度权重；它检验同一个传递局部化实现可用于不规则网格。尚未测试大型不规则网格。结果见 `scale129.json` 和 `irregular.json`。

## 5. 其他尝试的结论

以下是设计诊断，不同轮次的时间不用于和主表计算严格加速比。

| 改法 | 257 网格计划 | CDMC K2 | 观察 |
| --- | ---: | ---: | --- |
| 只截断参考粗矩阵，1e-4 | 7.18 s | 0.10409% | 参考尺度/粗空间变动较大，精度损失明显 |
| 只截断参考粗矩阵，1e-3 | 4.02 s | 0.14502% | 更快，但不作为推荐配置 |
| 四节点聚合 | 0.93 s | 0.62425% | 冷启动很快，但两轮粗空间不足 |
| 三节点聚合 | 2.80 s | 0.35981% | 4轮后约0.02833%，在线约60.50 ms |
| 四节点聚合、6轮 | 1.02 s | 0.00539%（K6） | 精度恢复，在线约58.79 ms |
| 粗层停止阈值256/512 | 49.57 / 44.82 s | 0.01928 / 0.01925% | 大头在更早中间层，没有解决冷启动 |
| 三节点聚合、每侧平滑4/6次 | 2.59 / 2.34 s | 0.26748 / 0.22395% | 增加局部平滑不能代替缺失粗空间 |

本轮最终保留64项局部传递，因为它不增加轮数，也不破坏严格的参考/当前样本 Galerkin 一致性。32项、16项、8项是更激进的速度档，但误差代价明显更大。

## 6. 入口、兼容性和证据

实现仍在原v2文件夹：

- `solver/numerics.py`：局部传递；默认上限0，等价原无限制路径。
- `adapters/darcy_system.py`：参考层级、传递和分段时间。
- `solver/matrixfree_cdmc.py`：完整计划时间；直接解配置不再计算未使用的参考逆。
- `direct_multiscale.py`：训练和评测共用层级选项；新checkpoint保存 `hierarchy_options`，评测默认继承。
- `checks/mesh_plan_accuracy_speed.py`：同初值/同权重的计时与误差诊断。
- `checks/test_mesh_plan_localization_cpu.py`：CPU回归，检查行和、缺失主聚合列的补偿支撑、默认不变、Galerkin等价及小系统正定性，5项通过。

旧checkpoint默认视为二节点、无限制传递；显式更改需加 `--allow-hierarchy-change`，输出JSON同时记录训练和评测设置。这个开关仅记录冻结权重跨层级测试，不引入自动回退或误差阈值。

从服务器Graph目录运行：

```sh
PY=/home/caohuilin/.conda/envs/gtmsmw/bin/python
CUDA_VISIBLE_DEVICES=5 "$PY" CDMC/experiments/v2/direct_multiscale.py evaluate \
  --data baselines/data/darcy_v2/generated/cdmc_final_257_s20271003 \
  --proposal-dir CDMC/experiments/v2/results/pilot_regular \
  --checkpoint-dir CDMC/experiments/v2/results/current/multiscale_direct_smooth_K2 \
  --out CDMC/experiments/v2/results/current/fast64_257 \
  --split test --eval-count 12 \
  --interpolation-max-entries 64 --allow-hierarchy-change
```

训练时指定 `--interpolation-max-entries 64`，参数会写入checkpoint；新checkpoint的评测不必重复指定。原默认训练/测试仍使用0；本轮主结果是复用旧权重的测试，没有将重训成本隐藏在前置时间之外。

本页时间口径：RTX4090、gtmsmw环境、CPU BLAS与PyTorch线程设为4；在线batch=1，同步墙钟，1次warmup、3次重复取每样本中位数再平均。在线计时含真实样本系统/末层Cholesky和图设置/两轮校正，不含NO、输入搬运、场误差计算或一次性计划。冷计划计时单列、每配置一次；系统负载会带来墙钟波动。

全部JSON在 `results/performance/plan_speed_20261010/`：主成对结果为 `paired_local3264.json`；其余设计诊断为 `regular257*.json`、`local.json`、`coarse*.json`、`agg3_smooth*.json`；小尺度与不规则验证为 `scale129.json`、`irregular.json`。正式入口烟测另见 `entry_eval257/direct_result.json`；不是主表的稳定计时来源。
