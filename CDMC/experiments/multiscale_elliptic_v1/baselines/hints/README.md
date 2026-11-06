# HINTS-Jacobi baseline

二维 DeepONet 与 Jacobi–DeepONet 交替求解，在当前 Darcy 上训练、测试，与冻结 FNO 和当前 CDMC 同口径比较。

## 方法

对细网格 P1 系统 $Au=b$，每块先执行 14 次阻尼 Jacobi：

$$u\leftarrow u+0.8D^{-1}(b-Au),\qquad D=\operatorname{diag}(A).$$

再将载荷残差转成源项 $f_r=M_{\mathrm{lump}}^{-1}(b-Au)$，将系数与源项插值到 31×31 传感器，执行：

$$u\leftarrow u+\mathcal G_\theta(a,f_r).$$

源项按 RMS 归一化，预测后乘回幅值。branch 为官方四层 stride-2 卷积与全连接层；trunk 为宽度 80 的三层 tanh MLP，乘齐次 Dirichlet 边界因子。固定几何 trunk 缓存复用。无 PCG、线搜索或额外校准。

## 训练与适配

- 每类使用原数据的 256 个训练材料、32 个独立验证材料；每材料 16 个 RHS（单位源项 + 15 个平滑随机源项），得到 4096 / 512 个训练 / 验证对。
- 标签来自原 129×129 细网格求解。65×65 节点仅用于 trunk 监督采样，测试直接查询 129×129 节点。
- 生成只读 train / val，未使用 test 标签；1000 epochs、batch 128、Adam，按验证误差选检查点。
- 当前材料分布与 FEM 离散沿用本项目；系数按训练均值 / 标准差归一化，输出按训练解 RMS 缩放，损失为归一化解单位的官方 L2_relative 形式，学习率使用 cosine schedule。这些是本项目训练适配。
- 官方来源与 SHA256 见 vendor/PROVENANCE.json。audit.py 逐项拷贝 branch / trunk 权重，数值核对官方 forward。

## 文件

| 文件 | 作用 |
|---|---|
| model.py | DeepONet 与传感器插值 |
| data.py | 同矩阵多 RHS 标签及真值对齐检查 |
| train.py | 训练与独立验证 |
| evaluate.py | 零初值 / FNO 初值 HINTS、Jacobi、CDMC 评测 |
| audit.py | 官方网络计算一致性 |
| diagnose.py | 检查点重载验证、单位源项与随机 RHS 诊断 |
| report.py | CSV、中文报告与误差—时间图 |

## 运行

在服务器 /data1/caohuilin/Graph 中选择空闲 GPU。以 smooth 为例，rough 使用相同设置。

~~~bash
py=/home/caohuilin/.conda/envs/gtmsmw/bin/python
exp=CDMC/experiments/multiscale_elliptic_v1
family=smooth
data=baselines/data/multiscale_elliptic_v1/generated/suite129_"$family"_20261103
run=$exp/results/hints_20261104/$family
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

$py $exp/baselines/hints/audit.py --out $exp/results/hints_20261104/official_audit.json
$py -u $exp/baselines/hints/train.py --data "$data" --out "$run" --epochs 1000
$py $exp/baselines/hints/diagnose.py --data "$data" --run "$run" --out "$run/diagnostics.json"
$py -u $exp/baselines/hints/evaluate.py \
  --data "$data" --model "$run/model.pt" \
  --no-dir "$exp/results/refinement_20261103/$family/fno_h1" \
  --dictionary "$exp/results/refinement_20261103/balanced128_train/model.pt" \
  --out "$run/loaded_test32" --count 32 --blocks 1 2 4 8 16 32 --repeats 3
$py $exp/baselines/hints/report.py --root "$exp/results/hints_20261104"
~~~

汇总命令需要两类测试都完成。复用检查点时直接运行 diagnose 和 evaluate。报告生成仅依赖 NumPy / Matplotlib，可使用本地现有 .venv。

## 输出与计时

有效目录：results/hints_20261104/{smooth,rough}/loaded_test32。

- summary.csv：场、梯度、能量、通量误差及在线总时间。
- evaluation.json：逐样本、mean / median / P95、残差、频段、阶段计时与计时口径。
- training.json / pairs/manifest.json：训练及生成成本、样本数、真值一致性。
- 汇总根目录的 REPORT.md、all_methods.csv、accuracy_time.png：完整比较。

mesh plan、assembly plan 与 trunk 缓存成本单列。在线时间含每次材料组装、全部 Jacobi / 神经调用，以及使用 FNO 时的初始预测。HINTS 块与 CDMC cycle 的工作量不同，通过完整时间曲线比较。

首轮 test32 因评测入口漏载权重作废；正式汇总只读取 loaded_test32。HINTS-MG / Krylov 为其他变体，本次运行的是 HINTS-Jacobi。

## 来源

- Zhang et al., *Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers*: https://arxiv.org/abs/2208.13273
- 官方代码：https://github.com/kopanicakova/HINTS_precond
