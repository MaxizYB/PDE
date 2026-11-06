# 运行入口与服务器资产

## 1. 环境与目录

原本地机器无 GPU，不安装 torch 等训练库。接手机器可用已有 NumPy/SciPy 环境做 CPU 检查；GPU 工作继续使用原服务器。

- 用户 caohuilin，SSH 端口 10085；地址和密码由用户私下提供，不存仓库。
- 原项目根：/data1/caohuilin/Graph。
- 环境 Python：/home/caohuilin/.conda/envs/gtmsmw/bin/python。
- 不用服务器系统 Python，不改系统路径和其他项目。
- 2026-11-06 最近连接检查为超时。本次交接没有远程核对文件现状，下列路径来自已有配置/记录。

未来继续使用同一服务器，优先读取原项目数据和权重。若修改代码，建议将本仓库放在 /data1/caohuilin/Graph/handoff_20261106/，保持内部目录，通过绝对参数读原资产，不覆盖旧代码和结果。

服务器只需同步所改源码，不同步本地大量文稿，不使用删除式同步。凭据由操作者自行配置。

## 2. 资产位置

路径均相对原服务器项目根。“本地缺失”指本交接仓库，不是服务器资产丢失。

| 资产 | 相对路径 | 本交接状态 |
| --- | --- | --- |
| 六尺度 257 数据 | baselines/data/multiscale_elliptic_v1/generated/n257_s20261023/ | 仅 manifest，大数据读服务器原件 |
| 平滑 129 数据 | baselines/data/multiscale_elliptic_v1/generated/suite129_smooth_20261103/ | 本地缺失 |
| 粗糙 129 数据 | 上行路径 smooth 改为 rough | 本地缺失 |
| 平滑 FNO | CDMC/experiments/multiscale_elliptic_v1/results/refinement_20261103/smooth/fno_h1/no.pt | 本地只有训练记录 |
| 粗糙 FNO | 上行 smooth 改为 rough | 本地只有训练记录 |
| 平滑 HINTS/trunk | CDMC/experiments/multiscale_elliptic_v1/results/hints_20261104/smooth/model.pt | 本地缺失，TB 与新原型需要它 |
| 粗糙 HINTS/trunk | 上行 smooth 改为 rough | 本地缺失 |
| 128 维字典 | CDMC/experiments/multiscale_elliptic_v1/results/refinement_20261103/balanced128_train/model.pt | 已包含 |
| 平滑适应空间 | CDMC/experiments/multiscale_elliptic_v1/results/stable_space_20261104/smooth/normal_full/model.pt | 已包含 |
| 粗糙适应空间 | 上行 smooth 改为 rough | 已包含 |
| 其他适应阶段 | results/stable_space_20261104 下 normal_balanced 等 | 已有小权重与记录已包含 |
| 三类 CPU 小数据 | baselines/data/multiscale_elliptic_v1/generated 下 poisson65_smoke_20261106、screened65_smoke_20261106、trigonometric65_smoke_20261106 | 均含 mesh/train/val/test/manifest |
| 高对比反应扩散 129 | baselines/data/multiscale_elliptic_v1/generated/screened129_c1000_20261106/ | 已生成并包含，模型实验未完成 |
| 新补充空间神经权重 | 尚无正式完成路径 | 未训练验证 |

原始 JSON 中的检查点路径、参数和硬件信息保留，未自动重写机器路径。仅加载可信来源的权重。

## 3. 代码地图

当前实验根记为 CDMC/experiments/multiscale_elliptic_v1。

| 文件/目录 | 职责 |
| --- | --- |
| baselines/data/multiscale_elliptic_v1/generate_suite.py | 单位源项、多种材料扩散数据 |
| 同目录 generate_challenge.py | 变化源项、Poisson、反应扩散等新协议 |
| 实验根 run.py | 原 FNO/旧 CDMC 训练与对照 |
| refinement/study.py、core.py | 统一响应字典与滤波/投影 |
| refinement/frame_space.py | 最新完整 GPU 适应空间训练/评测 |
| refinement/stable_space.py | FP64 小系统稳定性诊断 |
| refinement/complement_study.py、complement_space.py | 待 GPU 验证的新原型 |
| refinement/cpu_complement_audit.py | 零初值、解析空间、POD机制诊断 |
| baselines/hints/、baselines/trunk_basis/ | HINTS 训练与 TB 匹配空间对照 |
| CDMC/experiments/v2/adapters 与 solver | 依赖的网格、装配和原多级内核 |

从仓库根执行，保留相对布局。本次不改成新 Python 包，也不重构代码。

## 4. CPU 检查

以下 python 代表已有 NumPy/SciPy 环境。

~~~bash
python tools/check_snapshot.py
python CDMC/experiments/multiscale_elliptic_v1/refinement/test_frame_algebra.py
python CDMC/experiments/multiscale_elliptic_v1/refinement/test_complement_algebra.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  CDMC/experiments/multiscale_elliptic_v1/refinement/cpu_complement_audit.py \
  --data baselines/data/multiscale_elliptic_v1/generated/poisson65_smoke_20261106 \
  --out local_checks/poisson_audit --repeats 1
~~~

输出须为新目录。这是代数/机制检查，不是神经基线对比。误差可核对，CPU 时间依机器改变。

## 5. 服务器小规模复测已完成版本

假定从新仓库根执行，旧资产仍在原项目根。先测试 2 个样本，用空闲 GPU，不覆盖旧结果。

~~~bash
PY=/home/caohuilin/.conda/envs/gtmsmw/bin/python
ASSET_ROOT=/data1/caohuilin/Graph
EXP=CDMC/experiments/multiscale_elliptic_v1

CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP/refinement/frame_space.py" evaluate \
  --data "$ASSET_ROOT/baselines/data/multiscale_elliptic_v1/generated/suite129_smooth_20261103" \
  --no-dir "$ASSET_ROOT/$EXP/results/refinement_20261103/smooth/fno_h1" \
  --dictionary "$ASSET_ROOT/$EXP/results/refinement_20261103/balanced128_train/model.pt" \
  --trunk-model "$ASSET_ROOT/$EXP/results/hints_20261104/smooth/model.pt" \
  --checkpoint "$EXP/results/stable_space_20261104/smooth/normal_full/model.pt" \
  --out runs/handoff_frame_smooth_test2 \
  --split test --count 2 --cycles 1 2 --repeats 3 \
  --variants frame TB_old TB_exact
~~~

粗糙材料替换数据、FNO、trunk、适应空间检查点中的材料族，输出用新名。正式评测不启用 diagnostic-filter-override。

## 6. 新原型入口：待执行，不代表已成功

先验证稀疏 FP64、反向传播和配置，不直接大规模训练：

~~~bash
CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP/refinement/complement_study.py" train \
  --data "$ASSET_ROOT/baselines/data/multiscale_elliptic_v1/generated/suite129_smooth_20261103" \
  --no-dir "$ASSET_ROOT/$EXP/results/refinement_20261103/smooth/fno_h1" \
  --dictionary "$ASSET_ROOT/$EXP/results/refinement_20261103/balanced128_train/model.pt" \
  --trunk-model "$ASSET_ROOT/$EXP/results/hints_20261104/smooth/model.pt" \
  --out runs/complement_smoke_smooth \
  --epochs 1 --train-count 4 --val-count 2 --rank 8 --power 0
~~~

之后用 evaluate、checkpoint 参数和新输出目录评测，基础资产相同。先检查 help 与保存配置。

新 challenge 数据含变化源项和反应项，旧 GPU adapter 未支持；不能仅替换数据目录就测反应扩散。必须先统一物理输入和 $A,b$ 构造。

## 7. 输出和计时

- 逐样本和详细指标：evaluation.json；表格：summary.csv；训练：training.json 与检查点。
- 总在线包括新一次 NO、每材料系统构建、滤波、空间生成/分解与全部 K 轮更新。
- 网格计划、固定 trunk/QR 等缓存单列，新网格首次成本不能隐去。
- Batch=1；warmup 是丢弃的一次预热；repeats 通常三次同步计时取中位数，再跨样本求均值。
- 各阶段额外同步剖析不能强行相加代替端到端实测。
- 新输出用新目录，不覆盖继承的 results；快速迭代先单 seed。
