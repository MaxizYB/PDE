# DeepONet trunk-basis：共同后校正流程中的空间对比

## 实现范围

按 DeepONet Based Preconditioning Strategies for Solving Parametric Linear Systems of Equations 的 §4.2 构造坐标 trunk 空间。复用已经训练好的 HINTS DeepONet，加载权重后用双精度计算带齐次边界因子的 trunk，并进行 QR。在线不调用 branch，而以当前细网格矩阵求 Galerkin 系数。

本目录比较的是 **TB 空间与 CDMC 空间在同一固定次数后校正器中的效果**，不是原论文完整的 TB-Krylov 求解器复现，也不包含学习型 AMG 或 GMsFEM-NO。

- `evaluate.py`：同一 FNO、细网格组装、预后滤波、正则、校准规则，测试 K=1/2/4。
- `qr_audit.py`：独立验证集上将 CDMC 空间进行不降维 QR，诊断坐标条件及正则的影响；不用于宣称速度提升。
- `report.py`：读取结果生成汇总 Markdown、CSV 和精度—时间图。

TB 使用各材料族 256 个训练材料、每材料 16 RHS 的已有检查点；CDMC128 使用原有六尺度训练检查点。两者没有等训练预算，报告是已有模型的部署对比。测试真值仅用于指标及离线诊断，不输入预测流程。

## 服务器复现

在 `/data1/caohuilin/Graph` 下执行；使用已有 conda 环境，不使用系统 Python。以下为平滑材料，粗糙材料将路径中的 `smooth` 换为 `rough`，另选空闲 GPU。

```bash
CUDA_VISIBLE_DEVICES=2 /home/caohuilin/.conda/envs/gtmsmw/bin/python CDMC/experiments/multiscale_elliptic_v1/baselines/trunk_basis/evaluate.py \
  --data baselines/data/multiscale_elliptic_v1/generated/suite129_smooth_20261103 \
  --no-dir CDMC/experiments/multiscale_elliptic_v1/results/refinement_20261103/smooth/fno_h1 \
  --trunk-model CDMC/experiments/multiscale_elliptic_v1/results/hints_20261104/smooth/model.pt \
  --dictionary CDMC/experiments/multiscale_elliptic_v1/results/refinement_20261103/balanced128_train/model.pt \
  --compact-dictionary CDMC/experiments/multiscale_elliptic_v1/results/refinement_20261103/smooth/dictionary64/model.pt \
  --out CDMC/experiments/multiscale_elliptic_v1/results/trunk_basis_20261104/smooth/test32 \
  --split test --count 32 --cycles 1 2 4 --repeats 3
```

在线时间包含新一次 FNO、每材料细网格组装、空间准备与分解、所有校正。固定网格计划和固定 trunk/QR 缓存为一次性成本，单列在 JSON 与报告中。Batch=1，一次预热后取三次同步墙钟计时中位数。

## 输出

`results/trunk_basis_20261104/REPORT.md` 是完整报告；`all_methods.csv` 是统一主指标；`accuracy_time.png` 是曲线。各材料族的 `test32/evaluation.json` 保存全部逐样本指标、阶段计时、配置、检查点路径与离线诊断；`qr_audit_val8.json` 保存 QR 验证诊断。
