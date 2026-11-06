# CDMC v2：统一 Darcy 多级实现

项目总览见 [CDMC 版本与实验总整理](../../PROJECT_SUMMARY.md)。**当前空间主线是 `direct_multiscale.py` 的完整 V-cycle 同路径训练/测试、平滑-only设置和当前样本末层直接 Cholesky，无 PCG。** 下文 `run.py evaluate` / `evaluate-compiled` 保留早期对照与执行优化入口，不与无 PCG 主实验混记。

当前完整方法以[面向一般PDE的统一方法与实验文档](docs/CDMC_V2_UNIFIED_METHOD.md)为主：连续展开离散接口、Geo-FNO、图设置、完整V-cycle、训练、最新快速网格计划及分段成本；历史结果按配置单列。

2026-10-10 新网格计划加速：新增局部传递选项，257×257 成对测试完整计划 **42.97→5.43 s**，两轮误差 **0.01929→0.02033%**。保持原默认无限制传递；快速路径显式使用 `direct_multiscale.py` 的 `--interpolation-max-entries 64`，旧权重测试再加 `--allow-hierarchy-change`。新训练checkpoint记录层级设置，评测自动继承。方法、分段计时、失败尝试和原始结果见[快速网格计划报告](docs/MESH_PLAN_FAST_MODE_20261010.md)。

这是当前 v2 的正式入口。规则方形域与凹口不规则域共用 P1-FEM + Galerkin + CDMC 主干；NO 初值目前使用可替换的 Geo-FNO 风格模型。当前完整方法见[统一方法文档](docs/CDMC_V2_UNIFIED_METHOD.md)，[Darcy 原方法记录](docs/DARCY_CDMC_METHOD.md)保留旧配置；小网格结果见[阶段整理](docs/CURRENT_STAGE.md)和[实验汇总](docs/EXPERIMENTS.md)，66,049 节点的 CDMC / 固定 MG / FEM 历史对比见[大网格对比](docs/LARGE_MESH_COMPARISON.md)。9月29日训练/测试和成本见[Darcy v2 阶段总结](docs/V2_STAGE_SUMMARY_DARCY_20260929.md)。训练与测试共用矩阵无关、末层直接求解路径的多尺度实验见[无 PCG 大网格实验](docs/MULTISCALE_DIRECT_LARGE_MESH.md)，入口为 `direct_multiscale.py`。旧式逐样本装配由 evaluate 保留为对照；已有 evaluate-compiled 路径在末层使用固定两步 PCG，新多尺度路径则用当前样本的小系统直接求解。其他早期探索仍位于 experimental/；固定粗空间全局 PCG 是另一条纯数值基线。v2继承v1/v1.1的NO驱动互补后求解思想，进一步统一离散和执行；本目录数据为自生成验证集，与论文原始数据集分别报告。

## 目录

| 路径 | 职责与扩展位置 |
| --- | --- |
| run.py | Darcy 训练、评测与结果 JSON 的统一命令入口 |
| direct_multiscale.py | 多尺度 CDMC 训练与同一路径无 PCG 评测；冻结 NO，末层当前样本直接解 |
| adapters/darcy.py | Darcy 网格/数据读取、NO 输入、真值误差和规则/不规则 profile |
| adapters/darcy_system.py | Darcy P1 装配、边界消元、固定网格计划和逐样本矩阵构建 |
| adapters/darcy_matrixfree.py | Darcy P1 单元矩阵无关作用；新推理后端不逐样本装配刚度矩阵 |
| solver/model.py | NO 模型注册表；新增神经初值架构由这里接入 |
| solver/inference.py | 冻结 NO 的固定网格准备与 CUDA Graph 推理 |
| solver/numerics.py | PDE 无关的 SPD Galerkin 层级、固定轮数 V-cycle 与能量更新 |
| solver/correctors.py | 固定 MG 与 CDMC 图设置；新增后校正器由这里接入 |
| solver/matrixfree_cdmc.py | 固定网格参考层级、因子化 Galerkin 作用、CDMC 门控与多级校正 |
| checks/ | 数值一致性、推理一致性、能量下降、性能诊断与删减审计 |
| experimental/ | 与正式入口隔离的矩阵无关、固定粗空间及 FEM 对照预览代码 |
| docs/ | 正式方法/实验汇总及明确标记的独立探索报告 |
| results/current/ | 最后一次三几何主评测的逐样本 JSON |
| results/ablations/ | 相同权重的核心删减消融（旧后端计时） |
| results/performance/ | NO、数值系统及混合 GPU 后端的执行路径 A/B 对照 |
| results/archive/ | 早期 pilot、修复和训练日志，供追溯而非现行引用 |

主干的可复用部分是 NO 预测接口、图 setup、Galerkin 传递和固定轮数校正。Darcy P1 装配现已单独位于 adapters/darcy_system.py；扩展到另一 PDE 时应提供同样的矩阵/右端/特征契约，再判断 SPD V-cycle 是否适用，不能仅新增数据读取类。改变几何而不改变 Darcy 方程时，只需提供同格式 mesh 和样本。数据生成代码及格式说明分别在项目根目录 baselines/data/darcy_v2/generate.py 与 SPEC.md。

## 当前主线运行

- 精度基准、多尺度训练与 257×257 测试：使用 `direct_multiscale.py train/evaluate`，完整命令统一见 [无 PCG 多尺度实验](docs/MULTISCALE_DIRECT_LARGE_MESH.md#原始证据与复现入口)。
- 快速网格计划：同一个 `direct_multiscale.py evaluate` 入口，显式指定 `--interpolation-max-entries 64`；旧权重跨层级测试还需 `--allow-hierarchy-change`，完整命令见 [快速计划配置](docs/MESH_PLAN_FAST_MODE_20261010.md#6-入口兼容性和证据)。
- 当前路径不需要先运行下面的旧 `evaluate-compiled`，也不继承它的末层 PCG 配置。

## 历史基础入口与后端对照

在项目根目录执行。大模型训练/推理使用服务器 conda gtmsmw 环境；本地不安装 torch。以下路径按服务器的 Graph 工作目录表示。

    PY=/home/caohuilin/.conda/envs/gtmsmw/bin/python
    $PY baselines/data/darcy_v2/generate.py --kind regular --out baselines/data/darcy_v2/generated/pilot_regular --resolution 33 --train 64 --val 16 --test 16
    $PY baselines/data/darcy_v2/generate.py --kind irregular --out baselines/data/darcy_v2/generated/pilot_irregular --max-area 0.0008 --train 64 --val 16 --test 16

    CUDA_VISIBLE_DEVICES=4 $PY CDMC/experiments/v2/run.py all --data baselines/data/darcy_v2/generated/pilot_regular --out CDMC/experiments/v2/results/new_run_regular --no-epochs 30 --cdmc-epochs 8 --coarse-size 128

也可分别调用 train-no、train-cdmc、evaluate；评测几何 OOD 时用 --proposal-dir 和 --checkpoint-dir 指定不规则 ID 权重，再把 --data 指向新的网格数据。旧 evaluate 的 --proposal-mode 可选 reference、prepared、graph；--system-backend 可选 auto、cpu、gpu、gpu_compiled、gpu_hybrid、gpu_affine。新 evaluate-compiled 目前适用于 CUDA 上固定拓扑的 Darcy P1 数据，复用同一训练权重；默认两轮 CDMC，每层预/后各两次平滑，最底层固定两步 PCG。--compiled-cycles、--compiled-coarse-steps 和 --compiled-corrector 可控制测试配置。--eval-count 可作小型运行检查。

    CUDA_VISIBLE_DEVICES=7 $PY CDMC/experiments/v2/run.py evaluate-compiled --data baselines/data/darcy_v2/generated/pilot_regular --proposal-dir CDMC/experiments/v2/results/pilot_regular --checkpoint-dir CDMC/experiments/v2/results/final128_regular --out CDMC/experiments/v2/results/current/compiled_regular --eval-count 16

该命令输出 compiled_result.json，包含逐样本 NO 与 CDMC 场误差、输入/NO/校正图执行/总时间，以及一次性网格计划、NO 编译、校正图捕获时间。加 --compare-assembled 可在同进程对照原装配后端并记录两边的场误差、总时间和解差；独立旧路径的完整基线评测仍使用 evaluate，输出 pilot_result.json。其他旧后端参数及完整命令见帮助信息：

    $PY CDMC/experiments/v2/run.py --help

训练输出目录含 proposal.pt、cdmc.pt 和训练日志；evaluate 输出 pilot_result.json，evaluate-compiled 输出 compiled_result.json。当前本地 results/current 仅同步 JSON，**不含 checkpoint**；服务器原有 NO 权重分别在 results/pilot_regular、results/pilot_irregular，CDMC 权重分别在 results/final128_regular、results/final128_irregular，没有移动服务器文件。若在整理后的本地目录做评测，需用 --proposal-dir 和 --checkpoint-dir 分别指向实际权重位置。

## 一致性检查

从项目根目录运行下列脚本，数据/权重目录按实际位置替换：

    $PY CDMC/experiments/v2/checks/verify_system_backends.py --data baselines/data/darcy_v2/generated/pilot_regular
    $PY CDMC/experiments/v2/checks/verify_inference.py --data baselines/data/darcy_v2/generated/pilot_regular --proposal-dir CDMC/experiments/v2/results/pilot_regular
    $PY CDMC/experiments/v2/checks/verify_kernel.py --data baselines/data/darcy_v2/generated/pilot_regular --proposal-dir CDMC/experiments/v2/results/pilot_regular --setup-dir CDMC/experiments/v2/results/final128_regular

计时字段中 apply_ms_sample 是方法本身调用；end_to_end_ms_sample 额外包括该方法所需的在线系统与 NO，mesh_plan_ms_once 与 proposal_startup_ms_once 单列。逐样本热态口径不含训练、数据生成，也不等于批量吞吐。质量加权误差对照的是同一 P1 离散真值。
