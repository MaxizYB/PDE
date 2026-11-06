# Darcy v2 自生成数据规范

本数据不是 FNO/NORM 官方测试集的复制品，而是**决定性参数相近、可完整复现数值算子**的成对稳态基准。生成脚本为 [`generate.py`](generate.py)。规则方形域和不规则凹口域从同一母随机场采样，采用相同的 P1 三角有限元装配、单位源项与 Dirichlet 消元；因此方法之间不需要两种物理残差定义。

## 物理问题与原基准的关系

方程为 $-\nabla\cdot(a\nabla u)=1$。系数是 Neumann 谱基底中的 Gaussian random field 经零阈值后取 $a_{\rm low}$ 或 $a_{\rm high}$。默认取 $3/12$，保持与 FNO 原 Darcy 相同的 4:1 对比度；NORM irregular Darcy 为 $4/12$、另一相关尺度。原 FNO 的真值用规则网格有限差分，NORM 使用 MATLAB PDE 工具箱；此处**统一改为 P1-FEM**，所以新数据上的数字不能直接当作原论文测试集结果。

规则域是 $[0,1]^2$，节点排成 $n\times n$，每个方格剖成两个三角形。凹口域是方形顶部切入一个矩形槽，包含三条内侧边，三角网格由约束 Delaunay 剖分得到。默认不规则网格最大单元面积为 `0.00022`，节点数接近 NORM 的 2290 量级；具体节点数由生成器记录。`--boundary zero` 给两种几何相同零 Dirichlet 边界；`--boundary smooth` 给每样本不同的低幅度仿射边界值。复杂度不仅由节点数决定，还需报告系数对比度、相关尺度、凹口尺寸和边界分布。

同一 `--seed` 与同一全局 `sample_index` 在两种几何中生成**同一母系数场**；训练、验证、测试使用互不重复的样本索引。这样可以在基本相同的介质变化下观察几何影响，而不是混入采样分布差别。

## 文件格式和目标

- `mesh.npz`：`points[N,2]`、`cells[T,3]`、`boundary[N]`、质量集中的节点积分权重 `mass[N]`。
- `train.npz`、`val.npz`、`test.npz`：`coefficient[S,N]`、`lift[S,N]`、`solution[S,N]`、每样本相对离散残差及全局样本索引。系数在单元上取三个顶点系数均值；NO 输入的节点系数因此足以重建离散算子。
- `manifest.json`：方程、离散、网格、参数、划分、随机种子、目标定义及最大残差。
- 可选 `--refine-factor > 1` 额外生成 `fine_reference_on_mesh`；**主训练与物理残差评价使用 `solution`**，它是输出网格上同一个 P1-FEM 问题的高精度代数解。精细参考只用于独立衡量空间离散误差，不假设它满足粗网格方程。

稳态 v2 首先用固定凹口网格训练。几何 OOD 将在相同格式下生成另一个凹口尺寸/网格，并单独标注为 OOD 集；不可把新几何样本混入 ID 测试集。

## 示例

```bash
python generate.py --kind regular --out generated/regular_pilot --resolution 33 --train 64 --val 16 --test 16
python generate.py --kind irregular --out generated/irregular_pilot --max-area 0.0008 --train 64 --val 16 --test 16
```

正式规模建议规则 `--resolution 85`、不规则 `--max-area 0.00022`、`--train 1000 --val 100 --test 200`。先用小规模试验验证残差、模型接口和计时，不把 pilot 数字当正式结果。

来源：[FNO Darcy 的采样与离散](https://arxiv.org/html/2010.08895)、[NORM irregular Darcy 的采样与几何](https://arxiv.org/html/2302.08166)。
