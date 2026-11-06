# CDMC-PDE v1：固定预算神经—数值后求解器的定位与实验框架

版本：v1。日期：2026-09-21。

> 更新说明：NORM 主实验后的方法级误差审计已形成 v1.1 preview。Irregular Darcy 不再固定为 32-mode 单层代理，而采用容量受控的嵌套/分带层级；Heat 校正改为 compact shared-fixed-point apply。完整数字、失败分支与保留理由见 `../../experiments/norm_main_v1/METHOD_LEVEL_AUDIT_AND_V11.md`。

## 0. 定位结论

CDMC-PDE v1 的主身份是：

> **以神经算子预测为起点、以真实离散缺陷为驱动、以固定数值骨架执行有限步修正的固定预算后求解器。**

英文定位为 **fixed-budget neural--numerical post-solver** 或 **complementary-defect corrector**。

CDMC 当前不是神经预条件器方法。预条件器是它在稳态线性系统中的一个内部数学作用；只有当该作用被放进 PCG、FCG 或 FGMRES 并由外层 Krylov 迭代到给定容差时，才构成预条件器求解模式。

因此论文主实验回答的是：

\[
\text{给定固定且很小的在线预算，NO 后面怎样修正得最准、最快、最稳定？}
\]

而不是：

\[
\text{哪个预条件器能用最少 Krylov 迭代达到 }10^{-8}\text{ residual？}
\]

后一个问题保留为稳态线性任务的次要 solver-mode 实验，用于连接神经预条件器文献，但不决定本文主叙事。

## 1. 当前方法在数学上到底是什么

### 1.1 稳态：固定步数的学习型缺陷校正

离散问题为

\[
Ax=b.
\]

神经算子先预测

\[
x_0=N_\theta(\xi),
\qquad
r_0=b-Ax_0.
\]

CDMC 在实际 NO 缺陷分布上训练 setup 网络。setup 至少读取算子与几何；当前实现还可选地读取初始缺陷特征，产生一个固定多层校正作用

\[
d_k=B_\psi(A;z_0)r_k,
\qquad
r_k=b-Ax_k.
\]

其中 $z_0$ 表示可选的初始缺陷特征。严格消融表明，方法的 NO 特定性主要来自训练任务与在线作用对象都是 predictor-induced defect，而不是必须把 $r_0$ 作为 setup 的显式输入通道。

当前 SPD Darcy 实现使用能量最优标量

\[
\alpha_k=\frac{r_k^{\mathsf T}d_k}{d_k^{\mathsf T}Ad_k},
\qquad
x_{k+1}=x_k+\alpha_kd_k,
\]

并只执行固定的 \(K=1\) 或 \(K=2\) 步。

这在数值方法中最接近：

- defect correction / iterative refinement；
- 以学习型多层作用作为方向生成器的 preconditioned steepest descent；
- 固定深度的 nonlinear stationary iteration。

它不是 PCG，因为没有共轭方向递推；不是 FCG，因为没有 Krylov 正交化；不是 tolerance-driven solver，因为推理深度在运行前已经固定。

若把 \(B_\psi r_k\) 作为 PCG/FCG 每轮的预条件方向，则同一模块可以进入 solver mode。此时它才应使用迭代数、time-to-tolerance 和 condition reduction 作为主指标。

### 1.2 时域：闭合对齐的固定数值 rollout 后求解器

给定历史 \(\mathcal H_n\)，NO 产生未来时间块

\[
Z^0=N_\theta(\mathcal H_n).
\]

廉价数值推进器为 \(\Phi_c\)。高精度数据与该粗推进器之间的离散闭合为

\[
c_i^*=u_{i+1}^*-\Phi_c(u_i^*).
\]

闭合模型一次输出全部时间边闭合

\[
(c_0,\ldots,c_{m-1})=C_\phi(\mathcal H_n,Z^0),
\]

最终轨迹由固定数值传播产生

\[
\widehat u_{i+1}=\Phi_c(\widehat u_i)+c_i.
\]

它属于 closure-corrected numerical rollout，也属于 post-solver，而不是预条件器。NO 提供未来结构，闭合修复粗离散的系统偏差，数值推进器负责因果传播和稳定结构。

### 1.3 两个实例如何统一

二者共同遵循

\[
\boxed{
\text{NO proposal}
\rightarrow
\text{aligned discrete defect}
\rightarrow
\text{fixed-budget numerical correction}
}.
\]

稳态的 aligned defect 是 \(b-Ax\)；时域的 aligned defect 是时间边闭合后的推进缺陷。统一对象是后求解过程，不是某一种特定矩阵预条件器。

## 2. 与参考论文的准确关系

### 2.1 最接近：残差/误差后校正

Residual-Based Error Correction for Neural Operator Accelerated Infinite-Dimensional Bayesian Inverse Problems 和相关 variational corrector 在 NO 后根据真实 residual 求解线性化误差方程。PhysicsCorrect 缓存 Jacobian/伪逆，并对 NO rollout 做训练无关的线性化残差修正。

它们与 CDMC 一样，主要工作点是：

\[
\text{NO prediction}\rightarrow\text{physics defect}\rightarrow\text{post-correction}.
\]

CDMC 与它们的关键区别是：

1. CDMC 不在线完整求解一个 tangent PDE，而是用一次 defect-conditioned setup 构造固定预算的数值校正；
2. 稳态校正通过多层数值作用执行，模型学习的是有限预算如何路由，而不是直接输出完整误差场；
3. 时域先对齐数据生成离散与部署推进器，再执行固定因果传播；
4. CDMC 显式研究 \(K=0,1,2\) 的 accuracy--latency 前沿，而不是只追求校正后的最高精度。

所以 PhysicsCorrect 和 residual-based error corrector 是主相关工作和主混合基线。

### 2.2 内部数值结构参考：learned AMG 与神经多重网格

Learning Algebraic Multigrid Using Graph Neural Networks 学习 prolongation；RL AMG coarsening 学习 coarse/fine splitting；MGCFNN、MgNO 等工作把多层结构与神经网络结合。

CDMC-S 使用多层数值作用，但学习目标不同：

\[
\text{learned AMG：为 }A\text{ 学通用层级；}
\]

\[
\text{CDMC：在 }r_{NO}\text{ 的实际分布上，为固定预算后校正学习缺陷路由。}
\]

learned AMG 是 CDMC-S 的结构基线，不是 CDMC-PDE 的整体类别。

### 2.3 相邻但不同：神经预条件器与 FCG-NO

Learning Preconditioners for Conjugate Gradient PDE Solvers 学习 approximate factorization，作为 PCG 预条件器；FCG-NO 每轮调用神经算子近似 \(A^{-1}r\)，并由 FCG 达到严格残差容差。

这类方法的主问题是

\[
\min T_{\mathrm{solve}}(\varepsilon)
\quad\text{s.t.}\quad
\frac{\|Ax-b\|}{\|b\|}\leq\varepsilon.
\]

CDMC 主问题是

\[
\min E_{\mathrm{field}}(K)
\quad\text{s.t.}\quad
K\in\{0,1,2\},
\quad
T\text{ fixed and small}.
\]

因此预条件器论文不能直接作为主实验模板。它们进入一个独立的 solver-mode 表：把 CDMC 的多层作用嵌入 PCG/FCG，测试它是否还能减少 time-to-tolerance。

### 2.4 NO 架构论文：提供提议器，不决定方法类别

FNO、HPM、LRSA、NORM、MgNO 的主体目标是提高一次 solution-operator prediction 的精度、效率和泛化。CDMC 可以使用这些模型作为 \(N_\theta\)，但创新点不是重新设计一个更强 backbone。

主实验必须固定同一个 NO 权重，分别测试 NO 和 NO+CDMC；不同 NO 之间的论文数字放在独立 backbone 表中，不能用更强 NO 掩盖后校正器本身的贡献。

## 3. 论文定位与核心创新

建议论文主题：

> **Complementary-Defect Post-Solvers: Fixed-Budget Neural--Numerical Refinement for PDE Operators**

### 创新一：以 NO 诱导的互补缺陷分布定义后校正任务

若 NO 近似逆为 \(B_N\)，校正作用为 \(B_C\)，组合算子为

\[
B_H=B_N+B_C(I-AB_N),
\]

并满足

\[
I-AB_H=(I-AB_C)(I-AB_N).
\]

CDMC 学习的是 \(I-AB_N\) 产生的实际误差子空间，而不是完整 \(A^{-1}\) 或仅由 \(A\) 决定的通用预条件器。该定义体现在三个位置：训练样本来自 NO 实际 proposal、在线 numerical apply 作用于真实 \(r_0=b-Ax_0\)、目标直接监督最终场误差。setup 是否显式读取 residual feature 是次级实现选择。

### 创新二：将离散闭合作为 post-solver 的必要对齐层

高精度标签未必是廉价部署离散的根。CDMC 先学习

\[
c_\phi\approx \mathcal D_c(u^*),
\]

再修正

\[
\widetilde{\mathcal D}_c(u)=\mathcal D_c(u)-c_\phi.
\]

这把“求解误差”和“离散模型偏差”分开，使 residual-based correction 在数据 benchmark 上具有正确目标。

### 创新三：一次 setup、多次廉价 apply 的固定预算设计

网络只根据算子和 NO 缺陷运行一次；后续一至两次校正复用同一 setup。该设计直接优化 correction gain per millisecond，而不是每个外层迭代都重新调用大型网络。

### 创新四：统一稳态 defect correction 与时域 closure rollout

CDMC-S 和 CDMC-T 共享同一后求解接口：proposal、aligned defect、fixed numerical apply。统一性体现在求解过程，而不是把所有 PDE 强行输入一个 checkpoint。

## 4. 研究问题

### RQ1：固定在线预算下，CDMC 是否优于现有后校正方式？

比较 NO、固定纯数值 refinement、直接神经误差校正、PhysicsCorrect-style tangent correction 与 CDMC。主判断是同一 wall-clock 区间中的 relative field error、P95 和稳定性。

### RQ2：收益是否来自 NO 诱导的互补缺陷任务？

保持网络输入、容量和数值骨架相同，对比 generic inverse-probe training 与真实 NO-defect-distribution training。另以 operator-only 和 operator+residual-feature 作为次级诊断，区分“训练任务特定性”和“显式输入条件化”。

### RQ3：离散闭合对齐是否决定后校正方向？

比较 raw residual correction、aligned residual correction 和 oracle closure。检验 residual 降低与 field error 改善何时一致、何时分离。

### RQ4：同一 post-solver 原理能否跨稳态、时域、规则网格和非结构网格成立？

不要求共享权重；要求在不同 PDE 中保持相同流程、相同基线逻辑和相同固定预算收益。

### 次要 RQ：CDMC 的校正作用能否兼作 Krylov 预条件器？

只在稳态线性任务上测试。该结果连接 Li-2023 和 FCG-NO，但不改变 CDMC 的主身份。

## 5. 数据集重新选择

规则 Darcy 不再作为论文主结果。它保留为数学机制、损失和 cycle 消融的便宜 sandbox。

主实验使用三个能够展示 post-solver 特点、且服务器已经具备的数据集。

| 数据集 | 结构 | post-solver 数值骨架 | 它证明什么 |
|---|---|---|---|
| NORM Irregular Darcy | 2290 点、4338 三角形、1000/200 | 根据 MeshNodes、MeshElements 和系数场组装 FEM/graph elliptic operator；固定 AMG/MG correction | 非结构网格、稳态椭圆、几何图校正 |
| NORM Heat Transfer | 7199 点、34132 四面体、100/100 | 128 维 LBO/有限元热传播或稳态扩散粗骨架，加 boundary lifting 和 closure | 三维不规则网格、低样本、强压缩数值骨架 |
| Navier--Stokes | 64×64、\(\nu=10^{-5}\)、1000/200、20 帧 | 固定谱 Oseen 粗推进 | 非线性时域、长程稳定、离散闭合 |

### 5.1 Irregular Darcy 是新的稳态主任务

它比规则 85×85 Darcy 更能展示多层图校正的作用：网格非结构、节点度不规则、固定双线性 prolongation 不再天然占优。实验首先用存储解审计 FEM 离散根一致性；若数据生成离散与重建 FEM 不一致，使用与时域相同的 closure alignment，而不是强行把 residual 当成 field error。

### 5.2 Heat Transfer 是低样本三维主任务

该数据只有 100 个训练样本，输出位于 7199 个三维节点，并提供四面体网格及 128 维 LBO 基。NO 负责从边界状态给出完整温度场提议；低秩 LBO/FEM 热传播作为廉价数值骨架；CDMC 学习该粗骨架相对高精度数据的闭合和互补模态。

这比继续堆 Darcy 样本更能展示：CDMC 是否能用物理结构降低数据需求，并修复 NO 在低样本下遗漏的高频/边界影响。

### 5.3 NS 是时域主任务

NS 保留，但实验从“已有 10 帧数字”升级为标准 fixed-budget post-correction：同一 FNO rollout、同一粗推进器、同一 horizon，比较不同后校正方式，并报告场误差、对齐缺陷和时间分解。

### 5.4 规则 Darcy 的角色

规则 Darcy 只承担：

1. 线性误差乘积分解验证；
2. operator-only 与 defect-conditioned setup 消融；
3. \(K=0,1,2\) 和 setup reuse 消融；
4. solver mode 的 PCG/FCG time-to-tolerance 实验；
5. 原始 LRSA 标签与 A-consistent 标签的双真值分析。

它不再承担跨 PDE、复杂几何或最终主表优势的证明任务。

## 6. 主实验设计

### 6.1 Main-A：固定预算后校正主表

在 Irregular Darcy、Heat Transfer 和 NS 三个任务上统一比较：

1. **NO**：相同冻结 backbone 的原始输出；
2. **Numerical-only**：相同廉价数值骨架，从标准初值或最后真实状态执行相同固定预算；
3. **NO + fixed numerical refinement**：NO 初值后执行同样的固定数值步骤，但 setup 不学习；
4. **NO + direct neural corrector**：输入 NO 输出和 defect，直接预测最终场误差；参数量与 CDMC setup 匹配；
5. **NO + tangent/residual corrector**：PhysicsCorrect 或 residual-based linearized correction 的可复现版本；
6. **NO + CDMC**：完整方法，快档和精度档；
7. **High-accuracy numerical reference**：在算子可访问的任务上求到统一严格容差，作为精度和时间端点。

主表的核心是固定预算，不按样本自适应停止。

#### 主要指标

- relative L2：mean、median、P95、max；
- 梯度/通量误差或能量误差；
- aligned defect，而不是只报 raw residual；
- 边界误差与守恒/稳定性；
- \(T_{NO}\)、\(T_{setup}\)、\(T_{apply}\)、\(T_{total}\)；
- batch=1 latency 与吞吐 batch；
- 峰值显存、参数量、训练样本和训练时间；
- correction gain：

\[
G_T=\frac{E_{NO}-E_{method}}{T_{method}-T_{NO}}.
\]

#### 统计协议

- 三个训练种子；
- 同一设备、同一精度、同一 batch、充分 warm-up；
- CUDA event 计时，报告中位数和 P95 latency；
- 主表中所有 learned corrector 使用相同 NO checkpoint；
- 外部论文数字单列，不与本机端到端时间混在一起。

### 6.2 Main-B：泛化与困难设置

只做与 post-solver 机制直接相关的三种泛化：

1. **几何/网格泛化**：Irregular Darcy 在重网格或不同图划分上测试；
2. **低样本泛化**：Heat Transfer 使用 25、50、100 个训练样本；
3. **horizon 泛化**：NS 在训练窗口之外继续 rollout，报告 10、20 和可用更长 horizon 的误差增长率。

该实验检验数值骨架是否让 corrector 比直接神经误差头更稳，而不是简单增加网络容量。

### 6.3 Main-C：稳态 solver mode，作为次表

只在规则 Darcy 和 Irregular Darcy 上，将同一个 CDMC 多层作用作为预条件器放进 PCG/FCG。

比较：

- CG；
- Jacobi-PCG；
- AMG-PCG；
- learned-preconditioner PCG；
- FCG-NO；
- CDMC-PCG/FCG。

报告：

- 达到 \(10^{-4}\)、\(10^{-6}\)、\(10^{-8}\) relative residual 的迭代数；
- setup、单步 apply、总 solve time；
- 多 RHS 摊销；
- 跨网格尺寸和系数分布泛化。

这张表回答 CDMC 是否还能作为预条件器使用，但正文主结论仍来自固定预算 post-solver。

## 7. 核心消融，只保留四组

### Ablation 1：去掉 NO 特定训练任务

使用完全相同的 operator-only setup、参数量、数值骨架和预算，对比：

1. generic low/mid/high-frequency inverse probes，训练 $Ae\mapsto e$；
2. 真实 NO proposal 留下的误差分布。

然后附加 residual-feature on/off 诊断，判断显式缺陷通道是否必要。该设计把“针对 NO 误差子空间训练”与“给网络增加 residual 输入”严格分开。

### Ablation 2：去掉数值执行骨架

用等参数 direct error head 直接输出 \(\Delta u\)，对比 CDMC 的 setup+numerical apply。证明收益来自结构化后求解，而不是多一个神经网络。

### Ablation 3：去掉离散闭合对齐

比较 raw residual、learned aligned defect 和 oracle closure。Darcy 双真值、Heat Transfer 粗谱传播和 NS 时间边均报告 field error 与 defect 的对应关系。

### Ablation 4：固定预算与 setup 复用

比较：\(K=0\)、一次 setup 后 \(K=1,2\)、每步重新调用网络、以及同时间预算的纯数值 refinement。证明一次 setup、多次廉价 apply 的效率来源。

宽度、层数、学习率不进入主消融，只在 validation 上固定。

### 7.1 已完成的严格消融结果

完整协议、三种子结果和可视化见 CDMC/experiments/core_ablation_v1/ABLATION_REPORT.md。核心数字如下。

| 被移除或替换的部分 | 对照 | mean field error | 审计结论 |
|---|---|---:|---|
| NO 特定训练任务 | generic inverse probes | 0.21079% vs 0.15347% | NO-induced task 相对降低 27.2% |
| 数值执行骨架 | 等参数 direct error head | 0.83796% vs 0.15158% | direct head 误差为完整方法的 5.53 倍 |
| 离散闭合 | NS no closure vs learned closure | 8.33996% vs 6.39230% | 对齐决定时域修正方向 |
| setup 复用 | $K=4$ refresh vs reuse | 2.950 ms vs 1.833 ms | refresh 慢 60.9%，精度近似 |
| 显式 residual features | operator-only vs full | 0.15347% vs 0.15158% | 只改善 1.23%，降级为可选增强 |

这组消融修正了原始假设：complementary defect 的核心不是 residual 输入通道本身，而是训练分布、在线作用对象和最终监督目标共同围绕实际 NO 缺陷定义。

### 7.2 NORM 两项主实验结果

完整协议与结果见 `CDMC/experiments/norm_main_v1/MAIN_REPORT.md`，机器可读汇总见 `CDMC/results/v1/norm_main/summary.json`。

| 任务 | HPM NO | CDMC fast | CDMC accurate | 延迟变化 |
|---|---:|---:|---:|---:|
| NORM Irregular Darcy | 0.7461% | 0.5396%（K=1） | 0.4840%（K=2） | 0.4156 → 0.4890 / 0.5942 ms |
| NORM Heat Transfer | 0.03235% | 0.02014% | — | 0.2790 → 0.2845 ms |

Darcy 使用五折 OOF HPM proposal 训练 corrector。OOF proposal 误差为 0.7675%，与官方测试 HPM 的 0.7461% 对齐；CDMC route 在 77% 测试样本上优于固定 K=1。该实验进一步把创新点收紧为 **predictor-generalization-defect training**：corrector 必须观察未被 base NO 训练过的样本，训练内 proposal 不能替代部署缺陷。

Heat Transfer 使用 100 个训练样本，三个 corrector seed 的 CDMC mean error 为 0.02014%，相对同权重 HPM 降低 37.73%，P95 降低 49.85%，延迟仅增加 1.95%。

Darcy 的全局 32-mode closure 未通过主结果：它仅覆盖约 6.3% 的真实 closure，并略逊于 route-only。故 v1 的跨任务统一对象是 aligned defect 和固定预算数值 apply，而不是强制每个 PDE 共享相同的 closure 参数化。

## 8. 每个方法类别应该用什么指标

| 方法身份 | 主指标 | 次指标 |
|---|---|---|
| NO | field error、推理时间、样本效率 | resolution/generalization |
| post-corrector / CDMC 主模式 | 固定预算 field error、correction gain、总延迟、稳定性 | defect、能量/通量、P95 |
| neural preconditioner / solver mode | time-to-tolerance、迭代数、setup/apply、收敛因子 | field error、condition estimate |
| learned AMG | asymptotic/effective convergence factor、operator complexity | setup cost、跨规模泛化 |
| time corrector | horizon-wise field error、长期稳定性、每帧时间 | closure error、守恒误差 |

这张表是实验设计的硬约束：主模式不得拿 PCG 迭代数代替固定预算 field accuracy；solver mode 也不得只报告一两步 field improvement 而回避严格容差。

## 9. 最终论文主表和图

### Table 1：三数据集固定预算主结果

三个 panel：Irregular Darcy、Heat Transfer、NS。列出 NO、Numerical-only、NO+fixed numerical、direct corrector、tangent corrector、CDMC-fast、CDMC-accurate。报告 field mean/P95、aligned defect、总时间、显存。

### Table 2：稳态 solver mode

规则/不规则 Darcy 上的 PCG/FCG time-to-tolerance。该表负责与 Learning Preconditioners 和 FCG-NO 对话。

### Table 3：核心消融

四组消融统一报告固定预算收益和时间，验证 NO-induced defect task、数值骨架、closure alignment、setup reuse 四个创新部件。

### Figure 1：统一 accuracy--latency Pareto

三个数据集分别作图，不混合单位。NO 是左端点，高精度数值求解是右端点，CDMC 快/精度档应形成中间非支配点。

### Figure 2：误差模态和校正方向

稳态画低/中/高频或 LBO 模态误差；时域画逐 horizon field error。对比 direct corrector、tangent corrector 和 CDMC。

### Figure 3：field error 与 raw/aligned defect

展示 raw residual 下降不一定带来 field error 下降，而 closure alignment 恢复二者一致性的过程。

### Figure 4：setup reuse 与 correction gain

横轴额外时间，纵轴相对 NO 的误差下降，比较 fixed numerical、repeated neural corrector 和 CDMC。

## 10. 论文叙事

1. 强 NO 很快，但一次预测留下的误差具有结构；传统求解器可靠，但完整收敛成本高。
2. 现有 NP 研究 time-to-tolerance，现有 residual corrector 通常求完整 tangent correction；两者之间缺少固定预算的结构化后求解器。
3. CDMC 用 NO 诱导的互补缺陷分布定义后校正任务，并通过一次 setup 配置固定数值骨架。
4. 离散闭合把数据生成模型与廉价部署算子对齐，使 defect correction 真正指向 field accuracy。
5. Irregular Darcy、3D Heat Transfer 和 NS 分别证明非结构稳态、低样本三维场和非线性长程动力学。
6. 规则 Darcy 用于机制消融和 solver-mode 连接，不再承担整篇论文的主要证据。

最终贡献应概括为：

> CDMC 提出了一类位于 one-shot neural operator 与 tolerance-driven neural preconditioned solver 之间的 fixed-budget post-solver。它以 NO 诱导的互补缺陷分布为学习对象，以离散闭合保证目标一致，以一次 setup 和固定数值 apply 构造可控的 accuracy--latency 工作点。

## 11. 参考工作分组

**主相关：后校正**

- Cao et al., Residual-Based Error Correction for Neural Operator Accelerated Infinite-Dimensional Bayesian Inverse Problems, JCP 2023。
- Huang and Perdikaris, PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations, AAAI 2026 / arXiv:2507.02227。

**稳态 solver-mode 对照**

- Li et al., Learning Preconditioners for Conjugate Gradient PDE Solvers, ICML 2023。
- Rudikov et al., Neural Operators Meet Conjugate Gradients: FCG-NO, ICML 2024。

**多层结构参考**

- Luz et al., Learning Algebraic Multigrid Using Graph Neural Networks, ICML 2020。
- Greenfeld et al., Optimization-Based AMG Coarsening Using Reinforcement Learning, NeurIPS 2021。
- He et al., MgNO, ICLR 2024。
- Xie et al., MGCFNN, ICLR 2025。

**NO backbone 与数据协议参考**

- Li et al., FNO。
- Yue et al., HPM, ICML 2025。
- Yang et al., LRSA, 2026。
- Chen et al., NORM, 2024。
