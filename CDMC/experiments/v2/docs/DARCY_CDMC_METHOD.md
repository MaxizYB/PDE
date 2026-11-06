# CDMC v2：Darcy 方程上的完整方法

当前版本入口：本文保留早期Darcy pilot的装配/末层配置。2026-10-10改进后的完整主线、一般PDE接口、无PCG多尺度训练、64项快速传递及实验结果，统一见[CDMC v2方法与实验](CDMC_V2_UNIFIED_METHOD.md)。

本文件对应当前可运行的 v2 pilot。统一主干是：神经初值 → 同网格离散系统 → NO 缺陷条件化的图神经设置 → 固定次数 Galerkin 多级校正 → 完整解场。Darcy 适配器负责把系数、边界和三角网格变成主干所需的离散算子作用、右端、节点特征与场重建规则；算子可以显式装配，也可以按单元矩阵无关执行。规则方形域和凹口不规则域使用同一套 Darcy 适配器与主干算法，仅几何数据不同。当前已实现并验证的是稳态、线性、正定的 Darcy 离散系统；其他 PDE 要提供相应适配器，并核对多级求解器的适用条件。

| 流程位置 | 主干保持不变的操作 | Darcy 适配器提供的内容 |
| --- | --- | --- |
| 数据进入 | 调用初值模型的网格/物理输入接口 | 三角网格、节点系数、边界延拓、NO 特征 |
| 形成方程 | 接收自由度系统 $A x=b$ | P1 装配、Dirichlet 消元、自由节点与质量 |
| 多级计划 | 聚合、$P_\ell,R_\ell$、Galerkin 关系 | 参考 Darcy 刚度与固定拓扑装配映射 |
| 一次学习设置 | 从 $r_0=b-Ax_0$ 计算图门控 | 系数、刚度对角、坐标、质量及图 |
| 固定轮数求解 | 共用 V-cycle、能量步长、固定 $K$ | 把自由节点解放回完整压力场 |

## 1. 物理问题与统一的线性接口

Darcy 压力 $u$ 满足

$$-\nabla\cdot(a(\mathbf x)\nabla u(\mathbf x))=1,\qquad \mathbf x\in\Omega,$$

$$u(\mathbf x)=g(\mathbf x),\qquad \mathbf x\in\partial\Omega.$$

$a>0$ 是空间变化的介质系数。对边界为零的测试函数 $v$，乘以 $v$、积分并分部积分，得到弱形式

$$\int_\Omega a\nabla u\cdot\nabla v\,\mathrm d\mathbf x
=\int_\Omega v\,\mathrm d\mathbf x.$$

在三角网格上用线性节点基函数 $\phi_i$ 展开 $u_h=\sum_i u_i\phi_i$。三角形 $t$ 上，$\nabla\phi_i$ 为常量，系数取三顶点均值 $\bar a_t$；局部刚度和单位源项载荷分别是

$$K^t_{ij}=|t|\bar a_t\nabla\phi_i\cdot\nabla\phi_j,\qquad
f^t_i=|t|/3.$$

把局部贡献按节点相加得到 $Ku=f$。记边界节点集合为 $D$，自由节点集合为 $I$。已知边界值 $g_D$ 后直接消元：

$$A x=b,\qquad A=K_{II},\qquad b=f_I-K_{ID}g_D,\qquad x=u_I.$$

最终完整场由 $u_I=x$、$u_D=g_D$ 恢复。在正系数和 Dirichlet 约束下，当前 $A$ 对称正定。**主干接口**接收网格、$A$ 的作用、$b$、节点质量、模型特征以及满足边界的神经初值 $x_0$；原后端还接收显式矩阵。**Darcy 适配器**负责产生这些量，不让训练真值和在线缺陷落在不同的离散系统上。

## 2. Darcy 适配器：两种几何，一种物理离散

数据生成规范在项目根目录的 baselines/data/darcy_v2/SPEC.md。规则网格是单位方形的三角剖分；不规则网格是顶部有凹口的相容三角剖分。两者都用上述 P1 有限元；规则域并未另写有限差分。输入含顶点坐标 $V$、三角形连接 $T$、节点系数 $a$、边界掩码和边界延拓 $g$。节点积分质量为 $m_i=\sum_{t\ni i}|t|/3$。

给 NO 的五个节点通道是

$$z_i=\left((a_i-7.5)/4.5,\quad g_i/0.05,\quad
v_{i,x},\quad v_{i,y},\quad\chi_{D,i}\right).$$

给图设置网络的五个自由节点通道是在细层对角线已经得到后构造的：

$$q_i=\left(\log a_i/\log 12,\quad
\log(A_{ii}/\operatorname{median}_j A_{jj}),\quad
v_{i,x},\quad v_{i,y},\quad
\log(m_i/\operatorname{median}_j m_j)\right).$$

两组输入用途不同。当前系数取 $3$ 或 $12$，源项为 $1$；这些归一化常数对应当前数据分布，拓展参数范围时应重新标定。边界可为零或小幅平滑非零值，消元公式不变。

训练和评价的主真值 $u_h^\star$ 是**同一输出网格、同一 $Ax=b$** 的稀疏直接解。可另存精细网格参考评估空间离散误差，但它不替代校正损失的 $u_h^\star$：本网络只能修正当前离散系统内的代数误差。

## 3. 神经初值：可替换的 Geo-FNO 风格 NO

当前初值模型用节点输入 $z_i$、坐标和质量一次预测完整场。线性层先将五维输入提升至宽度 $c=24$。两层、宽度 $32$、tanh 激活的坐标 MLP 给出小幅几何变形

$$\Phi_\theta(v_i)=v_i+0.15\,d_\theta(v_i),$$

其末层零初始化。对保留波数 $k$ 做质量加权的非均匀傅里叶分析：

$$\widehat h_k=\sum_i\frac{m_i}{\sum_jm_j}h_i
\exp\{-2\pi\mathrm i\,k\cdot\Phi_\theta(v_i)\}.$$

系数放到 $32\times32$ 潜在格点并经逆 FFT 得到潜在场。四个串联傅里叶块各自对保留的正、负频带使用可学习复数通道矩阵；频域输出逆 FFT 后，与 $1\times1$ 通道混合及恒等路径相加，再经 GELU。处理后的潜在场重新 FFT，取对应模态，以共轭相位综合回原节点；综合特征与初始节点特征拼接，经节点 MLP（$48\to64\to1$）输出修正 $\delta u_i$。最后精确施加 Dirichlet 条件：

$$u_{0,i}=g_i+(1-\chi_{D,i})\,s\,\delta u_i,$$

其中 $s$ 是训练集目标减边界延拓的绝对值 $95$ 百分位尺度。自由节点部分就是 $x_0$。规则与不规则数据共享**网络结构**，但分别训练权重；几何 OOD 使用已训练的不规则 ID 权重。

NO 按质量加权相对场误差训练。权重冻结后，同一网格上的坐标变形、傅里叶相位和归一化质量只需准备一次；CUDA Graph 可重放该固定形状前向。它仅改变执行方式，不改模型权重或数学预测；换网格需重新准备。

## 4. 固定网格计划与 Galerkin 系统的两种执行

对于同一网格上的多个系数样本，先用常系数 $a\equiv1$ 得到参考矩阵 $A^{\rm ref}$。根据其非对角强连接形成每组最多两节点的确定性聚合；设 tentative prolongation 为 $Q_\ell$，用受谱上界约束的 Jacobi 平滑构造

$$P_\ell=\bigl(I-\omega_\ell^P(D_\ell^{\rm ref})^{-1}A_\ell^{\rm ref}\bigr)Q_\ell,
\qquad R_\ell=P_\ell^\top.$$

$P_\ell$ 固定于网格计划，并允许带符号条目；不按行强制归一化。计划还保存单元到 CSR 的散射位置、边界消元位置、传递结构、图与质量。约千节点规则样本的自由节点层级是 $961\to481\to241\to121$，粗层停止阈值为 $128$。

新系数样本到来后，两种执行都以当前样本的 P1 离散细层作用 $A_0(a)$ 和右端 $b(a,g)$ 为准。对应的 Galerkin 层级定义为：

$$A_{\ell+1}(a)=R_\ell A_\ell(a)P_\ell.$$

原装配路径 evaluate 显式生成 $A_0(a)$，逐层计算 $A_{\ell+1}(a)=R_\ell A_\ell(a)P_\ell$，并在最粗层当次求解。CUDA gpu_hybrid 后端对小层采用稠密 GPU 运算；CPU 路径作为同一离散的参考实现。

新编译路径 evaluate-compiled 不保存当前样本的整层矩阵。细层 $A_0(a)v$ 按 P1 单元作用计算，粗层 $A_\ell(a)v$ 则沿固定稀疏传递链先延拓、施加细层作用、再限制：

$$A_\ell(a)v=R_{\ell-1}\cdots R_0 A_0(a)P_0\cdots P_{\ell-1}v.$$

该式与显式 Galerkin 矩阵的作用代数相同。为避免每样本计算中间层精确对角线，Jacobi 平滑使用参考层级尺度与当前细层对角线的平方传递近似：

$$\widetilde d_0(a)=\operatorname{diag}A_0(a),\qquad
\widetilde d_{\ell+1}(a)=d_{\ell+1}^{\rm ref}\odot
\frac{(P_\ell\odot P_\ell)^\top\widetilde d_\ell(a)}
{(P_\ell\odot P_\ell)^\top d_\ell^{\rm ref}}.$$

最粗层不以参考矩阵代替当前样本矩阵；它通过上述精确作用和参考逆做固定两步 PCG。参考层级仅用于对角尺度、阻尼和这个固定预条件器。由此不需要在线完整矩阵装配，也不需要离线稠密累计投影基；同一个网格计划可供多个系数样本重放。新旧执行后端的训练权重一致，数值平滑细节则不同。

## 5. 一次图神经设置：学习 NO 遗留缺陷的分配

从 NO 自由节点初值计算同一离散系统的真实缺陷

$$r_0=b-A_0x_0.$$

按 Jacobi 对角和样本内均方根归一化：

$$\widetilde r_i=
\frac{r_{0,i}/\sqrt{A_{0,ii}}}
{\max\{\sqrt{|I|^{-1}\sum_{j\in I}r_{0,j}^2/A_{0,jj}},\,10^{-7}\}}.$$

CDMC 的输入为 $[q_i,\widetilde r_i]$，共六维。线性层升至宽度 $32$；两轮图消息传播分别计算节点自项和邻居项：

$$h_i^{(l+1)}=h_i^{(l)}+
\operatorname{GELU}\left(W_l^{\rm self}h_i^{(l)}
+W_l^{\rm nbr}\sum_jG_{ij}h_j^{(l)}\right),\quad l=0,1.$$

$G$ 由 $|A^{\rm ref}|$ 去掉对角后行归一化得到。输出线性层产生两个 logit，经 sigmoid 得到有界正尺度：

$$s_i=0.5+\sigma(z_i^{\rm smooth}),\qquad
c_i=0.5+\sigma(z_i^{\rm coarse})\in(0.5,1.5).$$

输出层零初始化，初始设置接近单位门控。$s$ 调整平滑器局部步长，$c$ 调整粗尺度缺陷的进入和返回。只根据**初始** NO 缺陷做一次 setup，后续固定 $K$ 轮复用；NO 与 CDMC 分别训练，不共享编码器。逐层门控采用非负传递平均：

$$s_{\ell+1}=\frac{|R_\ell|s_\ell}{|R_\ell|\mathbf1},
\qquad c_{\ell+1}=\frac{|R_\ell|c_\ell}{|R_\ell|\mathbf1}.$$

分母仅与网格有关，预先计算。物理缺陷仍用带符号的 $R_\ell=P_\ell^\top$ 限制；$|R_\ell|$ 仅用于门控尺度平均。

## 6. 共用 V-cycle、能量步长和场重建

从 $x_k$ 得到 $r_k=b-A_0x_k$。求方向 $d_k=B_{s,c}(r_k)$ 时，每个非末层先做两次加权 Jacobi 预平滑。若该层右端为 $r_\ell$，一次更新是

$$y\leftarrow y+\omega_\ell\,s_\ell\odot D_\ell^{-1}(r_\ell-A_\ell y),$$

原装配路径的 $\omega_\ell$ 根据该样本该层矩阵的谱上界设置，上限 $0.67$；编译路径复用参考层级阻尼，并以 $\widetilde d_\ell$ 代替精确 $D_\ell$。剩余缺陷 $\rho_\ell=r_\ell-A_\ell y$ 仍由当前样本的 Galerkin 作用计算并送入粗层：

$$r_{\ell+1}=R_\ell(c_\ell\odot\rho_\ell).$$

递归至最粗层时，原路径直接解 $A_Le_L=r_L$；编译路径使用当前样本精确 $A_L$ 作用与参考逆做固定两步 PCG。延拓回来后更新

$$y\leftarrow y+c_\ell\odot(P_\ell e_{\ell+1}),$$

再做两次同式后平滑得到 $d_k$。固定 MG 对照使用**同一个** V-cycle，只令 $s=c=\mathbf1$；CDMC 学习的是同一求解器的两类设置，不是替换 $A,P,R$ 或粗层方程。

由于 $A$ 对称正定，采用离散能量 $\mathcal E(x)=\tfrac12x^\top Ax-b^\top x$ 调整步长。展开 $\mathcal E(x_k+\alpha d_k)$ 并对 $\alpha$ 求导，得到

$$\alpha_k^\star=\frac{r_k^\top d_k}{d_k^\top A d_k},\qquad
\alpha_k=\operatorname{clip}_{[0,2]}(\alpha_k^\star),\qquad
x_{k+1}=x_k+\alpha_k d_k.$$

当前评价固定 $K=1$ 或 $2$，无误差阈值触发的变次数循环。将最终 $x_K$ 放回自由节点、边界保留 $g_D$，得到完整场 $u_K$。

## 7. 训练目标、对照与时间口径

先训练 NO，再冻结它。CDMC 只在实际 NO 初值诱导的缺陷上训练，目标同时覆盖一步和两步：

$$\mathcal L_{\rm CDMC}=\mathbb E\left[
\tfrac12\operatorname{Rel}_m(u_1,u_h^\star)
+\tfrac12\operatorname{Rel}_m(u_2,u_h^\star)\right],$$

$$\operatorname{Rel}_m(u,u_h^\star)=
\sqrt{\frac{\sum_i m_i(u_i-u_{h,i}^\star)^2}
{\sum_i m_i(u_{h,i}^\star)^2}}.$$

NO 使用相同的质量加权场损失，作用于 $u_0$。相对误差乘 $100\%$ 报 mean/P95。对照依次是 NO 单次预测、从零初值的固定 MG、从 NO 初值的固定 MG、从 NO 初值的 CDMC，以及同网格 P1-FEM 稀疏直接解。判断图 setup 收益时，必须比较**相同 NO 初值、相同层级和相同 $K$** 的固定 MG 与 CDMC。

完整单样本热态时间包含在线物理信息、NO（若用到）和校正；网格计划及 CUDA Graph 首次准备另列一次性成本。原装配路径的基线与消融见 [EXPERIMENTS.md](EXPERIMENTS.md)，编译路径的当前结果见 [CURRENT_STAGE.md](CURRENT_STAGE.md)。

## 8. 扩展边界

换 NO 时保持“网格/物理输入 $\mapsto$ 满足边界的 $u_0$”契约；换后校正器时保持“$A,b,x_0$ 与网格信息 $\mapsto x_K$”契约；换几何只需生成同格式网格并重新准备网格计划。换 PDE 时，适配器需定义自由度、边界消元或约束、离散矩阵与缺陷、节点/自由度质量、误差与场重建。当前 V-cycle 和能量步长使用 SPD 假设；非线性问题可经外层线性化后接入合适线性子问题，非 SPD 问题则需更换相应的稳定线性内核，不能未经验证直接沿用此公式。
