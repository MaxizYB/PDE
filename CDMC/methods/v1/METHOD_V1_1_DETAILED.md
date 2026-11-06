# CDMC-PDE v1.1：统一神经—数值后求解器的方法说明

修订日期：2026-09-26。

本文描述 CDMC-PDE v1.1 的固定预算后求解方法。第 1–5 节建立离散问题、互补缺陷、空间多层校正和时间闭合传播；第 6 节用规则 Darcy、不规则 Darcy、Heat Transfer 和 Navier–Stokes（NS）说明具体实现。统一性体现在预测、对齐和固定预算更新的数学流程，各任务采用适合其状态空间的算子与模型。

## 1. 方法定位、研究问题与贡献

CDMC-PDE 是 **fixed-budget neural–numerical post-solver**。一个已经训练的神经算子（NO）先给出解的提议；CDMC 根据该提议留下的离散缺陷，准备一次校正结构，再执行预定次数的数值更新，输出最终场。稳态分支使用多层作用产生校正方向，并在固定深度内完成后求解。

本节的研究定位沿用 [POSITIONING_AND_EVALUATION_V1.md](POSITIONING_AND_EVALUATION_V1.md)：

| 研究问题 | 需要检验的主张 |
|---|---|
| RQ1：固定在线预算下的后校正 | 在相同 NO、相同测试集和可比运行时间下，CDMC 的场误差、尾部误差与稳定性能否优于固定数值更新或直接神经误差校正？ |
| RQ2：NO 诱导的互补缺陷 | 真实 NO 提议产生的训练缺陷、在线缺陷及最终场监督分别贡献多少？显式缺陷特征又贡献多少？ |
| RQ3：离散闭合对齐 | 当高精度数据不满足廉价部署离散时，对齐闭合能否避免后校正趋向错误的粗离散解？ |
| RQ4：跨任务的后求解原理 | 同一个后求解流程能否在稳态/时域、规则/非规则网格上成立，同时允许任务专属的离散和数值作用？ |

与之相应，方法的四项核心设计为：**NO 诱导的缺陷任务、离散闭合对齐、一次 setup 与固定次数 apply、稳态缺陷校正与时域闭合传播的统一组织**。“NO 诱导”由训练提议的生成方式、在线校正所作用的缺陷及最终场监督共同定义；显式缺陷特征是其中一个可检验的模型选择。相关方法的比较关系与创新边界见上述定位文档和 [NOVELTY_AND_BOUNDARIES.md](NOVELTY_AND_BOUNDARIES.md)。

## 2. 离散问题与统一数学形式

### 2.1 连续方程到离散算子

输入实例记为 $\xi$，包含几何、网格、系数、源项、边界/初值以及时间设置。空间与时间离散将连续场转为有限维状态 $Z$。下面用通量离散和有限元说明矩阵 $A$、右端 $b$ 的来源。

对稳态扩散型方程，取

$$
-\nabla\!\cdot\!\bigl(a(\mathbf s)\nabla u(\mathbf s)\bigr)=f(\mathbf s)
\quad(\mathbf s\in\Omega),\qquad
u=g\quad(\mathbf s\in\partial\Omega).
$$

先定义通量 $\mathbf q=-a\nabla u$。方程变为局部守恒律 $\nabla\cdot\mathbf q=f$：一个网格控制区域内产生的源量等于穿过边界的总通量。规则网格在相邻节点 $i,j$ 的公共面上近似 $\mathbf q\cdot\mathbf n\approx-a_{ij}(x_j-x_i)/h$；再把四个面的通量除以控制区域面积并相加。令 $w_{ij}=a_{ij}/h^2$，即可得到

$$
(Ax)_i=\sum_{j\sim i,\ j\notin\partial\Omega}w_{ij}(x_i-x_j)
       +\sum_{j\sim i,\ j\in\partial\Omega}w_{ij}x_i,\qquad
b_i=f_i+\sum_{j\sim i,\ j\in\partial\Omega}w_{ij}g_j .
$$

当前规则 Darcy 实现取相邻系数的算术平均作为 $a_{ij}$；零 Dirichlet 情况下 $g=0$，实验右端 $b_i=1$。每个 $x_i$ 只与四邻值耦合，故 $A$ 是稀疏矩阵；边界上的已知值被移至右端 $b$。

对三角网格，采用分片线性有限元（P1-FEM）。取满足零边界条件的测试函数 $v$，将连续方程乘以 $v$ 并在区域上积分；分部积分把二阶导数转到测试函数上，得到弱形式

$$
\int_\Omega a\nabla u\cdot\nabla v\,\mathrm d\mathbf s
=\int_\Omega fv\,\mathrm d\mathbf s .
$$

令 $\varphi_i$ 为节点基函数、$u_h=\sum_jx_j\varphi_j$，依次令 $v=\varphi_i$，就得到矩阵行 $\sum_jA_{ij}x_j=f_i$，其中每个三角形贡献

$$
A_{ij}=\sum_{T}\int_T a\,\nabla\varphi_i\cdot\nabla\varphi_j\,\mathrm d\mathbf s,
\qquad
f_i=\int_\Omega f\varphi_i\,\mathrm d\mathbf s .
$$

把已知 Dirichlet 节点集合记作 $B$，未知内部节点集合记作 $I$。消去固定边界值 $g_B$ 后，真正更新的是

$$
A_{II}x_I=b_I,\qquad b_I=f_I-A_{IB}g_B .
$$

有限差分/通量离散与 FEM 给出不同的 $A$，但都提供同一种操作：给定 $x$，计算 $Ax$、右端 $b$ 以及缺陷 $b-Ax$。本文的“构造算子”或“离散编译”是按网格、系数和边界准备这些作用及其稀疏结构。

时域问题可写为 $\partial_tu=\mathcal G(u;\xi)$。固定粗推进器 $\Phi_c$ 接收当前帧 $z_i$ 并推进至下一观测时刻；$z_0$ 为已知最后历史帧。一个未来 $m$ 帧的离散时间块满足 $m$ 条边关系：

$$
d_i(Z;\xi)=z_{i+1}-\Phi_c(z_i;\xi)=0,\quad
i=0,\ldots,m-1,\qquad Z=(z_1,\ldots,z_m).
$$

空间算子 $A$ 和时间推进器 $\Phi_c$ 是任务适配器提供的不同数值对象。时域更新不必显式形成一个全局时空矩阵。

### 2.2 统一状态、提议与对齐缺陷

将部署时用于检查状态的离散关系统记为 $\mathcal D_c(Z;\xi)$；线性稳态可取 $\mathcal D_c(x)=Ax-b$，时域则取所有 $d_i$ 的集合。下标 $c$ 表示在线使用的廉价离散骨架。冻结 NO 产生初始提议

$$
Z^0=N_\theta(\xi).
$$

若标签 $Z^*$ 满足同一部署离散，即 $\mathcal D_c(Z^*;\xi)\simeq0$，可以直接处理提议留下的缺陷。在线性稳态且 $Ax^*=b$ 时，定义右端缺口 $r_0=b-Ax_0$，就有

$$
A(x^*-x_0)=r_0 .
$$

高精度标签来自另一空间离散、更细时间步或不同数值模型时，可以出现 $\mathcal D_c(Z^*;\xi)\ne0$。这意味着即使把部署方程解到零残差，得到的也可能是粗模型的解，而非数据标签。用标签在粗模型上的离散缺口定义闭合

$$
c^*(\xi)=\mathcal D_c(Z^*;\xi),
$$

再用训练所得的 $c_\phi(\xi,Z^0)$ 近似它，建立对齐缺陷

$$
\widetilde{\mathcal D}_\phi(Z;\xi)
=\mathcal D_c(Z;\xi)-c_\phi(\xi,Z^0).
$$

于是标签满足 $\widetilde{\mathcal D}_\phi(Z^*;\xi)\simeq0$。训练时可由标签计算 $c^*$，在线时由模型预测 $c_\phi$；当数据和部署离散一致时取 $c_\phi=0$。对齐之后的缺陷描述的是**在目标离散关系下仍需修正的状态误差**。

### 2.3 互补缺陷的组合关系

考虑线性稳态 $Ax^*=b$。NO 给出 $x_0$ 后，当前场误差 $e_0=x^*-x_0$ 满足 $Ae_0=b-Ax_0=r_0$。设本次 setup 确定一个校正作用 $B_C$，令 $d_0=B_Cr_0$，则

$$
x_0=N_\theta(\xi),\qquad
x_1=x_0+d_0=x_0+B_Cr_0.
$$

把新场重新代入同一方程，得到一步之后的缺陷：

$$
r_1=b-Ax_1=b-A(x_0+B_Cr_0)
   =r_0-AB_Cr_0=(I-AB_C)r_0.
$$

这一步是逐样本的确切关系：校正器处理的是 NO 已留下的 $r_0$。为了把这一关系写成算子组合，进一步考虑 NO 对右端 $b$ 可表示为线性近似逆 $B_N$ 的情形，即 $x_0=B_Nb$。这时

$$
x_1=\bigl[B_N+B_C(I-AB_N)\bigr]b
=:B_Hb,
$$

因此 $r_0=(I-AB_N)b$。将它代入刚才的 $r_1=(I-AB_C)r_0$，两次作用相接得到

$$
I-AB_H=(I-AB_C)(I-AB_N).
$$

右侧先由 $I-AB_N$ 产生 NO 剩余缺陷，再由 $I-AB_C$ 缩减它。这是线性情形的算子表达；对实际 FNO/HPM，方法直接使用每个样本计算出的 $r_0$，并以该 NO 的部署误差分布训练后校正器。

对非线性离散关系，仍可沿用“先处理一部分，再处理剩余部分”的构造。记当前对齐缺陷为 $D_0=\widetilde{\mathcal D}_\phi(Z^0;\xi)$，并在 $Z^0$ 处定义 Jacobian $J=\partial_Z\widetilde{\mathcal D}_\phi(Z^0;\xi)$。小位移 $\delta$ 的一阶展开为

$$
\widetilde{\mathcal D}_\phi(Z^0+\delta;\xi)
\simeq D_0+J\delta .
$$

先用廉价作用 $B_0$ 取 $\delta_0=-B_0D_0$。代入展开式，第一次更新后剩余的线性化缺陷为

$$
D_1\simeq D_0+J\delta_0=(I-JB_0)D_0.
$$

再以作用 $C$ 取 $\delta_1=-CD_1$，总位移便是

$$
\delta_0+\delta_1
=-\bigl[B_0+C(I-JB_0)\bigr]D_0.
$$

这给出了复合校正作用 $B_0+C(I-JB_0)$ 的推导。v1.1 的空间多层、模态固定点和时间闭合分别实现“先有可靠骨架，再处理其剩余误差”的结构。

### 2.4 固定预算

在线深度在推理前确定。稳态和模态实例使用 $K=0$（纯 NO）、$K=1$（快速档）、$K=2$（复用 setup 的精度档）。时域实例用固定的时间边数量和粗推进内部次数定义预算。每个档位具有固定的算子调用次数和可单独测量的总延迟。

## 3. 统一后求解流程与任务适配接口

CDMC 的统一过程为

$$
\xi
\;\longrightarrow\;
\text{离散算子/粗推进器}
\;\longrightarrow\;
Z^0=N_\theta(\xi)
\;\longrightarrow\;
\text{对齐缺陷}
\;\longrightarrow\;
\mathcal S
\;\longrightarrow\;
\text{固定预算更新}
\;\longrightarrow\;
Z^{\rm out}.
$$

$\mathcal S$ 称为 **setup**，是本样本后续更新要复用的准备结果。它包含任务所需的神经路由或闭合、固定传递算子、按样本装配的 Galerkin 粗矩阵或模态目标；不同任务按其状态空间选取相应组成。后续的 **apply** 按照 $\mathcal S$ 执行一次空间多层更新、一次模态收缩，或一条/一组时间边推进。NO、setup 和 apply 各自计时。

初值网络 $N_\theta$ 在各实例中保持冻结，但其结构也决定提议中剩余误差的分布。规则 Darcy 和 NS 使用 FNO：输入场与坐标编码后，由频域低模态混合、逐点通道混合和输出投影组成；NS 的自回归结构见第 6.4 节。不规则 Darcy 使用 HPM 的不规则网格版本：节点输入先经 MLP 编码，再由固定网格 LBO 基底支持的 calibrated spectral mixer 与残差 FFN 交互，最后投影到节点场。Heat 使用 HPM 的跨域版本：在输入网格的谱基底上编码与混合，通过一个输入域到输出域的谱变换块传送隐藏表示，随后在输出网格上继续混合并重建场。这些 NO 负责一次提议；CDMC 的学习模块和数值作用从各自提议留下的误差出发，具体结构见第 6 节。

接入一个新任务时，任务适配器将原始数据和离散方式转换为下表对象，从而确定该任务的可计算缺陷、基础数值作用与约束。

| 适配对象 | 对后求解器的实际含义 | 使用时机 |
|---|---|---|
| 状态、网格和边界 | 自由度排列、邻接/单元、坐标、固定边界及输出重建 | 离线准备与在线输入 |
| 部署缺陷 | 给定状态计算 $\mathcal D_c$；线性稳态为 $Ax-b$ | NO 后及每次必要更新后 |
| 基础数值作用 | 细层平滑/粗层更新或时间推进器 $\Phi_c$ | 固定次数 apply |
| 传递与约束 | 空间层级的限制/延拓、谱投影、边界/守恒投影 | setup 和 apply |
| 闭合目标 | 对训练标签计算 $\mathcal D_c(Z^*)$，判断是否需要学习对齐 | 训练；在线仅用预测闭合 |
| 误差与成本指标 | 场误差、P95、梯度/能量/守恒、setup/apply/总时间 | 验证与测试 |

稳态分支中，NO 提供便宜的初值，多层数值骨架给出有界而可复用的更新方向，规则网格路由网络分配有限的局部与粗层作用。时域分支中，NO 提供未来结构，闭合补偿粗推进器的系统偏差，数值推进器保持时间因果关系。模态分支以相同的“提议—对齐目标—固定更新”组织，其部署骨架是跨网格谱映射与固定点更新。

## 4. 稳态固定预算缺陷校正：CDMC-S

### 4.1 数值多层骨架

对离散线性系统 $Ax=b$，当前解为 $x_k$，当前缺陷 $r_k=b-Ax_k$。一次空间校正产生方向 $d_k$ 并更新

$$
d_k=B_{\rm multi}(r_k;\mathcal S),\qquad
x_{k+1}=x_k+\alpha_kd_k,\qquad k=0,\ldots,K-1.
$$

$B_{\rm multi}$ 的数值来源是多重网格（MG）。它从**误差方程**出发。当前全局解 $x_k$ 与精确离散解 $x^*$ 的差为 $e^*=x^*-x_k$。由 $Ax^*=b$ 和 $r_k=b-Ax_k$，直接相减得

$$
Ae^*=A(x^*-x_k)=b-Ax_k=r_k.
$$

因此一次校正的目标是在很少的计算步骤内近似求解 $Ae=r_k$。在以下层内推导中，$e$ 专指“本次 cycle 已累计的校正”，初值为零；它与未知的真实误差 $e^*$ 区分开。

先采用阻尼 Jacobi 作为局部平滑器。令 $D=\operatorname{diag}(A)$，一步为

$$
e^{(t+1)}
=e^{(t)}+\omega D^{-1}\!\left(r_k-Ae^{(t)}\right).
$$

其作用频段可用一维 Poisson 的离散矩阵说明。该矩阵第 $j$ 个正弦模态的特征值为 $\lambda_j=4h^{-2}\sin^2(j\pi/[2(n+1)])$，而 $D=2h^{-2}I$；在该模态上的误差放大因子是

$$
\rho_j
=1-\omega\lambda_j/(2h^{-2})
=1-2\omega\sin^2\!\left(\frac{j\pi}{2(n+1)}\right).
$$

较大的 $j$ 对应网格尺度振荡，$|\rho_j|$ 可以显著小于 1；较小的 $j$ 对应长波误差，$\rho_j$ 接近 1。这个计算解释了“平滑器处理局部误差、粗层处理长程误差”的分工。对其他系数和网格，具体谱值变化，但多层方法仍依赖这一互补作用。

现在考虑一个细层 $\ell$，记本层右端为 $r_\ell$、本层算子为 $A_\ell$。做 $\nu_{\rm pre}$ 次平滑后得到 $e_\ell^{\rm pre}$。剩余的真实误差是 $e_\ell^*-e_\ell^{\rm pre}$；其右端由代数关系直接算出：

$$
\widehat r_\ell
=r_\ell-A_\ell e_\ell^{\rm pre}
=A_\ell(e_\ell^*-e_\ell^{\rm pre}).
$$

取延拓矩阵 $P_\ell\in\mathbb R^{n_\ell\times n_{\ell+1}}$，使粗层向量 $e_{\ell+1}$ 在细层表示为 $P_\ell e_{\ell+1}$。若要求剩余方程在粗空间所有测试方向上成立，则

$$
P_\ell^{\mathsf T}
\!\left[\widehat r_\ell-A_\ell P_\ell e_{\ell+1}\right]=0.
$$

将项整理，便得到 **Galerkin 粗层方程**

$$
A_{\ell+1}e_{\ell+1}=r_{\ell+1},\qquad
A_{\ell+1}=P_\ell^{\mathsf T}A_\ell P_\ell,\qquad
r_{\ell+1}=P_\ell^{\mathsf T}\widehat r_\ell.
$$

对于对称正定 $A_\ell$，这也等价于在所有 $P_\ell e_{\ell+1}$ 中最小化剩余误差的 $A_\ell$ 能量范数。令 $e_{\rm rem}=e_\ell^*-e_\ell^{\rm pre}$，写出粗空间的能量目标

$$
\mathcal E(e_{\ell+1})=	frac12(e_{\rm rem}-P_\ell e_{\ell+1})^{\mathsf T}
A_\ell(e_{\rm rem}-P_\ell e_{\ell+1}).
$$

对 $e_{\ell+1}$ 求导并令导数为零，有

$$
P_\ell^{\mathsf T}A_\ell P_\ell e_{\ell+1}
=P_\ell^{\mathsf T}A_\ell e_{\rm rem}
=P_\ell^{\mathsf T}\widehat r_\ell,
$$

正好得到前述 Galerkin 方程。规则网格还可采用重新离散的粗算子与归一化限制 $R_\ell$；此时粗层方程写成 $A_{\ell+1}e_{\ell+1}=R_\ell\widehat r_\ell$，保留同一“把剩余缺陷交给粗层”的求解流程。

粗层解返回后，先加入延拓的校正 $e_\ell^{\rm mid}=e_\ell^{\rm pre}+P_\ell e_{\ell+1}$，再以新的缺陷 $r_\ell-A_\ell e_\ell^{\rm mid}$ 做 $\nu_{\rm post}$ 次 Jacobi 后平滑。后平滑处理延拓时产生的局部振荡。一次层递归因此为

$$
\begin{aligned}
e_\ell^{\rm pre}
&=\operatorname{Smooth}_{\ell}^{\nu_{\rm pre}}(0,r_\ell),\\
\widehat r_\ell&=r_\ell-A_\ell e_\ell^{\rm pre},\\
r_{\ell+1}&=R_\ell\widehat r_\ell,\qquad
e_{\ell+1}=V_{\ell+1}(r_{\ell+1}),\\
e_\ell^{\rm mid}&=e_\ell^{\rm pre}+P_\ell e_{\ell+1},\\
V_\ell(r_\ell)
&=\operatorname{Smooth}_{\ell}^{\nu_{\rm post}}
(e_\ell^{\rm mid},r_\ell).
\end{aligned}
$$

当 $\ell=L$ 时直接解最粗系统 $A_Le_L=r_L$。由细层递归下降到最粗层，再逐层返回，构成一个 V-cycle。规则实例在每个非末层采用 $\nu_{\rm pre}=\nu_{\rm post}=2$；当前不规则图实例每层各采用一次。上面推导说明了数值骨架每一步的来源，模型在下一节只调整其有限预算内的作用分配。

### 4.2 可学习 setup 与误差传播

若任务训练学习型 setup，网络可根据算子、几何和 NO 缺陷输出局部平滑门 $g_\ell^s$、粗层路由门 $g_\ell^c$ 与样本级阻尼 $\omega$。把它们置入上节相同的误差方程，相应 apply 变为

$$
\begin{aligned}
e_\ell&\leftarrow e_\ell+\eta_\ell D_\ell^{-1}(r_\ell-A_\ell e_\ell),
&\eta_\ell&=\min(\omega g_\ell^s,\eta_{\max}),\\
r_{\ell+1}&=R_\ell(g_\ell^c\odot\widehat r_\ell),
&e_\ell&\leftarrow e_\ell+g_\ell^c\odot P_\ell e_{\ell+1},
\end{aligned}
$$

其中 $\odot$ 是逐节点乘法。模型的数学作用可继续从真实误差推导。先只看一个细层和一个精确求解的粗层，并暂略后平滑：令 $S_\psi=\operatorname{diag}(\eta)D^{-1}$，则预平滑后的误差为 $e^{\rm pre}=(I-S_\psi A)e^*$。令 $G_c=\operatorname{diag}(g^c)$，网络路由后的粗方向作用为 $C_\psi=G_cPA_c^{-1}RG_c$；将该方向施加到预平滑后的缺陷，得到

$$
e^{\rm after}
=\underbrace{(I-C_\psi A)(I-S_\psi A)}_{E_\psi}e^*.
$$

固定 MG 对应位置无关的 $S=\omega D^{-1}$ 和 $G_c=I$。因此模型通过 $S_\psi$ 改变局部误差的衰减，通过 $G_c$ 改变剩余误差进入与返回粗空间的权重。实际实现将同一机制递归到多层并加入后平滑；第 6.1 节给出网络如何产生这两个门。

NO 特定性来自在真实 NO 提议造成的缺陷分布上训练、在线处理该缺陷并监督最终场。规则网格分支学习上述门；不规则 Darcy 分支保持图层级固定，其互补作用由图空间和终端谱空间共同构成。

### 4.3 对称正定系统的方向校准

在正系数扩散问题及相应 Dirichlet 处理下，内部自由度矩阵 $A$ 为对称正定（SPD）。沿已得到的方向 $d_k$，二次能量 $E(x)=\tfrac12x^{\mathsf T}Ax-b^{\mathsf T}x$ 展开为

$$
E(x_k+\alpha d_k)
=E(x_k)-\alpha r_k^{\mathsf T}d_k
 +\tfrac12\alpha^2 d_k^{\mathsf T}Ad_k.
$$

这里利用了 $\nabla E(x_k)=Ax_k-b=-r_k$。求导令零，得到

$$
\alpha_k=\frac{r_k^{\mathsf T}d_k}{d_k^{\mathsf T}Ad_k}.
$$

分母对非零方向为正，故该 $\alpha_k$ 是当前方向上的唯一能量最小点。CDMC 以此校准一次多层方向，然后重新计算缺陷以进入下一固定 cycle。场监督继续决定训练目标，因为实验评价的是最终场的相对 $L^2$ 误差。

### 4.4 训练目标

设训练标签为 $x^*$，在固定校正预算后得到 $x_K$。规则 Darcy 的最终路由器在冻结 NO 的条件下，以场误差为主监督，并用能量和残差约束方向：

$$
\begin{aligned}
\mathcal L_{\rm field}
 &=\frac{\|x_K-x^*\|_2^2}{\|x^*\|_2^2},\\
\mathcal L_A
 &=\frac{(x_K-x^*)^{\mathsf T}A(x_K-x^*)}{(x^*)^{\mathsf T}Ax^*},\\
\mathcal L_r
 &=\frac{\|Ax_K-b\|_2^2}{\|b\|_2^2},\\
\mathcal L
 &=\mathcal L_{\rm field}+0.005\mathcal L_A+10^{-5}\mathcal L_r .
\end{aligned}
$$

场项直接衡量最终预测目标，能量项约束算子加权的误差，残差项辅助离散稳定性。三项共同训练，使方向质量由最终场而非单一残差指标决定。当前规则 Darcy 训练针对一次校正；第二个 cycle 复用同一 setup，作为固定预算扩展评估。

## 5. 时域固定预算闭合传播：CDMC-T

### 5.1 时间边与对齐闭合

给定历史 $\mathcal H_n$，冻结 NO 提出未来块 $Z^0=(z_1^0,\ldots,z_m^0)$。设连续演化为 $\partial_tu=\mathcal G(u;\xi)$，相邻两个观测时刻相距 $\Delta t$；精确演化满足积分关系

$$
u(t_{i+1})
=u(t_i)+\int_{t_i}^{t_{i+1}}\mathcal G(u(t);\xi)\,\mathrm dt.
$$

廉价数值推进器 $\Phi_c$ 用固定粗步长和固定内部计算近似右侧积分。高精度数据相对它的剩余部分，定义为一条时间边的目标闭合

$$
c_i^*=u_{i+1}^*-\Phi_c(u_i^*),\quad i=0,\ldots,m-1,\qquad u_0^*=u_0.
$$

所以 $u_{i+1}^*=\Phi_c(u_i^*)+c_i^*$ 是由定义得到的精确分解：$\Phi_c$ 承担已解析的粗尺度演化，$c_i^*$ 收集该推进器相对高精度演化的离散差。网络以历史和 NO 未来块为条件，一次预测全部边的 $\widehat c_i=C_\phi(\mathcal H_n,Z^0)_i$。对齐后的时间边关系为

$$
\widetilde d_i(Z)=z_{i+1}-\Phi_c(z_i)-\widehat c_i=0.
$$

在线轨迹沿这一关系因果递推：

$$
\widehat u_0=u_0,\qquad
\widehat u_{i+1}=\Phi_c(\widehat u_i)+\widehat c_i .
$$

NO 未来块为闭合网络提供整段时间的上下文，$\Phi_c$ 逐帧传递当前状态。时间块长度 $m$、推进器内部迭代数和稳态 cycle 数 $K$ 分别是不同预算参数。

### 5.2 部署一致的闭合训练

先用真值相邻帧形成 $c_i^*$，训练闭合场的空间结构。然后把网络放进实际递推链微调，因为第 $i$ 条在线边的父状态是上一条边生成的 $\widehat u_i$。这个差异可由误差传播式直接看到。令 $e_i=\widehat u_i-u_i^*$，$\varepsilon_i=\widehat c_i-c_i^*$，则

$$
e_{i+1}
=\Phi_c(\widehat u_i)-\Phi_c(u_i^*)+\varepsilon_i
\simeq J_{\Phi_c}(u_i^*)e_i+\varepsilon_i.
$$

$J_{\Phi_c}$ 是粗推进器对输入状态的局部导数。第 $i$ 条边的误差既受本边闭合误差 $\varepsilon_i$ 影响，也包含先前所有边传播来的 $e_i$。因此逐边闭合误差低与整段场误差低是两个不同训练目标。定义 $\widetilde c_i^*=(c_i^*-\mu_c)/\sigma_c$，其中 $\mu_c,\sigma_c$ 来自训练集；网络输出标准化闭合 $\widehat{\widetilde c}$。第一阶段目标为

$$
\mathcal L_{\rm closure}
=\operatorname{MSE}(\widehat{\widetilde c}-\widetilde c^*)
+\lambda_{\rm spec}\operatorname{MSE}_{\Omega_H}
\!\left(\mathcal F(\widehat{\widetilde c})
-\mathcal F(\widetilde c^*)\right),
$$

其中 $\Omega_H$ 是空间 Fourier 高频集合；$\operatorname{MSE}(v)$ 表示 $v$ 元素平方的平均。二阶段按实际递推得到 $\widehat u_1,\ldots,\widehat u_m$，再最小化

$$
\mathcal L_{\rm roll}
=\frac1m\sum_{i=1}^{m}
 \frac{\|\widehat u_i-u_i^*\|_2^2}{\|u_i^*\|_2^2},\qquad
\mathcal L_{\rm stage2}
=\mathcal L_{\rm roll}
+\lambda_c\operatorname{MSE}(\widehat{\widetilde c}-\widetilde c^*).
$$

闭合监督学习每条时间边的局部差异，rollout 监督进一步考虑 $J_{\Phi_c}$ 引起的跨边传播。当前十帧主结果使用完整顺序因果扫描，每一帧使用刚更新的前一帧状态。

## 6. 已实现任务的具体化

以下分别给出前述统一流程在四个任务中的离散对象、NO、学习模块及固定预算更新。

### 6.1 规则网格 Darcy：缺陷路由多层校正

规则 Darcy 在 $85\times85$ 内部节点上以冻结 FNO 产生 $x_0$，根据同一五点通量算子计算 $r_0=b-Ax_0$。一次 setup 的输入是八个节点通道：

$$
h_i^0=\left[
\operatorname{norm}(\log a_i),\
\operatorname{norm}(a_i),\
\frac{r_{0,i}}{\sqrt{A_{ii}}},\
\widetilde r_i^{\rm low},\
\widetilde r_i^{\rm high},\
\operatorname{norm}(\log A_{ii}),\
p_{i,x},p_{i,y}\right],
$$

其中 $\widetilde r_i=r_{0,i}/\sqrt{A_{ii}}$，$\widetilde r^{\rm low}=\operatorname{AvgPool}_{3\times3}(\widetilde r)$，$\widetilde r^{\rm high}=\widetilde r-\widetilde r^{\rm low}$。系数、局部算子尺度、NO 缺陷及其局部平滑/变化部分共同确定路由特征；$p_{i,x},p_{i,y}$ 是归一化坐标。

DefectRouter 的结构分为四段，分别承担**物理量编码、局部信息交换、空间路由、样本级阻尼**：

1. **输入编码。** 将每个节点的八维 $h_i^0$ 作为 token；共享的 $1\times1$ 映射在每个节点把物理特征升到 24 维，并经 GELU 得到隐藏场 $H^{(0)}$。这一步只混合节点自身的通道。
2. **邻域信息交换。** 在规则网格上连续使用三个带残差连接的 $3\times3$ 卷积块。第 $t$ 块计算 $H^{(t+1)}=H^{(t)}+\operatorname{GELU}(W_t^{3\times3}*H^{(t)})$。卷积交换邻域信息，残差连接保留原有系数与缺陷特征；三个块把路由判断建立在局部缺陷形状而非单个节点的残差大小上。这里的 $*$ 表示空间卷积。
3. **空间路由输出。** 共享的输出特征经双通道逐节点线性投影，分别给出平滑 logit $q_s(i)$ 和粗层 logit $q_c(i)$。两门相互独立：前者决定局部 Jacobi 更新强度，后者同时控制剩余缺陷的限制与粗校正的延拓。
4. **全局阻尼输出。** 对隐藏场作空间平均池化，再用线性投影得到样本级 logit $q_\omega$；整个样本共用一个阻尼值，以免每个位置都单独决定迭代步长。

这相当于“节点 tokenizer—邻域消息交换—路由输出—全局控制”的完整网络，而固定的限制、延拓及 V-cycle 是网络输出所作用的数值骨架。输出约束为

$$
g^s=0.5+\sigma(q_s),\qquad
g^c=0.5+\sigma(q_c),\qquad
\omega=0.50+0.34\sigma(q_\omega),\qquad
\eta_{\max}=0.95.
$$

输出投影从零初始化开始，此时 $g^s=g^c=1$，$\omega=0.67$，对应一个可用的固定多层初态。正式配置固定双线性传递算子，学习集中在平滑与粗层作用的分配上。网络在最细层调用一次，再将空间 logits 双线性插值到其他层；同一个样本的 $K$ 次 V-cycle 复用这些路由。数值层级为

$$
85^2\rightarrow43^2\rightarrow22^2\rightarrow11^2\rightarrow6^2\rightarrow3^2.
$$

指数 2 表示二维内部节点数；各级单方向节点数为 85、43、22、11、6、3。例如 $85\times85$ 层的剩余缺陷先受 $g^c$ 加权，再限制到 $43\times43$ 层；在粗层求得方向后，双线性延拓返回 $85\times85$ 层，再受该门加权并做后平滑。粗层系数由缩放得到，再按同一通量模板**重新离散** $A_\ell$。固定双线性延拓与归一化转置型限制连接各层，每层预/后平滑各 $\nu=2$ 次，最粗 $3\times3$ 小系统直接求解。一次 setup 可供 $K=1/2$ 次 cycle 使用。

### 6.2 三角网格 Darcy：图 Galerkin 与终端 LBO

不规则 Darcy 使用 P1-FEM；实际网格有 2290 节点、4338 三角形、214 个固定 Dirichlet 节点，故内部自由度为 2076。冻结 HPM 给出提议 $x_0$，随后在 FEM 内部自由度上计算 $r=b-Ax_0$。图层级从网格和算子出发：以算子的强连接关系把相邻自由度贪心聚成较小的节点组；$T_\ell$ 将每个细层节点指派到其所属组，作为**初始延拓**。为了让粗空间也表达跨组变化，做一次 Jacobi 型平滑并逐行归一化，得到

$$
\widetilde P_\ell=(I-\omega_P D_\ell^{-1}A_\ell)T_\ell,
\qquad P_\ell=\operatorname{RowNorm}(\widetilde P_\ell),
\qquad D_\ell=\operatorname{diag}(A_\ell).
$$

这里 $T_\ell$ 是由图聚合固定的稀疏映射；平滑让组指示函数顺着当前算子的耦合关系扩展。图层级按

$$
2076\rightarrow558\rightarrow95\rightarrow14,\qquad
A_{\ell+1}=P_\ell^{\mathsf T}A_\ell P_\ell,\quad R_\ell=P_\ell^{\mathsf T}.
$$

这四个数是各层**自由度数**。每层以 $R_\ell=P_\ell^{\mathsf T}$ 把缺陷交给下一层，沿用第 4.1 节推导的 Galerkin 方程；返回时用 $P_\ell$ 把粗方向延拓回来。图 V-cycle 用固定 $\omega=0.7$ 的 Jacobi 预/后平滑，每层各一次。边界 trace 保持给定值，校正方向在边界为零。这样得到的是在原三角网格图上定义的粗空间，无须把网格切成矩形块。

图粗层仍可能遗漏少数跨全域的平滑方向。若图 V-cycle 得方向 $d_{\rm graph}$，令 $Q\in\mathbb R^{2076\times16}$ 为内部节点上的 16 个 Laplace–Beltrami operator（LBO）模态，把补充方向限制在 $\operatorname{span}(Q)$ 内，即 $d_{\rm LBO}=Qz$。要求补充后剩余缺陷在这一子空间内正交，便有

$$
Q^{\mathsf T}\bigl[r-A(d_{\rm graph}+Qz)\bigr]=0
\quad\Longrightarrow\quad
(Q^{\mathsf T}AQ)z=Q^{\mathsf T}(r-Ad_{\rm graph}).
$$

所以总方向为

$$
r_{\rm rem}=r-Ad_{\rm graph},\qquad
A_Q=Q^{\mathsf T}AQ,\qquad
d=d_{\rm graph}+QA_Q^{-1}Q^{\mathsf T}r_{\rm rem}.
$$

然后按第 4.3 节做能量步长更新。$A_Q$ 为 $16\times16$ 的投影算子。拓扑相关的层级索引预先准备，样本相关的刚度值与 Galerkin 算子按样本形成。该分支的学习模块是冻结的 HPM；图层级、LBO 补充与能量步长构成固定的数值后求解作用。

### 6.3 Heat Transfer：跨网格模态对齐与固定点

Heat 的输入/输出场定义在不同三维网格。冻结 HPM 给出输出场 $x_0$，输入和输出 LBO 基底分别为 $\Phi_{\rm in}$、$\Phi_{\rm out}$，各保留 128 模态。用伪逆将输入 $v$ 与提议 $x_0$ 投影到各自模态空间：

$$
c_{\rm in}=\Phi_{\rm in}^{+}v,\qquad c_0=\Phi_{\rm out}^{+}x_0.
$$

训练集 100 个样本拟合 ridge 仿射映射 $W$。将第 $n$ 个训练样本的输入模态加常数项记为 $\bar c_{{\rm in},n}=[c_{{\rm in},n};1]$，输出标签模态记为 $c_{{\rm out},n}^*$，则

$$
W=\arg\min_{\widetilde W}
\sum_{n=1}^{100}\|\widetilde W^{\mathsf T}\bar c_{{\rm in},n}-c_{{\rm out},n}^*\|_2^2
+\lambda_{\rm ridge}\|\widetilde W\|_F^2.
$$

它构成从输入模态到输出模态的参考目标 $c_{\rm base}=W^{\mathsf T}[c_{\rm in};1]$；当前 NO 提议与这个目标之间的差为 $d_{\rm modal}=c_{\rm base}-c_0$。因输入和输出网格不同，$c_{\rm in}$ 与 $c_0$ 分别在各自 LBO 基底中定义，映射 $W$ 才承担跨网格对应关系。这是从训练数据得到的谱参考骨架。

Heat setup 网络按“**跨网格模态表示—全局模态混合—双输出投影**”构成：

1. **模态表示。** 将标准化的 $[c_{\rm in};c_0;d_{\rm modal}]$ 拼成 384 维向量；128 个输入模态、128 个 NO 输出模态和 128 个二者的参考差在此对齐。
2. **全局模态混合。** 线性升维至宽度 64，并经过一个残差 FFN。设 $f$ 为拼接向量，则 $h_0=\operatorname{GELU}(W_{\rm in}f)$，$h_1=h_0+W_2\operatorname{GELU}\bigl(W_1\operatorname{LN}(h_0)\bigr)$。全连接混合使一个模态的判断可使用所有输入、提议和差值模态的信息。
3. **双输出投影。** 同一 $h_1$ 投影出 128 个模态 gate logits $q^g$ 和 128 个模态 closure 值 $q^c$。前者控制参考目标的采用比例，后者调整参考目标本身；两者的作用在下面的更新式中分别出现。

当前正式配置采用这个全模态 ResidualMLP，参数量为 57,984。输出层初始时 gate logit 为 2、closure 为 0：起点先利用已拟合的谱参考目标，再由训练学习各模态的收缩与调整。

令 $\lambda_j$ 为输出 LBO 特征值、$s_j$ 为训练集估计的校正尺度，网络给出原始输出 $q^g_j,q^c_j$：

$$
g_j=\sigma(q^g_j),\quad
c_{{\rm closure},j}=\frac{s_jq^c_j}{1+3\max(\lambda_j,0)},\quad
c_{{\rm align},j}=c_{{\rm base},j}+c_{{\rm closure},j}.
$$

其中 $s_j$ 是该模态的训练集校正尺度，$1/(1+3\max(\lambda_j,0))$ 使高特征值模态的 closure 幅度逐渐衰减。因为输出基底满足 $\Phi_{\rm out}c_0$ 是 NO 提议在保留模态上的投影，可将 $x_0$ 分解为低维部分 $\Phi_{\rm out}c_0$ 与其余部分 $x_0-\Phi_{\rm out}c_0$。只在低维部分从 $c^{(0)}=c_0$ 执行固定 $K$ 次更新：

$$
c^{(k+1)}=c^{(k)}+g\odot(c_{\rm align}-c^{(k)}),\qquad
x_K=\Phi_{\rm out}c^{(K)}+(x_0-\Phi_{\rm out}c_0).
$$

由于 setup 在 $K$ 次更新中保持 $g$ 和 $c_{\rm align}$ 不变，每个模态的递推可以直接展开。令 $\delta_j^{(k)}=c_j^{(k)}-c_{{\rm align},j}$，则 $\delta_j^{(k+1)}=(1-g_j)\delta_j^{(k)}$，所以

$$
c_j^{(K)}=c_{{\rm align},j}+(1-g_j)^K(c_{0,j}-c_{{\rm align},j}).
$$

这个式子解释 $K=2$ 的收益来源：再次缩减同一对齐目标上的剩余模态差，且不增加第二次网络 setup。训练对 $K=1$ 和 $K=2$ 同时监督，设 $E_K$ 为相对场 $L^2$ 误差、$G_K$ 为输出网格边差的相对 $L^2$ 误差，则实际目标为

$$
\mathcal L_{\rm Heat}
=(E_1^2+0.02G_1^2)+1.0(E_2^2+0.02G_2^2).
$$

正式数据划分为 100/100/100 个 train/validation/test 样本。HPM 冻结，模态 setup 只计算一次，$K=2$ 复用同一 gate 与目标。

### 6.4 Navier–Stokes：粗 Oseen 推进与十帧 closure

NS 实例在周期二维网格上预测**涡量场** $u$。对应连续模型为 $\partial_tu+\mathbf v(u)\cdot\nabla u=\nu\Delta u+f$，其中 $\nabla\cdot\mathbf v=0$。输入历史十帧 $\mathcal H_n=(u_{n-9},\ldots,u_n)$；冻结 recurrent FNO 自回归预测未来十帧 $Z^0$。其模型结构为：

1. **时空输入编码。** 将十张历史涡量帧和二维周期坐标的四个正弦/余弦通道拼接，用逐点线性映射提升至宽度 32 的隐藏场。周期坐标提供网格位置，同时与 Fourier 表示匹配。
2. **频域—局部混合。** 四个残差 FNO block 顺次更新隐藏场。每块对 Fourier 变换后的低频模态作可学习的通道混合，并与空间逐点的通道混合相加，经归一化和 GELU 后作为残差加入。此处每个方向保留 12 个低频模态；频域分支交换全局信息，逐点分支保留网格位置上的局部响应。
3. **单步预测与自回归。** 输出投影产生下一帧相对最新历史帧的增量；把预测帧移入历史窗口，再调用同一网络，连续生成十帧。因此 $Z^0$ 是由一个共享单步算子产生的未来轨迹。

为明确 FNO block 的作用，令 $H^{(t)}$ 为隐藏场，$\mathcal F$ 为二维 Fourier 变换，$\Pi_{12}$ 表示保留的低频模态。两种网络中的一个残差 block 均可概括为

$$
H^{(t+1)}=H^{(t)}+
\operatorname{GELU}\!\left(\operatorname{GN}\!\left[
\mathcal F^{-1}\!\bigl(M_t\,\Pi_{12}\mathcal F(H^{(t)})\bigr)
+W_t^{1\times1}H^{(t)}\right]\right).
$$

$M_t$ 在保留的 Fourier 模态上混合通道，$W_t^{1\times1}$ 在每个网格位置混合通道，$\operatorname{GN}$ 为组归一化。四块的 recurrent FNO 与三块的 closure 网络共享这种 block 形式，各自有独立参数、输入和输出投影。

固定粗推进器 $\Phi_c$ 在每个小步冻结输入涡量 $u^n$ 所诱导的对流速度 $\mathbf v_n=\mathbf v(u^n)$。对输运后的涡量采用半隐式 Oseen 离散，可写成

$$
\bigl(I+\Delta t\,\mathbf v_n\!\cdot\nabla-\Delta t\,\nu\Delta\bigr)u^{n+1}
=u^n+\Delta t\,f.
$$

这个线性方程仍耦合了对流与扩散。记 $D_{\Delta t}=(I-\Delta t\,\nu\Delta)^{-1}$；周期网格中 Fourier 模态 $\mathbf k$ 上，$D_{\Delta t}$ 只需乘以 $[1+\Delta t\,\nu|\mathbf k|^2]^{-1}$。从扩散解作为初值，在固定次数内迭代

$$
\widetilde u^{(q+1)}
=D_{\Delta t}\bigl[u^n+\Delta t f-\Delta t\,\mathbf v_n\cdot\nabla u^{(q)}\bigr],
\qquad
u^{(q+1)}=(1-\rho)u^{(q)}+\rho\widetilde u^{(q+1)},
\quad\rho=0.7.
$$

这给出粗推进器的每个 Oseen 小步；一条观测时间边可能包含多个这种小步。快/精度配置在每个小步分别固定 3 次或 4 次内部更新，故在线计算量在推理前即可确定。

closure 网络有独立而完整的“**整段条件编码—频域与空间混合—逐边输出**”结构：

1. **条件编码。** 将标准化的十张历史帧、十张 FNO 未来帧以及四个周期坐标通道拼成 24 通道场。逐点 lift 映射到宽度 24，形成每个网格位置的条件表示。历史给出已观测状态；FNO 未来块为所有闭合边提供长程时间上下文。
2. **交互主干。** 三个残差 FNO block 在整个空间场上交替使用低频 Fourier 通道混合与逐点通道混合，再经 GroupNorm、GELU；每块保留 12 个 Fourier 模态。主干由此把未来 NO 结构、历史信息和周期几何融入同一表示。
3. **时间边输出。** 逐点输出投影一次生成十个 closure 场 $\widehat c_0,\ldots,\widehat c_9$。这些场各对应一条观测时间边，进入第 5.1 节的因果推进式；十帧只进行一次 closure 网络推理，而粗推进器沿时间边依次使用它们。

训练阶段一采用第 5.2 节损失，$\lambda_{\rm spec}=0.02$；阶段二沿完整十帧在线递推微调，$\lambda_c=0.1$，该权重乘的是**标准化 closure MSE**。已保存正式结果的十帧主流程按时间顺序遍历全部时间边，以固定 3 次 Oseen 内部迭代构成快速预算，或以 4 次构成精度预算。

## 7. 离线与在线成本、分支一致性

离线阶段包括训练或加载 NO、构造训练提议与闭合标签、准备固定网格/谱结构、训练所需 setup，以及保存归一化统计和参数。在线阶段包括 NO 推理、缺陷计算、一次 setup、固定次数 apply 与最终重建。对空间或模态任务，可分解端到端时间为

$$
T_{\rm total}
=T_{\rm NO}+T_{\rm defect}+T_{\rm setup}
 +K\,T_{\rm apply}+T_{\rm reconstruct}.
$$

时间块任务的 $T_{\rm apply}$ 是固定数量时间边及其内部粗推进的总开销。稀疏空间层级每次 apply 的主要算子作用近似随 $\sum_{\ell=0}^{L}\operatorname{nnz}(A_\ell)$ 增长；层级装配、终端小系统、GPU 传输及各网络调用也计入实测总时间。Heat 的主要在线工作是模态投影、网络 setup 与低维更新；NS 则是一次闭合预测和连续时间边的谱推进。

| 分支 | NO 提议 | 对齐对象与 setup | 固定 apply | 训练中的 CDMC 部分 |
|---|---|---|---|---|
| 规则 Darcy | 冻结 FNO | $b-Ax_0$；一次 DefectRouter 输出多层路由 | 重新离散的 V-cycle，$K=1/2$ | 路由网络 |
| 不规则 Darcy | 冻结 HPM | FEM 缺陷；固定图 Galerkin 装配与 16 维 LBO | 图 V-cycle + 终端 LBO，$K=1/2$ | 当前无新增图 setup 网络 |
| Heat | 冻结 HPM | 输入/输出模态差；一次 ResidualMLP setup | 128 模态固定点，$K=1/2$ | gate 与模态 closure |
| NS | 冻结 recurrent FNO | 时间边闭合；一次 ClosureOperator setup | 固定十边因果 Oseen 传播 | 时间闭合网络 |

表中的“一次 setup”描述**每个推理样本的一次准备**；哪些固定结构可跨样本复用、哪些按系数逐样本构造，均在具体分支中说明。

## 8. 适用范围与非线性稳态扩展

四个已实现分支分别覆盖规则网格稳态椭圆、三角非规则网格稳态椭圆、不同输入/输出三维网格的模态校正，以及周期网格的非线性时域传播。它们共同验证“冻结任务 NO—任务适配缺陷/闭合—一次 setup—固定预算 apply”的方法接口。具体网络权重随任务训练，统一的是方法组织及可比较的预算定义。

一般稳态非线性问题可通过部署缺陷 $\mathcal D_c(x;\xi)$、边界/约束处理和相应的数值更新接入。例如在第 $k$ 个状态附近，线性化形式为

$$
J_k\delta x_k=-\mathcal D_c(x_k;\xi),
\qquad
J_k=\partial_x\mathcal D_c(x_k;\xi),\qquad
x_{k+1}=x_k+\eta_k\delta x_k .
$$

此时可在 NO 初值上构造固定次数的线性化更新，并将对齐缺陷训练扩展到非线性稳态标签。本文的实证范围是第 6 节的四个实例：NS 使用时域 Oseen closure，Heat 使用数据拟合的 ridge–LBO 谱参考目标；非线性稳态线性化接口给出下一类任务的具体接入形式。

对应的同骨干结果汇总在 [PRELIMINARY_VALIDATION_RESULTS_V1.md](../../experiments/main_benchmark_v1/results/PRELIMINARY_VALIDATION_RESULTS_V1.md)。以下给出各任务内部 NO 到固定预算 CDMC 的场误差变化：

| 任务与 NO | NO 原始误差 | 对应 CDMC 固定预算误差 |
|---|---:|---:|
| 规则 Darcy / FNO | 0.92375% | $K=1$: 0.15087%；$K=2$: 0.03640% |
| 不规则 Darcy / full HPM | 0.680571% | $K=1$: 0.461659%；$K=2$: 0.441846% |
| Heat / full HPM | 0.0182628% | $K=1$: 0.0148967%；$K=2$: 0.0148853% |
| NS 十帧 / FNO | 9.47102% | 完整十边传播：6.39253% |

这些结果说明四个实例均可按“同一 NO—固定数值基线—CDMC”的协议检验。关于各任务的 P95、时间、同预算纯数值对照及消融结论，应以链接中的正式报告和结果文件为准。

## 9. 推理算法

以下伪代码给出统一控制流；其中适配器提供具体离散与数值作用。稳态实例的 $\mathcal S$ 在多个 cycle 间复用，时域实例的 closure 在整个时间块中固定。

### 9.1 稳态空间层级

    operator, boundary, levels = adapter.prepare(xi)
    x = frozen_NO(xi)
    r0 = adapter.rhs(xi) - operator(x)
    setup = adapter.setup(operator, levels, x, r0)   # 一次
    for k in range(fixed_K):
        r = adapter.rhs(xi) - operator(x)
        d = adapter.multilevel_direction(r, setup)
        alpha = adapter.direction_scale(x, r, d)
        x = adapter.project_boundary(x + alpha * d)
    return adapter.reconstruct(x)

### 9.2 时域闭合传播

    proposal = frozen_NO(history)
    closure = closure_setup(history, proposal)        # 全部时间边，一次
    state = history.last_frame
    trajectory = []
    for i in range(fixed_future_length):
        state = coarse_time_step(state) + closure[i]
        trajectory.append(state)
    return trajectory

## 10. 正式实现与结果入口

| 内容 | 文件 |
|---|---|
| v1.1 任务定义与已保存结果审计 | [unified_profiles.py](unified_profiles.py) |
| 规则 Darcy 路由及多层方向 | [cdmc_dr.py](steady/cdmc_dr.py)，[METHOD_AND_RESULTS.md](steady/METHOD_AND_RESULTS.md) |
| 不规则 Darcy 图 Galerkin + LBO | [graph_galerkin_lbo.py](irregular/graph_galerkin_lbo.py) |
| Heat 模态固定点 | [compact_fixed_point.py](heat/compact_fixed_point.py)，[heat_main.py](../../experiments/norm_main_v1/heat_main.py) |
| NS 时间闭合与谱推进 | [train_space_time_closure_np.py](transient/train_space_time_closure_np.py)，[ns_spectral.py](transient/ns_spectral.py) |
| 同骨干主结果与核心消融 | [PRELIMINARY_VALIDATION_RESULTS_V1.md](../../experiments/main_benchmark_v1/results/PRELIMINARY_VALIDATION_RESULTS_V1.md)，[CORE_ABLATION_SUMMARY.md](../../experiments/main_benchmark_v1/results/ablation_core_v1/CORE_ABLATION_SUMMARY.md) |
