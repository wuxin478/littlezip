# 4. 核心算法原理

HOME-LBM (High-Order Moment-Encoded Lattice Boltzmann Method) 是一种轻量级、高精度的动能流体求解器，专为湍流不可压缩流体的视觉仿真而设计。与传统CFD中直接离散Navier-Stokes方程的方法不同，HOME-LBM从统计力学的介观动力学出发，通过演化粒子分布函数的矩（moments）而非完整的分布函数值，实现了三倍内存缩减与十倍速度提升，同时保持了与最先进LBM求解器近乎相同的数值精度。

## 4.1. 控制方程

### 4.1.1. 连续Boltzmann方程

在统计力学框架下，流体动力学由介观分布函数 $f(\boldsymbol{v}, \boldsymbol{x}, t)$ 的时间演化来描述，它表示在位置 $\boldsymbol{x}$、时刻 $t$ 找到速度为 $\boldsymbol{v}$ 的粒子的概率密度。对于单相流体，其演化的控制方程为**Boltzmann方程**：

$$\frac{\partial f}{\partial t} + \boldsymbol{v} \cdot \nabla f = \Omega(f) + \boldsymbol{F} \cdot \nabla\_{\boldsymbol{v}} f$$

其中 $\Omega$ 为碰撞项（collision term），驱使分布函数向局部热力学平衡态 $f^{eq}$ 松弛；$\boldsymbol{F}$ 为外力项。平衡态分布由**Maxwell-Boltzmann分布**给出：

$$f^{eq} = \frac{\rho}{(2\pi r T)^{d/2}} \exp\left(-\frac{|\boldsymbol{v} - \boldsymbol{u}|^2}{2rT}\right)$$

其中 $\rho$ 为密度，$\boldsymbol{u}$ 为宏观速度，$r$ 为气体常数，$T$ 为热力学温度，$d$ 为空间维数。

分布函数的前三阶速度矩（velocity moments）直接对应宏观物理量——密度 $\rho$、动量 $\rho\boldsymbol{u}$ 以及与应力张量相关的二阶张量 $\boldsymbol{S}$：

$$\rho = \int f , d\boldsymbol{v}, \quad \rho\boldsymbol{u} = \int \boldsymbol{v} f , d\boldsymbol{v}, \quad \rho S\_{\alpha\beta} = \int \left(v\_{\alpha}v\_{\beta} - \frac{1}{3}\delta\_{\alpha\beta}\right) f , d\boldsymbol{v}$$

### 4.1.2. Navier-Stokes方程恢复

当碰撞算子采用BGK（Bhatnagar-Gross-Krook）近似 $\Omega(f) = -(f - f^{eq})/\tau$ 时（$\tau$ 为松弛时间），通过Chapman-Enskog多尺度展开可以证明，Boltzmann方程在宏观尺度上恢复**不可压缩Navier-Stokes方程**：

- **连续性方程（质量守恒）**：$\nabla \cdot \boldsymbol{u} = 0$
- **动量方程**：
  $$\rho\left(\frac{\partial \boldsymbol{u}}{\partial t} + (\boldsymbol{u} \cdot \nabla)\boldsymbol{u}\right) = -\nabla p + \mu \nabla^2 \boldsymbol{u} + \boldsymbol{f}$$
- **状态方程**（理想气体）：$p = \rho c\_s^2$

其中运动黏度 $\nu$ 与松弛时间 $\tau$ 的关系为 $\nu = c\_s^2(\tau - 1/2)$，$c\_s = 1/\sqrt{3}$ 为声速（格子单位）。

### 4.1.3. 格子Boltzmann方程 (LBE)

为在计算机上求解，将时间离散为均匀步长 $\Delta t = 1$，空间离散为规则格子，速度空间则离散为有限个离散速度方向 ${\boldsymbol{c}_i}_{i=0}^{q-1}$（称为**格子结构**），从而得到**格子Boltzmann方程**：

$$f\_i(\boldsymbol{x} + \boldsymbol{c}\_i, t + 1) - f\_i(\boldsymbol{x}, t) = \Omega\_i + F\_i$$

其中 $f\_i(\boldsymbol{x}, t)$ 编码了位置 $\boldsymbol{x}$、时刻 $t$ 处第 $i$ 个方向上的分布函数值，$\boldsymbol{c}\_i$ 为离散格子速度，$\Omega\_i$ 为离散碰撞算子，$F\_i$ 为外力在分布函数空间的投影。

通过算子分裂（operator splitting），LBE可分两步求解：

**(a) Streaming（迁移/平流）**：粒子沿格子链接方向自由运动
$$f\_i^\*(\boldsymbol{x}, t) = f\_i(\boldsymbol{x} - \boldsymbol{c}\_i, t)$$

**(b) Collision（碰撞）**：局部松弛向平衡态
$$f\_i(\boldsymbol{x}, t + 1) = f\_i^\*(\boldsymbol{x}, t) + \Omega\_i + F\_i$$

宏观量的离散对应关系（含外力修正）为：

$$\rho = \sum\_{i=0}^{q-1} f\_i, \quad \rho\boldsymbol{u} = \sum\_{i=0}^{q-1} \boldsymbol{c}_i f\_i + \frac{1}{2}\boldsymbol{F}, \quad \rho S_{\alpha\beta} = \sum\_{i=0}^{q-1} \left(c\_{i\alpha}c\_{i\beta} - \frac{1}{3}\delta\_{\alpha\beta}\right) f\_i$$

***

## 4.2. HOME-LBM离散格式

传统LBM的致命弱点是巨大的内存需求：以D3Q27模型为例，每个网格节点至少需要存储58个变量（27方向分布函数$\times$2个时间副本 + 密度 + 动量），使其难以部署在消费级GPU上。HOME-LBM的核心创新在于：**仅存储前三个速度矩 $\rho$、$\rho\boldsymbol{u}$、$\rho\boldsymbol{S}$ 共10个标量，在需要时通过高阶Hermite展开动态重构27方向的分布函数值**，从而实现三倍内存缩减。

### 4.2.1. D3Q19速度离散

本项目采用 **D3Q19**（三维空间，19个离散速度方向）格子模型，在精度与效率之间取得良好平衡。D3Q19包含：

- **1个静止粒子**：$c\_0 = (0, 0, 0)$，权重 $w\_0 = 1/3$
- **6个面心方向**（一类链接）：$(\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1)$，权重 $w\_s = 1/18$
- **12个棱心方向**（二类链接）：$(\pm 1, \pm 1, 0), (\pm 1, 0, \pm 1), (0, \pm 1, \pm 1)$，权重 $w\_e = 1/36$

声速 $c\_s = 1/\sqrt{3}$，权重来自Maxwell-Boltzmann分布的Gauss-Hermite求积。

> 注：该项目同时支持D3Q15和D3Q27两种格子模型，通过编译宏 `D3Q15` / `D3Q19` 切换，以适应不同精度/性能需求。

### 4.2.2. Hermite多项式展开理论

分布函数的连续表达可通过**Hermite级数展开**（截断至 $N$ 阶）给出：

$$f(\boldsymbol{v}, \boldsymbol{x}, t) = \omega(\boldsymbol{v}) \sum\_{n=0}^{N} \frac{1}{n!} \boldsymbol{H}^{\[n]}(\boldsymbol{v}) : \boldsymbol{a}^{\[n]}(\boldsymbol{x}, t)$$

其中 $\omega(\boldsymbol{v}) = \exp(-|\boldsymbol{v}|^2/2)/(2\pi)^{d/2}$ 为权重函数，$\boldsymbol{H}^{\[n]}$ 为 $n$ 阶Hermite张量多项式，$\boldsymbol{a}^{\[n]}$ 为对应的展开系数，运算符 "$:$" 表示完全张量缩并。

Hermite多项式通过递推定义：

$$\boldsymbol{H}^{\[n]}(\boldsymbol{v}) = \frac{(-1)^n}{\omega(\boldsymbol{v})} \nabla^n \omega(\boldsymbol{v})$$

由于Hermite多项式构成正交基，系数可由加权内积求得：

$$\boldsymbol{a}^{\[n]}(\boldsymbol{x}, t) = \int f(\boldsymbol{v}, \boldsymbol{x}, t) \frac{\boldsymbol{H}^{\[n]}(\boldsymbol{v})}{\omega(\boldsymbol{v})} d\boldsymbol{v}$$

**关键性质**：前三个系数恰好对应三个速度矩：
$$\boldsymbol{a}^{\[0]} = \rho, \quad \boldsymbol{a}^{\[1]} = \rho\boldsymbol{u}, \quad \boldsymbol{a}^{\[2]} = \rho\boldsymbol{S}$$

经过格子离散化后，离散分布函数与连续分布函数的关系为：

$$f\_i(\boldsymbol{x}, t) = w\_i \frac{f(\boldsymbol{c}\_i, \boldsymbol{x}, t)}{\omega(\boldsymbol{c}\_i)}$$

### 4.2.3. 三阶矩编码重构

此前MR-LBM方法只使用二阶Hermite截断重建 $f\_i$，导致截断误差随雷诺数增大而剧烈增加，无法处理 $Re > 4000$ 的湍流。HOME-LBM采用**三阶Hermite截断**，利用Malaspinas（2015）提出的递推正则化思想，根据前三阶矩 $\rho, \rho\boldsymbol{u}, \rho\boldsymbol{S}$ 求解三阶系数 $\boldsymbol{a}^{\[3]}$，再重建分布函数。三阶重建的闭式表达为：

$$f\_i = \rho w\_i \left\[ 1 + \frac{\boldsymbol{c}\_i \cdot \boldsymbol{u}}{c\_s^2} + \frac{\boldsymbol{H}^{\[2]}(\boldsymbol{c}\_i) : \boldsymbol{S}}{2c\_s^4} + \frac{1}{2c\_s^6} \cdot (\text{三阶项}) \right]$$

其中三阶项包含7个非零分量，分别对应 $\boldsymbol{H}^{\[3]}$ 的独立分量 $xxy, xyy, xxz, xzz, yzz, yyz, xyz$，其系数为：

$$\begin{aligned}
\text{coeff}_{xxy} &= S_{xx}u\_y + 2S\_{xy}u\_x - 2u\_x^2 u\_y \\
\text{coeff}_{xyy} &= S_{yy}u\_x + 2S\_{xy}u\_y - 2u\_x u\_y^2 \\
\text{coeff}_{xxz} &= S_{xx}u\_z + 2S\_{xz}u\_x - 2u\_x^2 u\_z \\
\text{coeff}_{xzz} &= S_{zz}u\_x + 2S\_{xz}u\_z - 2u\_x u\_z^2 \\
\text{coeff}_{yzz} &= S_{zz}u\_y + 2S\_{yz}u\_z - 2u\_y u\_z^2 \\
\text{coeff}_{yyz} &= S_{yy}u\_z + 2S\_{yz}u\_y - 2u\_y^2 u\_z \\
\text{coeff}_{xyz} &= S_{xz}u\_y + S\_{yz}u\_x + S\_{xy}u\_z - 2u\_x u\_y u\_z
\end{aligned}$$

而三阶Hermite多项式的值为：

$$\begin{aligned}
H\_{xxy} &= c\_x^2 c\_y - c\_s^2 c\_y, \quad H\_{xyy} = c\_x c\_y^2 - c\_s^2 c\_x \\
H\_{xxz} &= c\_x^2 c\_z - c\_s^2 c\_z, \quad H\_{xzz} = c\_x c\_z^2 - c\_s^2 c\_x \\
H\_{yzz} &= c\_y c\_z^2 - c\_s^2 c\_y, \quad H\_{yyz} = c\_y^2 c\_z - c\_s^2 c\_z \\
H\_{xyz} &= c\_x c\_y c\_z
\end{aligned}$$

三阶重建带来了两方面收益：(1) 更精确地捕捉湍流中的高阶动力学；(2) 通过正则化过程滤除非物理的"幽灵模态"（ghost modes），显著提升数值稳定性。

### 4.2.4. Streaming与宏观矩累积

这是整个HOME-LBM求解器最核心、计算最密集的步骤。在单次kernel调用中，对每个网格节点执行以下操作：

**（1）读取邻居矩数据**

对于 D3Q19 模型，每个节点需读取19个邻居节点的矩数据 $(\rho\_{src}, \boldsymbol{u}_{src}, \boldsymbol{S}_{src})$。该步骤通过**共享内存协作加载**优化：每个work group（$4 \times 4 \times 4$ = 64线程）协作加载本block加上一层halo的全部$6 \times 6 \times 6$ = 216个网格节点的矩到共享内存。每个线程通过循环分摊策略负责加载多个节点，加载后通过 `barrier()` + `memoryBarrierShared()` 确保数据全局可见。

共享内存声明10个独立数组存储每个分量的矩数据：

```glsl
shared float shRho[SHARED_TOTAL_CELLS];   // 密度
shared float shUx, shUy, shUz[...];        // 速度分量
shared float shSxx, shSyy, shSzz, shSxy, shSxz, shSyz[...]; // 二阶矩
```

对于超出域边界的halo区域，调用 `get_moments_for_ghost()` 函数统一处理各类边界条件（详见 §4.3）。

**（2）分布函数重构与矩累积**

对于每个格子方向 $i$，从其上游邻居节点 $(\rho\_{src}, \boldsymbol{u}_{src}, \boldsymbol{S}_{src})$ 出发，用三阶Hermite展开重构流入的分布函数值 $f\_i$，然后立即累加到目标节点的宏观矩中（不保存 $f\_i$，避免中间数组占用）：

$$\begin{aligned}
\rho^\* &= \sum\_{i=0}^{q-1} f\_i \\
\rho\boldsymbol{u}^\* &= \sum\_{i=0}^{q-1} \boldsymbol{c}_i f\_i \\
\rho S_{\alpha\beta}^\* &= \sum\_{i=0}^{q-1} (c\_{i\alpha}c\_{i\beta} - c\_s^2\delta\_{\alpha\beta}) f\_i
\end{aligned}$$

这一设计实现了 **Streaming + 矩计算的一体化**——在单次循环中同时完成分布函数重构、数据迁移和矩累积，消除了传统LBM中间步的显式存储开销。

### 4.2.5. 高阶碰撞模型

原始MR-LBM采用BGK碰撞模型（一阶精度），只能处理低雷诺数流动。HOME-LBM改用**非正交中心矩多重松弛时间（NOCM-MRT）碰撞模型**。区别于传统LBM需要将分布函数`f`映射到中心矩空间执行碰撞再映射回来（涉及矩阵乘法 $\boldsymbol{M}$ 和 $\boldsymbol{M}^{-1}$），HOME-LBM利用矩编码表示的优势，推导出碰撞后速度矩的**闭式解析表达式**：

**密度与速度更新**（碰撞不改变密度，速度受半时间步外力修正）：

$$\rho(\boldsymbol{x}, t+1) = \rho^*, \quad u\_\alpha(\boldsymbol{x}, t+1) = u\_\alpha^* + \frac{1}{2\rho^\*} F\_\alpha$$

**非对角应力分量更新**：

$$S\_{xy}(\boldsymbol{x}, t+1) = \left(1 - \frac{1}{\tau}\right) S\_{xy}^\* + \frac{1}{\tau} u\_x^\* u\_y^\* + \frac{2\tau - 1}{2\tau\rho^*} (F\_x u\_y^* + F\_y u\_x^\*)$$

$S\_{xz}$ 和 $S\_{yz}$ 同理类推。

**对角应力分量更新**（以 $S\_{xx}$ 为例）：

$$\begin{aligned}
S\_{xx}(\boldsymbol{x}, t+1) = &\frac{\tau-1}{3\tau}(2S\_{xx}^\* - S\_{yy}^\* - S\_{zz}^\*) + \frac{1}{3}(u\_x^{*2} + u\_y^{2} + u\_z^{2}) \\
&+ \frac{1}{3\tau}(2u\_x^{2} - u\_y^{2} - u\_z^{2}) + \frac{1}{\rho^*} F\_x u\_x^ + \frac{\tau-1}{3\tau\rho^}(2F\_x u\_x^ - F\_y u\_y^ - F\_z u\_z^)
\end{aligned}$$

$S\_{yy}$ 和 $S\_{zz}$ 可通过对称性得到。

其中 $\tau$ 为松弛时间，与运动黏度 $\nu$ 的关系为 $\tau = 3\nu + 1/2$，$\omega = 1/\tau$ 为松弛频率。

**物理含义**：碰撞过程本质上是将非平衡应力向平衡态应力 $(\boldsymbol{u}\boldsymbol{u}^T)$ 松弛，松弛速率由 $\omega$ 控制。外力的贡献同时考虑了平衡态和非平衡态两个层面。

### 4.2.6. 体积力处理

外力 $\boldsymbol{F}$（如重力、自定义体积力）在碰撞步骤中通过矩空间的附加项自然融入，而非投影到分布函数空间。这在以下两个方面影响碰撞：

1. **速度的半时间步修正**：$u\_\alpha \leftarrow u\_\alpha + F\_\alpha / (2\rho)$
2. **应力张量的外力贡献**：对角分量和非对角分量均包含 $(1 - \omega/2) \cdot (F\_\alpha u\_\beta + F\_\beta u\_\alpha)/\rho$ 形式的修正项

外力通过 uniform buffer 动态传入着色器，支持逐时间步更新。

### 4.2.7. GPU性能优化策略

本项目的GPU实现在 [collide\_and\_stream.comp](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/collide_and_stream.comp) 中引入了多项关键优化：

**（a）Half精度（FP16）矩存储**

10个矩分量 ${\rho, \rho u\_x, \rho u\_y, \rho u\_z, S\_{xx}, S\_{yy}, S\_{zz}, S\_{xy}, S\_{xz}, S\_{yz}}$ 使用 `packHalf2x16` / `unpackHalf2x16` 压缩为5个 `uint32_t`，每节点存储从40字节降为20字节，显存带宽需求减半。FP16精度（约3.3位十进制有效数字）对LBM中间矩数据足够。布局为：

```
uint[0] = packHalf2x16(ρ, ρu_x)
uint[1] = packHalf2x16(ρu_y, ρu_z)
uint[2] = packHalf2x16(Sxx, Syy)
uint[3] = packHalf2x16(Szz, Sxy)
uint[4] = packHalf2x16(Sxz, Syz)
```

**（b）共享内存协作加载**

将同一work group内数据只读取一次到on-chip共享内存，streaming阶段直接从中读取邻居矩，避免大量冗余全局内存访问。

**（c）消除中间数组**

不保存 $f\_i$ 数组，直接在重构后累积到最终矩变量中，消除Q个float的寄存器占用和"写后读"数据依赖。

**（d）预计算常数替代运行时除法**

- $1/\tau \to$ `inv_tau`（CPU预计算，通过UBO传入）
- $1/c\_s^2 = 3.0 \to$ `INV_CS2`
- $1/(2c\_s^4) = 4.5 \to$ `INV_2CS4`
- $1/(2c\_s^6) = 13.5 \to$ `INV_2CS6`

**（e）硬编码索引常数**

共享内存索引计算使用编译期常量（如 `linear_idx / 36`、`linear_idx % 36 / 6`）而非运行时动态变量，允许编译器将其优化为乘法移位指令。

**（f）循环展开与分支精简**

使用 `#pragma unroll` 展开Q循环，将多行边界处理代码压缩为紧凑单行，减少控制流发散。

***

## 4.3. 边界条件

本求解器支持四种边界条件类型，通过UBO中6个面的独立配置参数（`bc_x0, bc_xn, bc_y0, bc_yn, bc_z0, bc_zn`）实现六面独立控制，支持运行时刻切换，无需重新编译着色器：

| 类型                   | 枚举值 | 描述       |
| -------------------- | --- | -------- |
| `BC_PERIODIC`        | 0   | 周期性边界    |
| `BC_INFLOW`          | 1   | 流入边界     |
| `BC_OUTFLOW`         | 2   | 流出边界     |
| `BC_SOLID_NO_SLIP`   | 3   | 无滑移固体壁面  |
| `BC_SOLID_FREE_SLIP` | 4   | 自由滑移固体壁面 |

边界处理在共享内存加载阶段统一完成（`get_moments_for_ghost()` 函数），后续streaming无需再次判断边界类型。

### 4.3.1. 周期性边界

炉石式（toroidal）环绕：超出网格边界即从对侧重新进入。

$$\text{coord}\_{\text{wrap}} = (\text{coord} \bmod \text{size} + \text{size}) \bmod \text{size}$$

矩数据直接从环绕后的镜像节点读取。

### 4.3.2. 流入/流出边界

**流入边界 (BC\_INFLOW)**：强制设定边界节点的矩为预设流入条件：

$$\rho = \rho\_{\text{inflow}}, \quad \boldsymbol{u} = \boldsymbol{u}_{\text{inflow}}, \quad S_{\alpha\beta} = \rho u\_\alpha u\_\beta$$

流入密度、速度通过UBO配置（`inflow_rho, inflow_vel_x/y/z`），支持运行时调节。

**流出边界 (BC\_OUTFLOW)**：从相邻内部节点复制矩数据，密度设为流出参考值，若速度指向域内则清空（防止回流扰动）。采用Neumann型零梯度近似。

### 4.3.3. 固体壁面边界

从壁面镜像位置（clamp后的参考节点）读取矩，然后修正速度和应力张量：

- **无滑移 (No-Slip)**：将壁面处速度设为零 $\boldsymbol{u}\_{\text{ghost}} = 0$
- **自由滑移 (Free-Slip)**：保留切向分量，消除法向分量 $\boldsymbol{u}_{\text{ghost}} = \boldsymbol{u}_{\text{ref}} - (\boldsymbol{u}\_{\text{ref}} \cdot \boldsymbol{n})\boldsymbol{n}$

应力张量根据速度的变化进行修正，保持非平衡应力不变：

$$S\_{\alpha\beta}^{\text{ghost}} = S\_{\alpha\beta}^{\text{ref}} - u\_\alpha^{\text{ref}} u\_\beta^{\text{ref}} + u\_\alpha^{\text{ghost}} u\_\beta^{\text{ghost}}$$

### 4.3.4. 浸入边界法 (IBM) 流固耦合

对于复杂几何体（球体、长方体、圆柱体、任意三角网格）与流体的双向耦合，本项目采用**基于罚函数的弥散界面浸入边界法（Penalty-based Diffused Immersed Boundary Method）**。固体表面离散为拉格朗日点集合（Lagrangian points），每个点携带其所代表的面元面积 $\Delta s\_p$。

**耦合流程**（每次时间步内的IBM迭代，本项目默认执行5次迭代）：

**第一步：速度插值与罚函数力计算** [`ib_interp_force.comp`](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/ib_interp_force.comp)

对每个拉格朗日采样点 $\boldsymbol{p}$：

1. 由刚体的线速度和角速度合成该点期望速度 $\boldsymbol{u}_b = \boldsymbol{v}_{cm} + \boldsymbol{\omega} \times (\boldsymbol{p} - \boldsymbol{r}\_{cm})$
2. 利用余弦核函数 $W(\boldsymbol{p}, \boldsymbol{x})$ 在 $3 \times 3 \times 3$ 支持域内插值流体宏观速度 $\boldsymbol{u}\_f$ 和密度 $\rho\_f$
3. 计算罚函数力：$\boldsymbol{F}\_{s \to f} = \rho\_f (\boldsymbol{u}\_b - \boldsymbol{u}\_f) \cdot \Delta s\_p$

**第二步：力扩散到流体网格** [`ib_spread_force.comp`](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/ib_spread_force.comp)

将拉格朗日点上的力通过余弦核函数反向分配到周围流体网格（$3 \times 3 \times 3$ 支持域）。由于多个拉格朗日点可能影响同一网格节点，必须使用 `atomicAdd` 确保写一致性。

$$\boldsymbol{F}_{\text{grid}}(\boldsymbol{x}) = \sum\_p \boldsymbol{F}_{s \to f}^{(p)} \cdot W(\boldsymbol{p}, \boldsymbol{x})$$

**第三步：LBM碰撞与流动**

`collide_and_stream.comp` 在碰撞阶段读取累积力场 `cfs`，通过 §4.2.5 的闭式解析式将其计入碰撞更新。

**第四步：更新刚体运动** [`ib_update_body.comp`](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/ib_update_body.comp)

收集流体对刚体的反作用力 $\boldsymbol{F}_{f \to s} = -\sum \boldsymbol{F}_{s \to f}^{(p)}$ 和扭矩 $\boldsymbol{\tau} = \sum \boldsymbol{r}_p \times \boldsymbol{F}_{f \to s}^{(p)}$（通过共享内存树状规约求和），计算加速度并更新刚体的位置（$\Delta\boldsymbol{x} = \boldsymbol{v} \Delta t$）和姿态（四元数积分）。

刚体运动方程（含重力和浮力）：

$$\begin{aligned}
\boldsymbol{a} &= \frac{\boldsymbol{F}_{\text{total}}}{m}, \quad \boldsymbol{v} \leftarrow (\boldsymbol{v} + \boldsymbol{a}\Delta t) \cdot 0.999 \\
\boldsymbol{\alpha} &= \boldsymbol{I}^{-1} \boldsymbol{\tau}_{\text{total}}, \quad \boldsymbol{\omega} \leftarrow (\boldsymbol{\omega} + \boldsymbol{\alpha}\Delta t) \cdot 0.999
\end{aligned}$$

位移被clamp在域内（保留 margin = 5格子的边距），速度上限受 `max_vel = 0.04` 约束。

**核函数**：IBM插值使用余弦核函数（$2 \times 2 \times 2$ 支持域）：

$$W\_{\text{cos}}(\boldsymbol{r}) = \prod\_{d \in {x,y,z}} \frac{1}{4}\left(1 + \cos\left(\frac{\pi}{2} |r\_d|\right)\right) \cdot \mathbb{1}\_{|r\_d| \le 1}$$

> 注：项目同时实现了Peskin核函数 $W\_{\text{Peskin}}(r) = \frac{1}{8}(3 - 2|r| + \sqrt{1 + 4|r| - 4|r|^2})$（$4 \times 4 \times 4$ 支持域）、Roma核函数等可选核函数，通过切换 `compute_weight_3d*` 函数调用适配不同耦合精度需求。

***

## 4.4. 辅助计算与可视化

### 4.4.1. 被动标量平流

为可视化烟/雾等被动示踪物，引入独立的密度标量场（3D纹理），采用 **MacCormack格式**的二阶精度平流：

1. **前向平流**：$\phi\_{\text{back}} = \text{sample}(\text{velocity}, \text{pos} - \boldsymbol{v} \cdot \Delta t / \Delta x)$
2. **回追速度**：在回追位置采样速度 $\boldsymbol{v}\_{\text{back}}$
3. **后向平流**：$\phi\_{\text{forward}} = \text{sample}(\text{density}, \text{pos}_{\text{back}} + \boldsymbol{v}_{\text{back}} \cdot \Delta t / \Delta x)$
4. **误差补偿**：$\phi\_{\text{final}} = \phi\_{\text{back}} + \frac{1}{2}(\phi\_{\text{orig}} - \phi\_{\text{forward}})$

这一设计抵消了单步平流中的数值耗散，保持标量界面的锐利度。密度通过 `spawnRate` 因子进行可控衰减。

密度注入 [`density_inject.comp`](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/density_inject.comp) 支持在球体发射器周围按距离衰减注入（falloff = $1 - d/R$）。

### 4.4.2. 涡量计算

在每个网格节点，使用中心差分计算速度场的旋度：

$$\boldsymbol{\omega} = \nabla \times \boldsymbol{u} = \begin{pmatrix}
\partial w/\partial y - \partial v/\partial z \\
\partial u/\partial z - \partial w/\partial x \\
\partial v/\partial x - \partial u/\partial y
\end{pmatrix}$$

偏导数由相邻节点差商计算：$\partial u\_y/\partial x \approx (u\_y(x+1) - u\_y(x-1))/(2\Delta x)$（格子单位 $\Delta x = 1$）。结果存储为3D纹理 `vorticityVolume`，供光线步进渲染和可视化使用。

### 4.4.3. 光线步进体积渲染

[ray\_march.comp](file:///d:/study/vulkan/HOME_LBM_Solver/shaders/ray_march.comp) 实现了基于涡量场（和/或密度场）的体积渲染。关键优化包括：

- **AABB相交测试**：每条光线先与体积包围盒求交所进入和退出的 $t$
- **空区域跳步**：当采样值低于阈值时，使用 $skipFactor$ 倍（默认4倍）的步长快速穿越空白区域
- **体光照模式**（renderMode=1）：使用 Beer-Lambert 吸收模型 $\text{opacity} = 1 - e^{-d \cdot \Delta t \cdot \alpha}$，累积透射率加权颜色
- **Alpha混合模式**（renderMode=0）：使用 `over` 算子进行前→后合成
- **早期终止**：透射率 < 0.05 或不透明度 > 0.95 时提前退出

涡量映射为颜色：将方向标准化后映射为RGB，强度决定alpha值。

***

## 4.5. 算法参数性能分析

HOME-LBM的参数体系可分为以下几类：

### 4.5.1. 网格参数

| 参数                             | 项目默认值                    | 说明                                  |
| ------------------------------ | ------------------------ | ----------------------------------- |
| $N\_x \times N\_y \times N\_z$ | $16 \times 32 \times 64$ | 计算网格分辨率，直接决定计算量 $O(N\_x N\_y N\_z)$ |
| $N\_{xyz}$                     | 32,768                   | 总网格节点数，影响 dispatch 分组数和内存分配         |

网格缩减对性能影响最为显著：从 $128^3$ ($\approx$ 2.1M) 降至 $48^3$ ($\approx$ 110K) 或 $32^3$ ($\approx$ 33K)，计算量分别降低约**19倍**和**64倍**。实际运行时需在分辨率与流体特征尺度之间平衡。

### 4.5.2. 格子模型

D3Q19 在精度与内存之间提供良好折衷：19个方向 vs. 27个方向（D3Q27），streaming的邻居查找量降低约30%。对于极高雷诺数（$Re > 10^4$）场景可切换至D3Q27以获得更优的各向同性。

### 4.5.3. 松弛参数

| 参数                | 公式                  | 说明                    |
| ----------------- | ------------------- | --------------------- |
| $\tau$            | $\tau = 3\nu + 0.5$ | 松弛时间，与运动黏度 $\nu$ 成正比  |
| $\omega = 1/\tau$ | <br />              | 松弛频率，值越大碰撞越强（趋近平衡态越快） |
| $\nu$             | 由场景设定               | 运动黏度，控制能量耗散速率         |

低黏度（高$Re$）时 $\tau \to 0.5$，$\omega \to 2$，碰撞步对数值误差敏感度增大。本求解器通过三阶Hermite重建提供了与全分布函数LBM相当的稳定性。

### 4.5.4. IBM耦合参数

| 参数                     | 默认值      | 说明                                                 |
| ---------------------- | -------- | -------------------------------------------------- |
| 罚刚度 (stiffness)        | 1.0      | 控制罚函数力的强度，增大则耦合力更强但可能引起数值振荡                        |
| IBM迭代次数                | 1-5      | 每时间步IBM子迭代次数，增大可改善耦合精度                             |
| 核函数类型                  | 余弦核      | 余弦核支持域为$3\times3\times3$，Peskin核为$5\times5\times5$ |
| 粒子间距 (target\_spacing) | 约0.5-1.0 | 拉格朗日点采样间距，密度越高耦合越精细但力分布点越多                         |

### 4.5.5. 渲染参数

| 参数                              | 说明                                  |
| ------------------------------- | ----------------------------------- |
| stepSize ($0.1 \sim 2.0$)       | 光线步进采样步长（$\times$体素尺寸），越小画面越精细但性能越低 |
| emptySkipFactor (默认4.0)         | 空区域跳步倍数，增大可加速穿越空白区但可能跳过薄细节          |
| vorticityThreshold              | 涡量阈值，低于此值视为空区域                      |
| absorption (体光照模式)              | 吸收系数，控制介质不透明度                       |
| densityScale / densityThreshold | 密度标量场的缩放和阈值，控制烟雾浓度                  |

***

## 4.6. 计算管线总览

每时间步的完整GPU计算管线如下（括号内为对应的Compute Shader）：

```
[1]  清空网格力场 (vkCmdFillBuffer)
───────────────────────────────────────
[2]  更新RigidBodyInfo缓冲区
[3]  更新拉格朗日点世界坐标   (update_positions.comp)
[4]  速度插值 + 罚函数力计算   (ib_interp_force.comp)     ← 受 enableIBM 控制
[5]  力扩散到流体网格           (ib_spread_force.comp)     ← 受 enableIBM 控制
───────────────────────────────────────
[6]  Streaming + 碰撞一体化     (collide_and_stream.comp)  ← 核心步骤
───────────────────────────────────────
[7]  刚体运动状态更新           (ib_update_body.comp)      ← 受 enableIBM 控制
[8]  拉格朗日点历史位置保存
───────────────────────────────────────
[9]  涡量 → 3D纹理             (vorticity_to_texture.comp) ← 受 useRayMarching 控制
[10] 速度矩 → 3D纹理            (velocity_to_texture.comp) ← 受 useDensityAdvection 控制
[11] 密度标量注入               (density_inject.comp)
[12] 密度标量MacCormack平流    (density_advect.comp)
───────────────────────────────────────
[13] 光线步进体积渲染           (ray_march.comp)
```

步骤\[3]-\[7]构成了**流固双向耦合**的完整回路：拉格朗日点在流体场中感受速度并产生力 → 力反扩散回流体网格 → 修正LBM碰撞 → 流体反作用力驱动刚体运动。

> **注**：所有Compute Shader之间的数据一致性通过 `VkMemoryBarrier`（`srcAccessMask = SHADER_WRITE, dstAccessMask = SHADER_WRITE | SHADER_READ`）保障，确保前一步的写入对后续步骤完全可见。

