# 章节四：核心算法原理

本章节详细阐述本项目中流体仿真的核心算法。不同于传统基于宏观Navier-Stokes方程的求解器（如PBF或基于网格的Eulerian方法），本项目采用了**格子玻尔兹曼方法 (Lattice Boltzmann Method, LBM)** 作为核心流体求解器，并结合**高阶矩编码技术 (HOME-LBM)** 进行显存优化。流固耦合方面引入了**浸入边界法 (Immersed Boundary Method, IBM)**，最后通过**光线步进 (Ray Marching)** 实现高质量的屏幕空间流体渲染。

## 4.1 控制方程

### 4.1.1 Boltzmann方程与Maxwell-Boltzmann分布
LBM 的物理基础是介观尺度的 Boltzmann 方程。该方程描述了微观粒子分布函数 $f(\mathbf{x}, \mathbf{v}, t)$ 随时间和空间的演化过程。在没有外力的情况下，连续 Boltzmann 方程表示为：
$$ \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f = \Omega(f) $$
其中，$\Omega(f)$ 为碰撞算子，描述粒子碰撞导致分布函数的变化。流体系统在热力学平衡时，分布函数服从 Maxwell-Boltzmann 平衡态分布 $f^{eq}$。

### 4.1.2 宏观Navier-Stokes方程的恢复
通过 Chapman-Enskog 展开，可以证明在低马赫数（$Ma \ll 1$）和连续介质假设（Knudsen数 $Kn \ll 1$）下，LBM的介观演化方程可以渐进恢复到宏观的弱可压 Navier-Stokes (N-S) 方程：
$$ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0 $$
$$ \frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u}\mathbf{u}) = -\nabla p + \nu \nabla^2 (\rho \mathbf{u}) + \mathbf{F} $$
其中 $\rho$ 为宏观密度，$\mathbf{u}$ 为宏观速度，$\nu$ 为流体运动粘度，$\mathbf{F}$ 为外力项。

### 4.1.3 状态方程
在 LBM 中，流体的压力 $p$ 与密度 $\rho$ 之间通过等温理想气体状态方程直接关联：
$$ p = c_s^2 \rho $$
其中 $c_s$ 为格子声速。本项目中 $c_s^2 = 1/3$。这种显式的状态方程避免了传统流体求解器中求解泊松方程（Poisson Equation）带来的巨大计算开销。

## 4.2 LBM 离散格式

### 4.2.1 速度空间离散化 (D3Q19模型)
为在计算机上求解 Boltzmann 方程，需要将连续的速度空间离散化为有限个方向。本项目支持 D3Q15, D3Q19, D3Q27 模型，并默认使用 **D3Q19** 模型（三维19速模型）。
在 D3Q19 模型中，离散速度集合 $\mathbf{c}_i$ ($i=0,1,\dots,18$) 包含：
- 1 个静止中心向量
- 6 个面心向量
- 12 个棱心向量
对应的权重 $w_i$ 分别为：$w_0 = 1/3$, 面心 $w_s = 1/18$, 棱心 $w_e = 1/36$。

### 4.2.2 碰撞与迁移算子 (BGK模型)
本项目采用经典的 BGK (Bhatnagar-Gross-Krook) 单弛豫时间碰撞模型。完整的 LBM 演化包含两个步骤：
1. **碰撞 (Collision)**：粒子在网格节点上发生碰撞，向平衡态分布弛豫。
   $$ f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} [f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)] $$
2. **迁移 (Streaming)**：碰撞后的粒子沿离散速度方向传播到相邻节点。
   $$ f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t) $$
其中，$\tau$ 为无量纲弛豫时间，决定了流体的粘度。

### 4.2.3 高阶矩编码算法 (HOME-LBM)
传统 LBM 需要为每个网格节点存储 19 个（D3Q19）分布函数，这带来了巨大的显存和访存带宽开销。本项目引入了 **HOME-LBM (High-Order Moment-Encoded LBM)** 算法。
核心思想：不在内存中保存全量的分布函数 $f_i$，而是仅存储其宏观物理量及其高阶矩，具体包括：
- 零阶矩：密度 $\rho$ (1 个分量)
- 一阶矩：动量 $\rho \mathbf{u}$ (3 个分量)
- 二阶矩：应力张量 $S_{\alpha\beta}$ (6 个分量：$S_{xx}, S_{yy}, S_{zz}, S_{xy}, S_{xz}, S_{yz}$)

每个节点仅需存储 10 个变量，显存占用降低至传统的 50% 左右。在 `collide_and_stream.comp` 的计算过程中，利用这 10 个宏观矩通过快速多项式展开重构分布函数：
$$ f_i \approx \rho w_i \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i\mathbf{c}_i - c_s^2\mathbf{I}) : \mathbf{S}}{2c_s^4} + \text{Third-Order Terms} \right] $$
这种高阶矩编码不仅大幅减少了内存占用，同时极大地提高了 GPU 的内存读取效率。

### 4.2.4 空间与时间离散化
计算域被划分为均匀的笛卡尔网格（Grid Size）。时间步长 $\Delta t$ 与空间步长 $\Delta x$ 的比值固定为 1。更新过程中，碰撞和迁移操作在 Compute Shader 中合并为一个 Pass (`collide_and_stream.comp`)，通过从相邻节点拉取矩信息完成演化，避免了数据的读写冲突。

## 4.3 边界条件与流固耦合

### 4.3.1 固壁与流场边界
项目实现了多种边界条件处理机制，针对网格边界（Ghost Nodes）：
- **刚性边界 (No-Slip / Free-Slip)**：采用半步反弹格式 (Bounce-back scheme)。当流体粒子遇到固体壁面时，将其分布函数沿原路反弹，以满足壁面处的无滑移或自由滑移条件。
- **周期边界 (Periodic)**：流体从一侧流出后，直接从另一侧对称位置流入。
- **出入口边界 (Inflow / Outflow)**：在入口处强制设定密度 $\rho_{in}$ 和速度 $\mathbf{u}_{in}$；在出口处应用零梯度外推条件或指定压力。

### 4.3.2 浸入边界法 (Immersed Boundary Method)
对于复杂形态的运动刚体与流体的相互作用（流固耦合），本项目采用 **浸入边界法 (IBM)** (`ibm_force.comp`, `ib_interp_force.comp` 等)。
1. **速度插值**：利用平滑核函数（如 cosine 1D 核函数）将流体网格节点的速度插值到浸入的拉格朗日边界点上。
2. **受力计算**：根据拉格朗日点处的流体速度与刚体表面速度的差值，计算所需的恢复力（惩罚力或直接力）。
3. **力源分布**：将计算得到的拉格朗日力，通过同样的平滑核函数反向散布（Spread）到流体欧拉网格节点上，作为 LBM 碰撞过程中的外力源项 $\mathbf{F}$ 加入动量更新。

### 4.3.3 刚体动力学求解器
在受到流体反作用力后，刚体的状态更新由 `rigid_body_solver.comp` 处理。求解器整合流体施加的合力和合力矩，根据刚体的质量和惯性张量，利用显式欧拉或辛欧拉方法更新刚体的位置、四元数姿态、线速度与角速度。

## 4.4 流体渲染与可视化

### 4.4.1 光线步进 (Ray Marching)
流体的密度与涡度场作为 3D 纹理传递给渲染管线。本项目在 `ray_march.comp` 中实现了基于物理的体积渲染技术。
- 从相机向屏幕每个像素发射射线。
- 沿着射线步进 (Step Size) 采样 3D 密度/涡度纹理。
- 根据局部密度计算光线吸收率 (Absorption) 和散射，通过累积计算像素最终颜色，从而呈现出烟雾、流体云等真实的体积感。

### 4.4.2 场景深度遮挡与融合
为了使渲染出的体积流体与场景中的实体网格（如刚体、天空盒）正确融合，Ray Marching 算法引入了深度纹理（Scene Depth）。在射线步进时，一旦当前射线的深度超过了场景深度缓冲的值，即终止步进，确保流体不会遮挡前方的固体物体。

### 4.4.3 拉格朗日粒子可视化
除了体渲染外，项目还支持基于粒子 (Lagrangian Particles) 的流体可视化。在 `update_positions.comp` 中，流场速度通过三线性插值更新无质量的拉格朗日粒子的位置。随后在 `particle.vert/frag` 中，根据粒子的速度大小进行伪彩着色（Color Mapping），直观展现湍流的旋涡结构。

## 4.5 算法参数性能分析

### 4.5.1 格子分辨率与松弛时间
- **格子分辨率 (Grid Size)**：决定了流体仿真的精度与开销。网格越密，可解析的湍流涡旋越精细，但显存占用呈三次方增长。
- **弛豫时间 ($\tau$)**：与流体的宏观运动粘度直接相关。公式为 $\nu = c_s^2 (\tau - 0.5) \Delta t$。当 $\tau$ 接近 0.5 时，粘度极小，雷诺数极大，能产生丰富的湍流细节，但也容易导致数值不稳定。本项目默认 $\tau$ 值经过调优，在保持稳定性的同时尽可能降低粘度。

### 4.5.2 共享内存与计算加速
在 `collide_and_stream.comp` 的实现中，采用了 **共享内存 (Shared Memory)** 技术来加速邻域数据的拉取。由于 HOME-LBM 在重构分布函数时需要读取相邻节点的宏观矩数据，算法将本 WorkGroup 及其周围一层 Halo 区域的矩数据一次性协同加载到 GPU 共享内存中，有效降低了全局内存 (Global Memory) 的带宽压力。

### 4.5.3 显存瓶颈与优化
对于大规模 3D LBM 仿真，最大的性能瓶颈在于显存带宽。采用 HOME-LBM 架构后，数据读写量从每节点 19 个浮点数降低为 10 个，节约了约 47% 的内存带宽。结合 GPU 的合并访存 (Coalesced Memory Access) 优化，整体仿真帧率相比传统 LBM 方法可提升数倍，使得在移动端或普通消费级 GPU 上进行高精度流体模拟成为可能。
