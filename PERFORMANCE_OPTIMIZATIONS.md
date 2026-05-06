# HOME-LBM Solver 性能优化文档

## 概述

本文档系统性地记录了项目中的性能优化策略，覆盖算法层、内存层、内核计算层和渲染层四个维度。

---

## 一、LBM 核心计算优化

### 1.1 网格规模缩减

将模拟网格从 **128³ → 32³**，计算量直接降低 **64 倍**。

- 降低流体粘度参数，改善数值稳定性
- 同步调整涡环参数保持物理合理性

### 1.2 共享内存协作加载

在 `collide_and_stream.comp` 中引入共享内存（shared memory）机制：

- 每个 work group 的线程协作加载本 block 及一层 halo 的矩数据到共享内存
- 采用**循环分摊策略**，每个线程负责加载若干个网格点
- 共享内存布局为 `[SHARED_SIZE_X × SHARED_SIZE_Y × SHARED_SIZE_Z]`，包含 10 个分量数组（rho, ux, uy, uz, Sxx, Syy, Szz, Sxy, Sxz, Syz）
- 加载完成后执行 `barrier()` + `memoryBarrierShared()` 确保数据一致性
- streaming 阶段直接从共享内存读取邻居矩数据，避免重复访问全局内存

**优化原理**：LBM 的 streaming 步骤需要读取 19~27 个邻居节点的矩数据。若不使用共享内存，每个线程需要独立从全局内存读取邻居，产生大量冗余的全局内存访问。共享内存将同一 block 内数据只读取一次，后续都在 on-chip 缓存中访问，大幅降低显存带宽压力。

### 1.3 消除中间数组

在 streaming 后的矩累加阶段，删除 `f_streamed[Q]` 中间数组：

**优化前**：
```glsl
float f_streamed[Q];
for (int i = 0; i < Q; i++) {
    f_streamed[i] = reconstruct_single_ddf(i, rho_src, u_src, S_src);
}
for (int i = 0; i < Q; i++) {
    float fi = f_streamed[i];
    rho_star += fi;
    rhou_star += vec3(c[i]) * fi;
    // ... 累积二阶矩 ...
}
```

**优化后**：
```glsl
for (int i = 0; i < Q; i++) {
    float fi = reconstruct_single_ddf_fast(i, rho_src, u_src, S_src);
    rho_star += fi;
    rhou_star += vec3(c[i]) * fi;
    // ... 累积二阶矩 ...
}
```

**优化原理**：消除 Q 个 float 的寄存器/局部内存占用；避免先写入数组再读取的"写后读"（RAW）数据依赖延迟；允许编译器更好地进行指令级并行调度。

### 1.4 预计算常数替代运行时除法

| 原始写法 | 优化后 | 说明 |
|---------|--------|------|
| `omega = 1.0 / ubo.tau` | `omega = ubo.inv_tau` | 在 CPU 端预计算 `inv_tau = 1.0/tau`，通过 UBO 传入 |
| `x / 3.0` (重复出现) | `x * inv_3` | 循环外预计算 `inv_3 = 1.0/3.0` |
| `x / rho_star` (重复出现) | `x * inv_rho` | 提前计算 `inv_rho = 1.0/rho_star` |

**优化原理**：GPU 上浮点除法的延迟远高于乘法（FMA 指令）。将运行时除法替换为预计算的乘法，在热点循环中可带来显著收益。

### 1.5 硬编码常数加速共享内存索引

**优化前**（动态常量）：
```glsl
uint lz = linear_idx / SHARED_STRIDE_Z;
uint ly = (linear_idx % SHARED_STRIDE_Z) / SHARED_STRIDE_Y;
uint lx = linear_idx % SHARED_STRIDE_Y;
```

**优化后**（硬编码值）：
```glsl
uint lz = linear_idx / 36;
uint ly = (linear_idx % 36) / 6;
uint lx = linear_idx % 6;
```

**优化原理**：当除数为编译期常量时，GLSL 编译器可将除法优化为乘法移位指令；而动态 uniform 变量的除法在运行时仍需执行完整的除法运算。

### 1.6 分支简化与快速路径

- 将边界条件处理代码从多行展开改为紧凑的单行写法，减少控制流发散
- 在分布函数重构中使用直接展开的三阶 Hermite 多项式，避免函数调用开销
- 使用 `#pragma unroll` 提示编译器展开 Q 循环

---

## 二、内存带宽优化

### 2.1 Half 精度数据压缩

使用 `packHalf2x16` / `unpackHalf2x16` 对矩数据存储进行压缩：

**数据结构变化**：
```
Before: float[10] = 40 bytes per grid cell
After:  uint[5]   = 20 bytes per grid cell  (带宽减半)
```

**布局映射**：
```
uint[0] = packHalf2x16(rho, rhou.x)
uint[1] = packHalf2x16(rhou.y, rhou.z)
uint[2] = packHalf2x16(Sxx, Syy)
uint[3] = packHalf2x16(Szz, Sxy)
uint[4] = packHalf2x16(Sxz, Syz)
```

**修改范围**：

- `simulate_header.glsl`：`load_moments()` 和 `store_moments()` 全部改为 pack/unpack 操作
- `ib_interp_force.comp`：插值力的读取适配解压格式
- `velocity.vert`：速度场可视化管线适配
- `velocity_to_texture.comp`：速度纹理转换适配
- `vorticity_to_texture.comp`：涡量纹理转换适配
- `main.cpp`：缓冲区分配大小从 `Nxyz * 10 * sizeof(float)` 改为 `Nxyz * 5 * sizeof(uint32_t)`，初始化数据使用 `glm::packHalf2x16()` 编码

**精度考量**：half（FP16）精度约为 3.3 位十进制有效数字，对于 LBM 中间矩数据的存储足够使用，不会引入明显的数值误差，但带宽需求减半在内存密集型的 LBM 计算中收益巨大。

### 2.2 描述符集精简

- 移除 `rho/DDF` 中间缓冲区，不再创建、更新及销毁
- 描述符集布局绑定数量从 **23 → 20**
- 重新映射绑定编号：去掉原 3（rho）和 5（DDF），后续缓冲区和存储图像依次前移
- 同步调整 `createComputeDescriptorSetLayout` 及所有描述符写入代码

**优化原理**：减少描述符集绑定数量可降低 GPU 描述符表遍历开销，同时减少不必要的缓冲区分配和内存占用。

---

## 三、光线步进渲染优化

### 3.1 UVW 步进增量预计算

**优化前**（循环内重复计算）：
```glsl
for (uint step = 0u; step < maxSteps; step++) {
    vec3 pos = ray.origin + t * ray.direction;
    vec3 uvw = (pos - volumeOffset) * invVolumeExtent;
    // ... 采样和累积 ...
    t += stepSize;
}
```

**优化后**（循环外预计算，循环内增量更新）：
```glsl
vec3 dUVW_per_t = rayDir * invVolumeExtent;
vec3 dUVW_step = dUVW_per_t * stepSize;
vec3 dUVW_skip = dUVW_per_t * skipStep;

// 初始化起始 uvw
vec3 uvw = (rayOrigin + t * rayDir - volumeOffset) * invVolumeExtent;

for (...) {
    // 直接使用 uvw 采样
    // 前进时：uvw += dUVW_step  或  uvw += dUVW_skip
}
```

**优化原理**：每次循环迭代省去一次向量加法（`ray.origin + t * ray.direction`）、一次向量减法和一次向量乘法，转换为简单的向量加法增量更新。

### 3.2 渲染模式分支分离

将两种渲染模式（体积光照 `renderMode==1` 和 alpha 混合 `renderMode==0`）的逻辑在外层分离：

- 避免循环内每次迭代都判断 `renderMode`
- 每种模式拥有独立的循环，编译器可针对特定模式做更好的优化
- 提前在循环外读取 `sceneColor`，消除循环后才读取的延迟

### 3.3 快速指数近似

实现 `fastOneMinusExpNeg(x)` 替代 `1.0 - exp(-x)`：

```glsl
float fastOneMinusExpNeg(float x) {
    if (x < 0.5) {
        // 三阶泰勒展开: 1 - e^(-x) ≈ x - x²/2 + x³/6
        return x * (1.0 - x * (0.5 - x * 0.166667));
    }
    // 大值回退到标准 exp2
    return 1.0 - exp2(-x * 1.44269504);
}
```

**优化原理**：
- 体积光照中 `exp()` 是最昂贵的超越函数调用
- 光线步进中大部分采样点的 `x = density * stepSize * absorption * densityScale` 值很小（< 0.5），走泰勒展开路径只需几次乘加运算
- 对于大值情况回退到 `exp2()`，利用 GPU 硬件对 `exp2` 的原生支持

### 3.4 空区域跳步优化

- 引入 `emptySkipFactor`，当采样值低于阈值时跳过更大的步长
- 空区域使用 `dUVW_skip`（乘以 skipFactor），密集区域使用 `dUVW_step`（常规步长）
- 显著减少穿越空白区域的迭代次数

### 3.5 循环条件精简

使用 `tRemaining > 0.0f` 替代 `t < tMaxEffective` 作为循环判断条件：

- `tRemaining` 在每个步长推进后自减，为简单标量
- 避免每次迭代重新计算 `tMaxEffective - t` 的向量运算

---

## 四、边界条件处理优化

### 4.1 UBO 动态配置替代硬编码

将边界条件类型从硬编码改为通过 UBO 动态传递：

- 边界类型（流入/流出/周期性/固体壁面）通过 uniform buffer 配置
- 提取通用边界处理为独立函数，减少着色器中的条件分支
- 支持运行时切换边界条件，无需重新编译着色器

### 4.2 边界处理逻辑简化

- 重构边界条件处理为核心函数 + 边界参数配置的模式
- 简化网格节点类型定义，移除冗余标志位
- 流入/流出/周期性边界在共享内存加载阶段统一处理，后续 streaming 无需再判断

---

## 五、其他优化

### 5.1 网格计算优化

- 使用 `(N+3)/4` 计算工作组数量，确保完整覆盖而非遗漏边界
- 修正边界检查位置，确保越界线程仍参与共享内存加载（但不执行计算）
- 网格尺寸从 `64×64×64` 调整为 `48×48×48`，平衡精度与性能
- 根据网格尺寸动态调整体积边界

### 5.2 CPU 端初始化优化

- 涡环速度场初始化改用 `glm::packHalf2x16()` 编码，直接生成压缩格式的初始数据
- 避免先初始化 float 数组再全量转换的中间步骤
- 描述符缓冲区信息同步更新为压缩后的大小

---

## 优化层次全景

```
┌─────────────────────────────────────────────────────┐
│ 算法层：128³→32³ 网格缩减（64× 加速）               │
├─────────────────────────────────────────────────────┤
│ 内核层：消除中间数组、预计算常数替代除法             │
│         硬编码常数加速索引、累加器替代数组           │
│         分支消除、快速路径、循环展开                 │
├─────────────────────────────────────────────────────┤
│ 内存层：共享内存协作加载、half 精度压缩（50% 带宽） │
│         描述符集精简（23→20）、移除冗余缓冲区        │
├─────────────────────────────────────────────────────┤
│ 渲染层：UVW 增量预计算、快速指数近似                 │
│         渲染模式分离、空区域跳步优化                 │
│         tRemaining 标量化循环条件                    │
├─────────────────────────────────────────────────────┤
│ 架构层：边界条件 UBO 动态化、网格计算四舍五入        │
│         CPU 端直接生成压缩格式数据                  │
└─────────────────────────────────────────────────────┘
```

### 核心优化路径

```
减少计算量（网格缩减）
    ↓
减少全局内存访问（共享内存）
    ↓
减少内存带宽（half 压缩、描述符精简、消除中间数组）
    ↓
精细打磨热点（预计算、分支消除、快速近似函数）
```
