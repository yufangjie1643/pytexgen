# Cross-Language And JIT Modernization Report

本文面向 `pytexgen` 与 `Voxel-ACDM` 的当前代码形态，评估 SWIG 之外的现代跨语言技术栈、迁移难度，以及 JIT 是否适合现有体素化和求解流程。

结论先行：

1. 不建议短期全量替换 SWIG。SWIG 仍适合维持 TexGen 大量历史 C++ 建模 API 的兼容性。
2. 新增功能不应继续扩大 SWIG 暴露面，而应建设一个现代数据接口层：`CTextile -> VoxelGridData -> numpy/torch/DLPack -> solver`。
3. 对当前代码，JIT 可以用，但不能“一键加速全部”。最适合 JIT 的是体素分类核心，不是 TexGen/SWIG 对象遍历，也不是 FFT 本身。
4. 推荐路线是：保留 SWIG 兼容层，新增 nanobind/pybind11 小扩展导出连续数组，逐步引入 Numba/Triton/custom torch op 优化热点。

## 当前项目现状

当前 `pytexgen` 有三层：

```text
TexGen C++ Core
  -> SWIG Core binding
  -> Python utility layer
  -> gpu_voxelizer / acdm_solver direct data interface
```

当前新增的数据流：

```text
TexGen CTextile
  -> extract_snapshots(...)
  -> voxelize_textile_data(...)
  -> VoxelGridData
  -> numpy / torch / Voxel-ACDM
```

当前代码还提供了一个更适合 Python-C 边界的中间契约：

```text
TexGen CTextile
  -> extract_snapshot_bundle(...)
  -> SnapshotBundle  # structure-of-arrays
  -> voxelize_snapshot_bundle_data(...)
  -> VoxelGridData
```

`extract_snapshot_bundle(...)` 会优先寻找可选 `_fastdata` provider；没有
provider 时回退到现有 SWIG 对象遍历。这样后续 nanobind/pybind11 扩展只
需要实现一个窄接口并返回连续数组，不需要替换整套 SWIG 建模 API。
`SnapshotBundle` 构造入口会校验 shape、offset 单调性、offset 终点和 yarn
数量一致性；`fastdata_provider_status()` 可以让上层代码明确判断当前是否
真正加载了 `_fastdata`，避免把 fallback 误当成 C++ 快路径。
当前 `_fastdata` 是 CPython facade：它不重复链接 `TexGenCore`，而是导入
`_Core._fastdata_extract_snapshot_bundle_direct(...)`。`_Core` 先用 SWIG
runtime 把 Python proxy 安全转换成 `TexGen::CTextile*`，随后在同一个
`TexGenCore` 静态实例内直接遍历 `CTextile/CYarn/CSlaveNode`，一次性返回
通过 NumPy C API 分配的 owned contiguous numpy arrays。numpy voxelizer 的
`SnapshotBundle` 路径会直接消费 flat arrays，不再拆回 per-yarn
`YarnSnapshot` Python 对象。这已经完成了真正的 C++ 指针直连，同时避免了
wheel 静态链接场景下的 TexGen singleton 分裂问题。
`VoxelGridData.to_dlpack(...)` 则提供了 `yarn_id`、`material_id`、
`occupancy` 的 DLPack 出口，方便 torch/CuPy/JAX 类张量库消费结果。

对 `Voxel-ACDM`，目前已经有两条直连方向：

```text
numpy-numpy:
VoxelGridData -> solve_acdm_fft_numpy_from_voxel_data(...)

CUDA/Triton:
VoxelGridData -> phase_ids -> FEMHomogenizerBatchedIsotropicPhases
```

当前最大的性能事实是：小网格下，体素化耗时主要不是分类，而是从 TexGen/SWIG 对象抽取 yarn 几何快照。例如 24^3 测试中：

```text
voxelize extract:   ~0.91 s
voxel classify:     ~0.08 s
FFT solve:          ~4.55 s
```

所以如果只给 Python 分类函数套 JIT，不能解决全部瓶颈。

## 技术栈总览

| 技术 | 类型 | 适合解决的问题 | 对本项目建议 |
|---|---|---|---|
| SWIG | 自动绑定生成 | 大面积暴露已有 C/C++ API | 保留兼容层 |
| pybind11 | 手写 C++/Python 绑定 | 高质量 Python API、numpy buffer、对象生命周期 | 适合新增小模块 |
| nanobind | 现代 C++/Python 绑定 | 类似 pybind11，但更强调轻量和较低开销 | 优先考虑新增性能接口 |
| cppyy | 动态 C++ 绑定 | 快速探索大型 C++ API | 可用于实验，不建议作为 wheel 主路径 |
| Python Limited API / Stable ABI | ABI 策略 | 减少 Python 小版本重编译 | 适合小型 C API，不适合复杂 C++ 类全暴露 |
| HPy | 新 Python 扩展 API | 面向多解释器、长期 ABI 演进 | 长期观察，不作为当前主迁移路线 |
| PyO3/Rust | Rust/Python 绑定 | 新数值模块、安全系统代码 | 可用于新模块，不适合直接包 TexGen C++ |
| DLPack | 张量交换协议 | torch/cupy/jax 等零拷贝交接 | 推荐加入 `VoxelGridData` |
| Arrow C Data/Device | 列式/设备数据交换 | 大规模表格/列式元数据交换 | 非核心，除非以后做数据集管线 |
| Numba | Python/NumPy JIT | CPU 数值循环加速 | 可用于 CPU 体素分类内核 |
| torch.compile | PyTorch 编译器 | 纯 torch 张量程序的图优化 | 当前收益有限，重构后可试 |
| Triton | Python 风格 GPU kernel | 自定义 GPU 分类/装配 kernel | 长期最有价值 |
| PyTorch custom op | C++/CUDA torch op | 与 torch.compile/调度器集成 | 适合稳定高性能后端 |

## 绑定技术路线

### SWIG

SWIG 的优势是自动化和覆盖面大。它可以把大量 C++ 类包装成 Python proxy class，这正好匹配 TexGen 当前“很多 C++ 建模类都要暴露”的历史结构。

当前不建议全量迁移 SWIG，原因：

- TexGen API 面很宽，类层级、重载、枚举、容器都多。
- 全量迁移到 pybind11/nanobind 需要逐类重写绑定，回归风险很高。
- 用户已有 `from pytexgen import CTextileWeave2D, CYarn, XYZ` 等习惯。

建议：

```text
SWIG = legacy modeling API compatibility layer
new binding = performance/data extraction layer
```

也就是不拆 SWIG，但新接口不要继续依赖 SWIG 对象逐点读写。

迁移难度：

```text
保留 SWIG：低
局部新增现代绑定：中
全量替换 SWIG：高到极高
```

### pybind11

pybind11 是成熟主流方案。它是 header-only，不需要 SWIG 那种中间生成步骤，适合手写清晰的 Python API。

适合在本项目中做：

```cpp
extract_yarn_snapshots(CTextile&) -> py::dict / py::array
voxelize_structured(...) -> py::array_t<int32_t>
build_orientation_field(...) -> py::array_t<float>
```

优势：

- C++ 写法直接。
- numpy buffer 支持成熟。
- 社区大、问题容易查。

劣势：

- 每个绑定函数都要手写。
- 大规模绑定时编译时间和二进制大小可能上升。
- 对 ABI3/Stable ABI 的支持不是它的核心优势。

迁移难度：

```text
新增 3-5 个数据导出函数：中
替换全部 Core.py：高
```

### nanobind

nanobind 是更现代的 C++/Python 绑定库，目标与 pybind11 类似，但更强调轻量、编译和运行开销控制。官方文档中也明确说明它通过独立支持库减少每个绑定文件重复编译大量代码的问题。

对本项目，nanobind 比 pybind11 更适合新增“窄而快”的接口：

```text
TexGenCore -> nanobind extension -> contiguous numpy arrays
```

当前模块名：

```text
pytexgen._fastdata
```

当前接口：

```python
from pytexgen._fastdata import extract_snapshot_bundle

bundle = extract_snapshot_bundle(textile)
```

迁移难度：

```text
新增数组导出 extension：中
替换 SWIG：高
```

### cppyy

cppyy 可以动态访问 C++，适合交互式探索大型 C++ API。它的优点是少写 wrapper，缺点是打包、启动、运行环境和可控性需要谨慎评估。

对本项目建议：

```text
可用于探索 TexGen C++ API 和验证 wrapper 设计
不建议作为 PyPI wheel 主交付路径
```

迁移难度：

```text
实验：低到中
生产 wheel：中到高
```

### Python Limited API / Stable ABI 与 HPy

Stable ABI 的价值是减少 Python 小版本重编译。HPy 的目标是提供更现代、更少绑定 CPython 内部细节的扩展 API。

但对本项目要现实：

- TexGen 是复杂 C++ 类库，不是几个简单 C 函数。
- pytexgen 现在依赖 C++ ABI、编译器 ABI、Windows `.pyd`，不仅是 Python ABI。
- Stable ABI 适合小型、C 风格、边界清晰的扩展模块。

建议：

```text
短期不作为主线
如果未来写一个很小的 C API 数据导出 shim，可以考虑 Limited API
HPy 作为长期观察项
```

## 数据交换技术

### numpy buffer / array protocol

这是当前最直接、最稳定的路线。`VoxelGridData` 已经在做：

```python
data.yarn_id      # flat array
data.grid         # (Nz, Ny, Nx)
data.material_id()
data.to("numpy")
data.to("torch", device="cuda")
```

当前已把 yarn 快照从 `_Core` 内一次性导出为 numpy array，避免 Python
循环逐点读取 SWIG 对象。后续优化应继续把更靠近 voxel/solver 的数据保持为
连续数组或 DLPack tensor，减少对象级跨语言调用。

迁移难度：

```text
Python 层继续完善：低
C++ 直接导出 numpy：中
```

### DLPack

DLPack 是当前张量库之间交换设备内存的关键协议。Python array API 标准也把 DLPack 纳入数据交换机制。对本项目，它可以让：

```text
torch tensor
CuPy array
JAX array
其他支持 DLPack 的求解器
```

共享张量数据，避免中间 CPU 拷贝。

推荐给 `VoxelGridData` 增加：

```python
data.to_dlpack(field="material_id")
data.from_dlpack(...)
```

不过注意：DLPack 适合 tensor 数据，不负责表达 TexGen 的复杂对象语义。它应该用于 `yarn_id/material_id/orientation/C_field`，不是用于 `CTextile`。

迁移难度：

```text
torch tensor -> DLPack：低
numpy CPU -> DLPack：中，收益有限
C++ 自己生产 DLPack capsule：中到高
```

### Arrow C Data / Device

Arrow 更适合列式数据、表格、数据集管理。对于当前体素求解核心，不如 DLPack 直接。

可能用途：

```text
大量 RVE 样本的 metadata / material table / result table
```

不建议用于核心 voxel tensor。

## JIT 是否能用于当前代码

可以，但要分段看。

### 不能直接 JIT 的部分

以下部分 JIT 帮不上主要忙：

```text
TexGen C++ 建模
SWIG 对象方法调用
extract_snapshots 里的 Python/SWIG 对象遍历
FFTHomogenizer 里的 numpy.fft 本身
```

原因：

- TexGen 建模已经在 C++ 内部。
- SWIG 调用是 Python/C++ 边界，不是纯 Python 数值循环。
- numpy FFT 已经调用底层 C/Fortran/FFT 实现，Numba 不会把 `np.fft` 编得更快。

### Numba 可用位置

Numba 适合 CPU 数值循环，尤其是当前 numpy 体素分类的候选方向：

```text
centers + packed yarn arrays -> yarn_id
```

但需要重构数据结构。当前 `YarnSnapshot` 是 Python dataclass，里面有变长数组；Numba 更喜欢：

```text
P_flat, T_flat, U_flat, S_flat
offset arrays
section_flat
translation_flat
```

也就是 structure-of-arrays：

```python
@numba.njit(parallel=True, fastmath=True, cache=True)
def classify_voxels_numba(...):
    ...
```

预期收益：

- 对中大型 CPU 网格，分类阶段可能明显加速。
- 对 8^3、24^3 小网格，收益可能被编译时间和 extract 阶段淹没。
- 对当前测试中的 `extract ~0.9s`，Numba 没有直接帮助。

迁移难度：

```text
写一个 numba CPU 分类内核：中
让所有变长 yarn/section 数据适配 numba：中到高
稳定维护 Windows/CPython 多版本：中
```

建议优先级：

```text
中期可做，但不是第一优先级
```

### torch.compile 可用位置

`torch.compile` 适合纯 torch 张量程序。当前 torch 体素分类代码里有：

```text
Python for y_idx
Python for t_idx
.item() 取标量
candidate.nonzero(...)
动态 active_idx
数据相关分支
```

这些会导致 graph breaks 或频繁重编译，torch.compile 很难吃满。

所以当前直接套：

```python
compiled = torch.compile(_classify_voxels_torch)
```

大概率收益有限，甚至可能更慢。

如果要让 torch.compile 有用，需要先重构：

```text
固定 padded tensors
减少 .item()
减少 Python data-dependent branch
把 yarn/translation 维度批量化
把 point-in-polygon 改成更规则的 tensor kernel
```

迁移难度：

```text
直接尝试：低，但收益不稳
为 compile 重构：高
```

建议：

```text
不作为主优化路线
作为重构后的 benchmark 选项保留
```

### Triton 可用位置

Triton 是最适合本项目 GPU 体素化的 JIT 技术，尤其是目标已经接 `Voxel-ACDM`，而 Voxel-ACDM 主求解器本身也使用 Triton。

适合做：

```text
voxel centers -> yarn_id
phase_ids packing
material_id mapping
orientation field sampling
```

一个合理的 Triton 方向：

```text
每个 program block 处理一段 voxel centers
候选 yarn/translation 用 AABB 过滤
局部做 nearest slave node + section polygon test
输出 yarn_id / orientation
```

优势：

- 输出可以直接是 torch CUDA tensor。
- 可与 Voxel-ACDM CUDA/Triton 求解器处在同一设备。
- 大网格下能减少 CPU/GPU 往返。

难点：

- 当前 point-in-polygon + nearest slave node 对变长 yarn 数据不够规则。
- 需要将 yarn 几何彻底转成 padded/flat GPU arrays。
- AABB pruning 的数据相关分支在 GPU 上要重新设计。
- Windows + Triton 环境需要按目标机器验证。

迁移难度：

```text
prototype kernel：高
生产稳定 kernel：高到极高
```

建议：

```text
长期主优化路线
先做一个限制版：固定截面点数、固定 max slave nodes、单 translation 或少量 translation
```

### PyTorch custom C++/CUDA op

如果 Triton 原型稳定，或者需要更底层控制，可以写 PyTorch custom op：

```text
torch.ops.pytexgen.voxelize(...)
```

它的优势是能更好接入 PyTorch dispatcher、torch.compile、FakeTensor 和后续生态。官方 PyTorch custom operator 教程也强调了 C++/CUDA op 与 PyTorch 子系统的组合方式。

劣势：

- 开发和 CI 更重。
- Windows CUDA 编译和 wheel 复杂度高。
- 调试成本高于 Triton。

迁移难度：

```text
高
```

建议：

```text
先 Triton，后 custom op
```

## 对当前代码的迁移难度评估

| 任务 | 难度 | 价值 | 说明 |
|---|---:|---:|---|
| 保留 SWIG，继续维护当前 Core 绑定 | 低 | 高 | 保障兼容 |
| `VoxelGridData` 补 `to_dlpack()` | 低到中 | 中 | 方便接其他 GPU 求解器 |
| numpy FFT 直连接口 | 已完成 | 高 | 无 CUDA 环境可验证 |
| Voxel-ACDM isotropic phase-LUT 直连 | 已完成接口 | 高 | 需目标 CUDA/Triton 环境验证 |
| 输出 orientation field | 中到高 | 很高 | 各向异性求解必须 |
| nanobind/pybind11 新增 C++ 快照导出 | 中 | 很高 | 解决 SWIG extract 开销 |
| Numba CPU voxel classifier | 中 | 中 | 加速 CPU 分类，但不解决 extract |
| torch.compile 当前 torch classifier | 低 | 低到中 | 当前结构 graph break 风险大 |
| Triton GPU voxel classifier | 高 | 很高 | 大网格和 Voxel-ACDM CUDA 直连核心 |
| PyTorch custom CUDA op | 高 | 很高 | 适合成熟后固化 |
| 全量 SWIG -> nanobind/pybind11 | 极高 | 中 | 风险大，不推荐 |

## 推荐实施路线

### Phase 1：数据接口稳定化

目标：让所有求解器接同一份结构化数据。

应完成：

```python
VoxelGridData.yarn_id
VoxelGridData.material_id()
VoxelGridData.voxel_size
VoxelGridData.domain_size
VoxelGridData.to("numpy")
VoxelGridData.to("torch", device="cuda")
VoxelGridData.to_dlpack()
```

同时稳定：

```python
solve_acdm_fft_numpy_from_voxel_data(...)
solve_acdm_isotropic_from_voxel_data(...)
```

难度：低到中。

### Phase 2：orientation field

目标：支持各向异性 yarn。

新增：

```python
voxelize_textile_fields(
    textile,
    fields=("yarn_id", "orientation"),
)
```

输出：

```text
orientation1: (Nz, Ny, Nx, 3)
orientation2: (Nz, Ny, Nx, 3)
```

然后接 Voxel-ACDM：

```text
orientation field -> build_voigt_stiffness_fields_gpu
orientation field -> build_stiffness_field for FFT
```

难度：中到高。

### Phase 3：C++ 快照导出 extension

目标：减少 `extract_snapshots` 的 Python/SWIG 开销。

当前已新增：

```text
pytexgen._fastdata
pytexgen._Core._fastdata_extract_snapshot_bundle_direct
```

当前技术选型：

```text
_Core 内 C++ 直连提取
SWIG runtime 只负责入口指针转换
_fastdata 只做 provider facade
```

接口：

```python
extract_snapshot_bundle(textile)
_Core._fastdata_extract_snapshot_bundle_direct(textile)
```

难度：中。

### Phase 4：JIT 分类内核

先做 CPU：

```text
Numba classify_voxels_numba
```

再做 GPU：

```text
Triton classify_voxels_triton
```

难度：中到高。

### Phase 5：成熟 GPU 后端

如果 Triton 原型收益明确，再考虑：

```text
PyTorch custom C++/CUDA op
```

用于长期稳定、与 torch.compile / dispatcher / packaged wheels 集成。

难度：高。

## 关于“迁移到代码里的难度”

最现实的代码迁移不是“重写一切”，而是新增并行通道。

推荐目录形态：

```text
TexGen/
  gpu_voxelizer.py        # 当前 Python numpy/torch 路径
  acdm_solver.py          # 当前 Voxel-ACDM adapter

src/pytexgen/
  __init__.py
  _core_docs.py

fastdata/ or CppFastData/
  CMakeLists.txt
  fastdata.cpp            # nanobind/pybind11
  snapshot_extract.cpp
  voxel_output.cpp
```

CMake 侧：

```text
TEXGEN_ENABLE_FASTDATA=ON/OFF
TEXGEN_FASTDATA_BINDING=nanobind/pybind11
```

Python 侧：

```python
try:
    from ._fastdata import extract_snapshot_bundle
except ImportError:
    extract_snapshot_bundle = None
```

这样 wheel 构建可以先不强制依赖新扩展；没有 fastdata 时走当前 Python fallback。

难度判断：

```text
文档和 Python adapter：低
可选 C++ extension：中
默认启用 C++ extension：中到高
替换 SWIG：极高
```

## JIT 使用建议

当前不要把 JIT 当成统一方案。建议按以下优先级：

1. 先做 benchmark harness  
   固定 24^3、64^3、128^3，分别统计：

   ```text
   model
   extract
   classify
   material/phase map
   solver build
   solver solve
   ```

2. 如果 classify 成为瓶颈，再做 Numba CPU 内核。

3. 如果目标是 Voxel-ACDM CUDA/Triton，全力做 Triton classifier，而不是 torch.compile 当前函数。

4. torch.compile 只用于重构后的纯 torch 函数，或 solver-side tensor assembly，不用于 SWIG/extract。

5. FFT numpy 求解器本身不要优先 JIT。它的核心是 FFT 和 einsum，优化方向应是：

   ```text
   更好的 FFT backend
   GPU FFT
   Voxel-ACDM CUDA FEM path
   ```

## 最终建议

短期：

```text
保留 SWIG
完善 VoxelGridData
补 DLPack
补 orientation field
继续维护 numpy FFT 和 Voxel-ACDM adapter
```

中期：

```text
新增 nanobind/pybind11 fastdata extension
减少 SWIG extract 开销
实现 Numba CPU classifier
```

长期：

```text
Triton GPU voxelizer
PyTorch custom op
与 Voxel-ACDM CUDA/Triton 求解器零拷贝直连
```

这条路线的核心原则是：

```text
SWIG 管对象兼容
nanobind/pybind11 管高性能数据出口
DLPack/torch 管张量交接
Triton/custom op 管大规模 GPU kernel
```

## References

- SWIG Python documentation: https://www.swig.org/Doc4.3/Python.html
- pybind11 documentation: https://pybind11.readthedocs.io/en/stable/basics.html
- nanobind documentation: https://nanobind.readthedocs.io/en/latest/why.html
- cppyy documentation: https://cppyy.readthedocs.io/
- Python Stable ABI / Limited API: https://docs.python.org/3/c-api/stable.html
- HPy documentation: https://docs.hpyproject.org/
- DLPack Python specification: https://dmlc.github.io/dlpack/latest/python_spec.html
- Apache Arrow C Data Interface: https://arrow.apache.org/docs/format/CDataInterface.html
- Apache Arrow C Device Data Interface: https://arrow.apache.org/docs/format/CDeviceDataInterface.html
- PyTorch custom C++/CUDA operators: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- Triton introduction: https://openai.com/index/triton/
- Numba JIT documentation: https://numba.pydata.org/numba-doc/0.40.0/reference/jit-compilation.html
