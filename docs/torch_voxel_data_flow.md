# Torch Voxel Data Flow

本文说明从 TexGen 模型生成、直接体素化到 torch 张量、再做矩阵范数计算的真实数据流。对应接口是 `voxelize_textile_data(...)`，它跳过 Abaqus `.inp` 文件写出，适合把体素结果直接交给其他求解器。

## 入口

```python
from pytexgen.gpu_voxelizer import voxelize_textile_data

data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="torch",
    device="cuda",
    output="backend",
)
```

关键参数：

- `backend="torch"`：用 torch 后端做体素分类。
- `device="cuda"`：把分类计算放到 NVIDIA GPU 上。
- `output="backend"`：保留后端原生结果。torch 后端下，`yarn_id` 不会立刻拷回 CPU。

## 1. TexGen 模型对象

上游脚本先生成一个 TexGen `CTextile` 对象，例如 2D weave、3D layer-to-layer、SiC/SiC RVE 等。此时几何仍在 TexGen C++/SWIG 对象里，包含：

- yarn centerline 和 slave nodes
- yarn tangent / up / side 局部坐标系
- yarn section polygon
- periodic domain translations
- domain mesh / AABB

这个阶段还不是 numpy 或 torch 数据。

## 2. 几何快照

`voxelize_textile_data(...)` 内部先调用：

```python
snapshots, aabb = extract_snapshots(textile)
```

这一步把 TexGen C++/SWIG 对象转成 Python 可处理的 numpy 快照。每根 yarn 会生成一个 `YarnSnapshot`：

```text
positions     numpy.ndarray, shape (M, 3)
tangents      numpy.ndarray, shape (M, 3)
ups           numpy.ndarray, shape (M, 3)
sides         numpy.ndarray, shape (M, 3)
section       numpy.ndarray, shape (N, 2)
translations  numpy.ndarray, shape (K, 3)
```

其中：

- `M` 是该 yarn 的 slave node 数量。
- `N` 是截面多边形点数。
- `K` 是周期性平移数量。
- `aabb` 是 domain 的轴对齐包围盒，shape 为 `(2, 3)`。

这是从 TexGen 对象到数组世界的第一次转换。

## 3. 构造体素中心

对 `64 * 64 * 64` 体素：

```text
V = 64 * 64 * 64 = 262,144
centers_np.shape = (262144, 3)
```

体素中心由 `_structured_voxel_centers(...)` 生成，顺序是 TexGen element 顺序：

```text
ix + iy*nx + iz*nx*ny
```

也就是 x 最内层、z 最外层。后续 `yarn_id` 的扁平数组也沿用这个顺序。

## 4. 搬到 Torch / CUDA

torch 后端会做两件事。

第一，打包 yarn 几何：

```python
packed = _pack_yarns(
    snapshots,
    device="cuda",
    dtype=torch.float32,
)
```

打包后主要张量在 GPU 上：

```text
P, T, U, S     torch.float32, device cuda
Sec            torch.float32, device cuda
Tr             torch.float32, device cuda
BoundsLo/Hi    torch.float32, device cuda
```

第二，搬运体素中心：

```python
centers = torch.from_numpy(centers_np).to(
    device="cuda",
    dtype=torch.float32,
)
```

此时：

```text
centers  torch.float32, shape (262144, 3), device cuda
```

注意：当前实现里，体素中心先在 CPU numpy 中生成，再拷到 GPU。几何快照也先从 TexGen/SWIG 转成 numpy，再打包到 GPU。这两次是必要的数据边界转换；之后分类结果可以留在 GPU。

## 5. GPU 体素分类

分类调用：

```python
yarn_id = _classify_voxels_torch(
    centers,
    packed,
    chunk=65536,
    aabb_pruning=True,
)
```

64^3 体素默认会分 4 个 chunk：

```text
262144 / 65536 = 4
```

每个 chunk 内部会：

1. 计算当前体素点范围 `chunk_lo/chunk_hi`。
2. 用 `BoundsLo/BoundsHi` 做 AABB pruning，跳过不可能相交的 yarn/translation。
3. 对候选点找最近的 yarn slave node。
4. 把点投影到 yarn 局部 section 坐标 `(u, v)`。
5. 用 point-in-polygon 判断是否落在截面内。
6. 若多个 yarn 命中，用距离代理选择最近 yarn。

输出：

```text
yarn_id  torch.int32, shape (262144,), device cuda
```

语义：

```text
-1  matrix / background
 0  yarn 0
 1  yarn 1
 ...
```

## 6. VoxelGridData 封装

`voxelize_textile_data(...)` 返回 `VoxelGridData`：

```python
data.yarn_id
data.grid
data.aabb
data.voxel_size
```

torch 后端且 `output="backend"` 时：

```text
data.yarn_id  torch.int32, shape (262144,), device cuda
data.grid     zero-copy reshape, shape (64, 64, 64)
data.aabb     torch.float32, shape (2, 3), device cuda
```

`data.grid` 是 `data.yarn_id.reshape((nz, ny, nx))`，不额外复制数据。

如果调用：

```python
data.to("numpy")
```

才会发生 GPU -> CPU 拷贝。正常求解器接入 torch 张量时不需要这一步。
也可以使用 torch 风格的 dtype 参数：

```python
data_cpu = data.to("numpy", dtype="float32")
data_gpu = data.to("torch", device="cuda", dtype="float32")
```

`dtype` 只作用于 `aabb`、`centers` 这类浮点数组；`yarn_id` 这类标签数组保持整数。

## 7. 生成材料网格

求解器通常不直接使用 `-1, 0, 1, ...` 的 yarn id，而是需要材料 id。可以调用：

```python
materials = data.material_id()
```

默认映射：

```text
matrix  -1 -> 0
yarn 0   0 -> 1
yarn 1   1 -> 2
...
```

输出仍然在 GPU 上：

```text
materials  torch.int32, shape (64, 64, 64), device cuda
```

## 8. 矩阵范数计算

benchmark 中的 torch 范数路径：

```python
nx, ny, nz = data.resolution
matrix = data.material_id().to(dtype=torch.float32).reshape(nz * ny, nx)
value = torch.linalg.matrix_norm(matrix, ord="fro")
```

对 64^3 体素：

```text
materials.shape = (64, 64, 64)
matrix.shape    = (4096, 64)
value           = torch.float32 scalar, device cuda
```

这一步没有 `.inp` 文件，也不需要把 `yarn_id` 拷回 CPU。只有最终打印数值时，`value.item()` 或 `value.detach().cpu()` 会把标量传回 CPU。

## 当前 Benchmark 示例

命令：

```powershell
.\.venv\Scripts\python.exe bench_gpu_voxelizer_backends.py `
  --resolution 64 `
  --yarn-grid 4 `
  --workers 1 `
  --chunk-voxels 8192 `
  --repeat 1 `
  --include-torch `
  --device cuda
```

示例输出：

```text
voxels=262,144 yarns=16 dtype=float32 workers=1
numpy pruned:   0.0563s
direct numpy data: 0.1566s  matrix_norm(fro)=1547.13 in 0.0052s
torch/cuda pruned: 0.9973s
direct torch/cuda data: 0.8963s  matrix_norm(fro)=1547.13 in 0.0596s
```

解释：

- `direct torch/cuda data`：从 TexGen/numpy snapshot 到 GPU `VoxelGridData.yarn_id` 的直接体素化耗时。
- `matrix_norm(fro)`：从 GPU `material_id()` 到 `torch.linalg.matrix_norm(...)` 的耗时。
- benchmark 为了校验 numpy 和 torch 结果一致，会额外调用 `torch_data.to("numpy")`；这一步会产生 GPU -> CPU 拷贝，但不是求解器真实接入路径所必需的步骤。

## 面向其他求解器的推荐接入点

如果求解器使用 torch：

```python
data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="torch",
    device="cuda",
    output="backend",
)

material_grid = data.material_id()
```

如果求解器使用 numpy：

```python
data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="numpy",
    output="numpy",
)

material_grid = data.material_id()
```

避免在中间写出 `.inp` 再解析回来，这就是当前直接数据接口减少性能损失的主要位置。
