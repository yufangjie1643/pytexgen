# Voxel-ACDM Direct Solver Interface

本文记录 pytexgen 到 `git@github.com:yufangjie1643/Voxel-ACDM.git` 的直接接口设计。目标是避免 TexGen `.inp/.eld/.ori` 文件往返，把 TexGen 体素结果直接传给 Voxel-ACDM 求解器。

## 当前支持范围

当前适配层位于：

```text
TexGen/acdm_solver.py
```

已支持 Voxel-ACDM 的各向同性 phase-LUT 求解路径：

```text
CTextile
  -> voxelize_textile_data(...)
  -> VoxelGridData
  -> to_acdm_phase_ids(...)
  -> FEMHomogenizerBatchedIsotropicPhases
  -> C_eff / engineering constants
```

这个路径适合 matrix/yarn 都能先近似为各向同性材料的快速联调和求解器接入验证。

各向异性 yarn 的完整路径需要体素化阶段输出 yarn orientation field，然后接 Voxel-ACDM 的 `FEMHomogenizerBatched(C_voigt_fields=...)` 或 `build_voigt_stiffness_fields_gpu(...)`。`voxelize_textile_data(..., include_orientations=True)` 已可返回 `orientation1`/`orientation2` 体素方向场；各向异性 ACDM 适配层仍需单独接入。

## 环境发现

适配层会按以下顺序寻找 Voxel-ACDM：

1. 显式参数 `acdm_root=...`
2. 环境变量 `VOXEL_ACDM_ROOT`
3. pytexgen 同级目录 `../Voxel-ACDM`
4. 当前工作目录或其上一级下的 `Voxel-ACDM`

当前开发机上仓库位置：

```text
E:\yfj_code\Voxel-ACDM
```

## 从已有 VoxelGridData 求解

```python
from pytexgen.gpu_voxelizer import voxelize_textile_data
from pytexgen.acdm_solver import solve_acdm_isotropic_from_voxel_data

data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="numpy",
    output="numpy",
)

result = solve_acdm_isotropic_from_voxel_data(
    data,
    matrix_material={"E": 3.0e9, "Nu": 0.2},
    yarn_material={"E": 2.0e11, "Nu": 0.3},
    acdm_root=r"E:\yfj_code\Voxel-ACDM",
    device="cuda",
    dtype="fp32",
    precond="fft",
    tol=2e-6,
    max_iter=2000,
)

print(result.C_eff)
print(result.engineering_constants)
print(result.timings)
```

## 一站式体素化并求解

```python
from pytexgen.acdm_solver import voxelize_and_solve_acdm_isotropic

result = voxelize_and_solve_acdm_isotropic(
    textile,
    nx=64, ny=64, nz=64,
    matrix_material={"E": 3.0e9, "Nu": 0.2},
    yarn_material={"E": 2.0e11, "Nu": 0.3},
    acdm_root=r"E:\yfj_code\Voxel-ACDM",
    voxel_backend="numpy",
    solver_device="cuda",
    solver_dtype="fp32",
    precond="fft",
)
```

## Phase ID 映射

pytexgen 的 `VoxelGridData.yarn_id` 语义：

```text
-1  matrix
 0  yarn 0
 1  yarn 1
 ...
```

Voxel-ACDM isotropic phase-LUT 路径需要 `0..15` 的 phase id。默认映射：

```text
matrix -> phase 0
all yarns -> phase 1
```

可以自定义：

```python
from pytexgen.acdm_solver import to_acdm_phase_ids

phase_ids = to_acdm_phase_ids(
    data,
    matrix_phase=0,
    yarn_phase_by_id={
        0: 1,
        1: 2,
        2: 3,
    },
)
```

返回 shape：

```text
(B, Nz, Ny, Nx)
```

默认 `B=1`。

## 与 Voxel-ACDM 的数据约定

Voxel-ACDM 主要使用：

```text
grid_shape = (Nz, Ny, Nx)
voxel_size = (dx, dy, dz)
phase_ids  = (B, Nz, Ny, Nx), uint8/int, values 0..15
```

pytexgen `VoxelGridData` 中：

```text
data.resolution = (Nx, Ny, Nz)
data.shape      = (Nz, Ny, Nx)
data.voxel_size = (dx, dy, dz)
data.grid       = (Nz, Ny, Nx)
data.orientation1 = (Nz, Ny, Nx, 3)  # optional yarn tangent
data.orientation2 = (Nz, Ny, Nx, 3)  # optional yarn up vector
```

因此适配层只做布局确认和 phase id 映射，不需要重新排列体素顺序。

## 当前限制

- Voxel-ACDM 主 FEM 路径要求 CUDA + Triton。当前适配层只在真正调用求解器时导入 `femlib.fem_batched`。
- 当前 pytexgen 适配层只支持各向同性 phase-LUT 快路径。
- `FEMHomogenizerBatchedIsotropicPhases` 当前内部会把 phase ids 打包成 int4，因此即使 pytexgen 体素化输出是 torch tensor，调用该求解器前也会转成 numpy phase ids。
- 对各向异性 yarn，orientation field 已可由体素化阶段输出；仍需要后续装配 `C_voigt_fields` 并接到各向异性求解器。

## 下一步接口

优先补：

1. `build_acdm_voigt_field_from_voxel_data(...)`
2. `solve_acdm_anisotropic_from_voxel_data(...)`

这样才能把 TexGen yarn 局部方向和 Voxel-ACDM 的正交各向异性刚度旋转路径完整连起来。
