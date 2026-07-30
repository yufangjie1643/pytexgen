# Voxel-ACDM Direct Solver Interface

> 这是源码级工作流扩展，不包含在默认 PyTexGen wheel 中。自定义本地构建需显式启用
> `TEXGEN_INSTALL_WORKFLOW_EXTENSIONS`。

本文记录 pytexgen 到 `git@github.com:yufangjie1643/Voxel-ACDM.git` 的直接接口设计。目标是避免 TexGen `.inp/.eld/.ori` 文件往返，把 TexGen 体素结果直接传给 Voxel-ACDM 求解器。

## 当前支持范围

当前适配层位于：

```text
TexGen/acdm_solver.py
```

已支持两条直接内存路径：

```text
isotropic:
  VoxelGridData -> backend-resident phase IDs
  -> FEMHomogenizerBatchedIsotropicPhases

anisotropic:
  SimulationSample -> dense C_voigt on the current CUDA device
  -> FEMHomogenizerBatched
```

各向同性路径适合 phase-LUT 快速联调。各向异性路径接收已经验证过
方向、材料 ID、单位和 C21 顺序的 `SimulationSample`，在 GPU 上展开为
`(B,6,6,Nz,Ny,Nx)`，不会经过 NumPy。

## 环境发现

适配层会按以下顺序寻找 Voxel-ACDM：

1. 显式参数 `acdm_root=...`
2. 环境变量 `VOXEL_ACDM_ROOT`
3. pytexgen 同级目录 `../Voxel-ACDM`
4. 当前工作目录或其上一级下的 `Voxel-ACDM`

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
    phase_materials={
        0: {"E": 3.0e9, "Nu": 0.2},
        5: {"E": 2.0e11, "Nu": 0.3},
    },
    matrix_phase=0,
    yarn_phase=5,
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

旧的 `matrix_material=...` / `yarn_material=...` 参数仍兼容。新代码建议
使用显式 `phase_materials`；表可以包含未使用的 phase，但所有实际出现的
phase 必须有材料行。

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

NumPy 输入返回 NumPy；Torch 输入返回同设备 Torch tensor。所有 phase
值在转成 `uint8` 前验证，因此负数和大于 15 的值不会发生截断。先应用
`yarn_phase` 默认值，再应用 `yarn_phase_by_id` 局部覆盖。

## 各向异性 CUDA 路径

```python
from pytexgen.acdm_solver import solve_acdm_anisotropic_from_sample

result = solve_acdm_anisotropic_from_sample(
    sample,                 # Torch/CUDA SimulationSample
    dtype="fp32",
    precond="fft",
    element_type="c3d8",
)

C_eff_gpu = result.C_eff_tensor
C_eff_numpy = result.effective_stiffness_numpy()  # 显式 D2H
```

适配器要求 sample 和求解器位于同一 CUDA 设备。若目标 Voxel-ACDM
明确声明不接受 Torch `C_voigt_fields`，构造求解器之前即报兼容性错误。

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
- 当前 upstream compact phase 构造器仍通过 NumPy 打包。CUDA phase
  默认 `allow_host_phase_pack=False`，会在传输前报错；只有显式设为
  `True` 才允许旧的 CPU pack，并在 `timings` 中记录设备和字节数。
- 当后续 Voxel-ACDM 版本声明
  `SUPPORTS_CUDA_PHASE_IDS=True` 时，同一 API 会直接传入 CUDA phase。
- 通用 dense C21 路径已经同设备直连，但 Voxel-ACDM 当前的
  `compute_effective_stiffness` 仍可能自行返回 NumPy；若未来返回 Torch，
  PyTexGen 会保留 `C_eff_tensor`，不自动创建 NumPy 副本。
