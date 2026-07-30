# Voxel Backends

PyTexGen provides the original C++ structured voxel mesh and a portable Python
backend for direct NumPy or Torch arrays. The Python path is intended for
geometry preparation, material-field construction, and batch export.

## Backend Selection

| Path | Entry point | Best use |
| --- | --- | --- |
| TexGen C++ | `CRectangularVoxelMesh.SaveVoxelMesh(...)` | Reference-compatible Abaqus output |
| NumPy | `voxelize_textile_data(..., backend="numpy")` | Portable CPU arrays |
| Torch | `voxelize_textile_data(..., backend="torch")` | CUDA, MPS, or Torch CPU arrays |
| Adaptive NumPy | `voxelize_textile(..., adaptive=True)` | Exploratory non-uniform C3D8R meshes |

The default structured Python backend is NumPy. Torch is imported lazily and
is only required when selected.

## Structured Voxel Data

```python
from pytexgen.gpu_voxelizer import voxelize_textile_data

data = voxelize_textile_data(
    textile,
    nx=128,
    ny=128,
    nz=64,
    backend="numpy",
    workers=4,
    output="numpy",
)

yarn_ids = data.grid
material_ids = data.material_id()
```

Structured arrays use `(Nz, Ny, Nx, ...)` shapes. Flat voxel order is:

```text
ix + iy*nx + iz*nx*ny
```

Use `data.save_npz(...)` for a compact archive or `data.save_npy_dir(...)`
when individual arrays should remain memory-mappable.

For CUDA:

```python
data = voxelize_textile_data(
    textile,
    nx=128,
    ny=128,
    nz=128,
    backend="torch",
    device="cuda",
    output="backend",
)
```

`output="backend"` keeps arrays on the selected backend. Requesting NumPy from
a CUDA run is an explicit device-to-host transfer.

## Orientation and Material Fields

Sparse orientation avoids allocating direction vectors for matrix voxels:

```python
data = voxelize_textile_data(
    textile,
    nx=128,
    ny=128,
    nz=128,
    backend="torch",
    device="cuda",
    output="backend",
    include_orientations=True,
    orientation_storage="sparse",
)
```

Build rotated engineering-Voigt stiffness without leaving the selected
backend:

```python
from pytexgen.material_fields import (
    build_stiffness_field,
    isotropic_stiffness_c21,
    orthotropic_stiffness_c21,
)

matrix_c21 = isotropic_stiffness_c21(3.5e9, 0.35)
yarn_c21 = orthotropic_stiffness_c21(
    150e9, 10e9, 10e9,
    0.25, 0.25, 0.30,
    5e9, 5e9, 3.8e9,
)
stiffness = build_stiffness_field(
    data,
    matrix_stiffness=matrix_c21,
    default_yarn_stiffness=yarn_c21,
    output="sparse",
    unit="Pa",
)
```

C21 uses engineering-Voigt component order
`(xx, yy, zz, yz, xz, xy)` and stores the row-major upper triangle. Dense
material views are created only when requested.

## Prepared Geometry and File Batches

TG3 remains the editable model. PTGB is a read-only, memory-mappable cache of
the flattened geometry required by the voxelizer:

```python
from pytexgen.batch import MaterialSpec, prepare_geometry, voxelize_files_batch

prepare_geometry("plain.tg3", "plain.ptgb")

report = voxelize_files_batch(
    ["plain.ptgb", "twill.tg3"],
    resolution=(128, 128, 128),
    output_dir="voxel_output",
    fields=("material_id", "orientation", "stiffness_c21"),
    materials=MaterialSpec(
        matrix_c21=matrix_c21,
        default_yarn_c21=yarn_c21,
        unit="Pa",
    ),
    device="cuda",
    dtype="float32",
    batch_size="auto",
    memory_budget_bytes=12 << 30,
)
```

One directory is written per source, using uncompressed `.npy` arrays and a
`metadata.json` manifest. GPU computation streams one geometry at a time while
bounded background writers overlap disk output with subsequent work. See
[PTGB v1](ptgb_v1.md) for the byte-level cache format.

## Performance Controls

- `chunk_voxels` bounds the center-classification working set.
- `workers` controls NumPy chunk workers and is clamped to available chunks.
- `aabb_pruning=True` skips yarn translations that cannot intersect a chunk.
- `orientation_storage="sparse"` avoids dense matrix-voxel directions.
- `batch_size` bounds pending output writes rather than stacking unrelated
  geometries into one large tensor.
- `memory_budget_bytes` rejects a dense-output configuration before work
  begins when its conservative allowance is too large.

Use `float32` for production CUDA throughput unless downstream accuracy
requires `float64`.

## Adaptive NumPy Mode

`adaptive=True` writes non-uniform Abaqus C3D8R cells. It is intended for
exploration and does not provide p4est-style 2:1 balancing or hanging-node
constraint equations. Use a p4est-enabled native build when those guarantees
are required.

## Verification

Run the focused backend tests:

```bash
python test_gpu_voxelizer_backends.py
python test_material_fields.py
python test_batch.py
```

Run portable and CUDA benchmarks:

```bash
python bench_gpu_voxelizer_backends.py \
  --resolution 64 --yarn-grid 4 --workers 4

python bench_gpu_voxelizer_backends.py \
  --include-torch --device cuda

python bench_gpu_material_fields.py \
  --resolutions 16 --repeat 1 --warmup 1 --device cuda \
  --json-out build/material_fields_smoke.json
```

The default wheel intentionally excludes source-only solver, training, and
experimental Torch mesher workflow extensions. Local custom builds can opt in
with `-DTEXGEN_INSTALL_WORKFLOW_EXTENSIONS=ON`.
