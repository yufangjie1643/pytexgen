# Voxel Backends

pytexgen exposes two practical voxelization paths in normal wheel installs and
one advanced local-build path for p4est users.

## Backend Selection

| Path | Entry point | Dependencies | Best use |
|---|---|---|---|
| C++ rectangular voxel mesh | `CRectangularVoxelMesh.SaveVoxelMesh(...)` | bundled TexGen core | Reference-compatible structured voxel output |
| Python numpy backend | `voxelize_textile(..., backend="numpy")` | `numpy` | Portable OpenMP-free CPU voxelization |
| Python torch backend | `voxelize_textile(..., backend="torch")` | `torch`, optional CUDA/MPS | GPU or torch-accelerated structured voxelization |
| Python adaptive numpy backend | `voxelize_textile(..., backend="numpy", adaptive=True)` | `numpy` | Lightweight non-uniform exploratory voxel output |
| C++ p4est octree mesh | `COctreeVoxelMesh.SaveVoxelMesh(...)` | local p4est/sc build | Advanced p4est-based octree refinement |

Default wheels intentionally avoid OpenMP, p4est, native CPU flags, and SWIG at
install time. That keeps `pip install pytexgen` more reliable across Windows,
Linux, macOS, and older CPUs.

## Python Structured Backend

The Python voxelizer snapshots TexGen yarn geometry into numpy arrays, classifies
voxel centers, and writes Abaqus `C3D8R` elements. It is a replacement path for
users who want portable installs without C++ OpenMP runtime issues.

```python
from pytexgen.gpu_voxelizer import voxelize_textile

info = voxelize_textile(
    textile,
    nx=64, ny=64, nz=64,
    out_inp="mesh_numpy.inp",
    backend="numpy",
    workers=4,
    aabb_pruning=True,
)
```

`aabb_pruning=True` is enabled by default. It skips yarn/translation candidates
whose conservative bounding boxes cannot overlap the current voxel chunk. This
does not change classification results in the backend tests, but the speedup
depends on textile density, yarn count, and voxel resolution.

Use `backend="torch"` when torch is installed:

```python
info = voxelize_textile(
    textile,
    nx=64, ny=64, nz=64,
    out_inp="mesh_torch.inp",
    backend="torch",
    device="cuda",  # or "mps" / "cpu"
)
```

Torch is most useful for larger structured grids. Small grids can be slower on
GPU because transfer, kernel setup, and synchronization overhead dominate.

## Adaptive Numpy Mode

Adaptive numpy mode is a lightweight linear-octree output path:

```python
info = voxelize_textile(
    textile,
    nx=16, ny=16, nz=8,
    out_inp="mesh_adaptive_numpy.inp",
    backend="numpy",
    adaptive=True,
    adaptive_levels=2,
    max_adaptive_cells=2_000_000,
)
```

Current behavior:

- Starts from the requested structured base grid.
- Samples each candidate cell at the center and eight corners.
- Refines cells where those samples disagree on yarn ownership.
- Classifies final leaf cells by center point.
- Writes non-uniform Abaqus `C3D8R` elements and yarn/matrix element sets.

Important limits:

- It does not generate p4est-style 2:1 balancing.
- It does not generate hanging-node constraint equations.
- It is currently numpy-only; `adaptive=True, backend="torch"` is rejected.
- It is intended for portable exploratory output and refinement experiments, not
  as a full replacement for p4est-based octree FEM workflows.

Use the p4est path when downstream FEM tooling requires a p4est-style balanced
octree mesh or a project depends specifically on TexGen's `COctreeVoxelMesh`.

## Advanced p4est Build

`COctreeVoxelMesh` is guarded by `TEXGEN_USE_P4EST` in the C++ code. In normal
wheel/SKBUILD builds, p4est is deliberately disabled:

- `Core/CMakeLists.txt` prints `SKBUILD mode: skipping prebuilt p4est/sc`.
- The committed `Python/Core_wrap.cxx` omits `COctreeVoxelMesh`.
- `TEXGEN_REGENERATE_SWIG=OFF` means users do not need SWIG for normal installs.

To expose `COctreeVoxelMesh`, use a local legacy CMake build instead of the
default wheel path:

1. Build or obtain matching p4est/sc libraries for the same compiler and runtime.
2. Place the libraries where this project already looks:
   - Windows: `OctreeRefinement/libp4est.lib` and `OctreeRefinement/libsc.lib`
   - Unix-like: `OctreeRefinement/libp4est.*` and `OctreeRefinement/libsc.*`
3. Install SWIG.
4. Configure with SWIG regeneration enabled:

```bash
cmake -S . -B build-p4est \
  -DBUILD_PYTHON_INTERFACE=ON \
  -DTEXGEN_REGENERATE_SWIG=ON \
  -DBUILD_RENDERER=OFF \
  -DBUILD_GUI=OFF
cmake --build build-p4est --config Release
cmake --install build-p4est
```

On Windows, the p4est/sc libraries must match the compiler ABI used for the
TexGen build. Mixing MSVC and MinGW libraries is not supported.

## Verification

Backend smoke tests:

```bash
python test_gpu_voxelizer_backends.py
```

Synthetic pruning benchmark:

```bash
python bench_gpu_voxelizer_backends.py --resolution 32 --yarn-grid 4 --workers 4
```

Torch/CUDA benchmark when torch is installed:

```bash
python bench_gpu_voxelizer_backends.py --include-torch --device cuda
```
