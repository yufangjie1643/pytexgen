# Voxel Backends

pytexgen exposes two practical voxelization paths in normal wheel installs and
one advanced local-build path for p4est users.

## Backend Selection

| Path | Entry point | Dependencies | Best use |
|---|---|---|---|
| C++ rectangular voxel mesh | `CRectangularVoxelMesh.SaveVoxelMesh(...)` | bundled TexGen core | Reference-compatible structured voxel output |
| Python numpy backend | `voxelize_textile(..., backend="numpy")` | `numpy` | Portable OpenMP-free CPU voxelization |
| Python torch backend | `voxelize_textile(..., backend="torch")` | `torch`, optional CUDA/MPS | GPU or torch-accelerated structured voxelization |
| Direct data handoff | `voxelize_textile_data(...)` | `numpy`, optional `torch` | Solver integration without `.inp` write/read overhead |
| Python adaptive numpy backend | `voxelize_textile(..., backend="numpy", adaptive=True)` | `numpy` | Lightweight non-uniform exploratory voxel output |
| C++ p4est octree mesh | `COctreeVoxelMesh.SaveVoxelMesh(...)` | local p4est/sc build | Advanced p4est-based octree refinement |

Default wheels intentionally avoid OpenMP, p4est, native CPU flags, and SWIG at
install time. That keeps `pip install pytexgen` more reliable across Windows,
Linux, macOS, and older CPUs.

## Python Structured Backend

The Python voxelizer snapshots TexGen yarn geometry into numpy arrays, classifies
voxel centers, and writes Abaqus `C3D8R` elements. It is the default structured
voxel path for users who want portable installs without C++ OpenMP runtime
issues.

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

The public APIs default to `backend="numpy"` and `chunk_voxels=8192`. Numpy
parallelism is chunk based: a `40x40x40` grid has 64,000 voxel centers, so the
default chunk size creates eight classification tasks. The reported `workers`
value is the effective number of worker threads after this chunk count is
applied.

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

## Direct Solver Data

Use `voxelize_textile_data(...)` when a downstream solver can consume arrays or
tensors directly:

```python
from pytexgen.gpu_voxelizer import voxelize_textile_data

data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="torch",
    device="cuda",
    output="backend",
)

grid = data.grid              # yarn ids, shape (nz, ny, nx)
materials = data.material_id() # matrix=0, yarn 0 -> 1, yarn 1 -> 2, ...
spacing = data.voxel_size
```

`data.yarn_id` is the flat TexGen element-order array
`ix + iy*nx + iz*nx*ny`. `data.to("numpy", dtype=...)` and
`data.to("torch", device=..., dtype=...)` convert storage explicitly,
following the container-level spirit of `torch.Tensor.to(...)`. The `dtype`
argument applies to floating metadata such as `aabb` and optional `centers`;
integer label arrays such as `yarn_id` stay integer. With
`backend="torch", output="backend"`, the
classification result stays as a torch tensor on the selected device, avoiding
an immediate CPU copy.

Direct voxel data can be saved and reloaded without text mesh files:

```python
from pytexgen.gpu_voxelizer import VoxelGridData

data.save_npz("weave_64.npz")
loaded = VoxelGridData.load_npz("weave_64.npz", output="numpy")
torch_loaded = VoxelGridData.load_npz("weave_64.npz", output="torch", device="cuda")
```

When the same textile is voxelized repeatedly, cache the TexGen geometry
snapshot once:

```python
from pytexgen.gpu_voxelizer import VoxelizationCache

cache = VoxelizationCache.from_textile(textile)

data24 = cache.voxelize(nx=24, ny=24, nz=24, backend="numpy", output="numpy")
data64 = cache.voxelize(nx=64, ny=64, nz=64, backend="torch", device="cuda")
```

This skips repeated `extract_snapshots(textile)` calls and only reruns voxel
center classification for each resolution/backend.

For extension authors, the stable high-throughput boundary is
`SnapshotBundle`, a structure-of-arrays representation of the same geometry:

```text
positions/tangents/ups/sides + node_offsets
sections + section_offsets
translations + translation_offsets
aabb
```

`extract_snapshot_bundle(textile)` first looks for an optional compiled
`_fastdata` provider with `extract_snapshot_bundle(...)`. If no provider is
installed, it falls back to the current SWIG-based extraction. Providers may
return either a `SnapshotBundle` instance or a mapping with the same array
fields; both paths are validated for array shapes, monotonic offsets, and
consistent yarn counts. Use `fastdata_provider_status()` to report whether the
compiled provider is active, and use `voxelize_snapshot_bundle_data(...)` when
a compiled provider or a precomputed bundle already exists.

The compiled provider is `pytexgen._fastdata`, a small CPython facade over
`pytexgen._Core._fastdata_extract_snapshot_bundle_direct(...)`. `_Core` converts
the SWIG proxy to a `TexGen::CTextile*` once, then extracts yarn geometry
directly in C++ and returns contiguous numpy arrays. `_fastdata` deliberately
does not link `TexGenCore`, because wheel builds link `TexGenCore` statically
into `_Core`; linking it again would create a second TexGen singleton. If the
SWIG wrapper is regenerated, keep the `_fastdata_extract_snapshot_bundle_direct`
shim in `_Core` or add an equivalent explicit export.

`VoxelGridData.to_dlpack("yarn_id" | "material_id" | "occupancy")` exports a
DLPack capsule through torch when a downstream tensor library wants to consume
the voxel labels without an Abaqus text-file round trip.

See [torch_voxel_data_flow.md](torch_voxel_data_flow.md) for the full data flow
from TexGen model generation to torch tensor output and matrix norm benchmark.
See [voxel_acdm_interface.md](voxel_acdm_interface.md) for the direct
Voxel-ACDM solver adapter.
See [cross_language_modernization_report.md](cross_language_modernization_report.md)
for binding, data-exchange, and JIT modernization options.

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

Synthetic pruning and direct data benchmark:

```bash
python bench_gpu_voxelizer_backends.py --resolution 64 --yarn-grid 4 --workers 4
```

Torch/CUDA benchmark when torch is installed:

```bash
python bench_gpu_voxelizer_backends.py --include-torch --device cuda
```
