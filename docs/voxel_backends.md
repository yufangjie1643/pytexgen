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

Use the direct solver data path below when downstream code needs arrays or
tensors instead of an Abaqus input deck.

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

For anisotropic downstream solvers, pass `include_orientations=True` to return
two per-voxel direction grids:

```python
data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="numpy",
    output="numpy",
    include_orientations=True,
)

orientation1 = data.orientation1  # yarn tangent, shape (nz, ny, nx, 3)
orientation2 = data.orientation2  # yarn up vector, shape (nz, ny, nx, 3)
```

Dense storage fills matrix voxels with zero direction vectors. Both NumPy and
Torch classifiers also support `orientation_storage="sparse"`, which puts only
yarn-voxel directions in `data.sparse_orientation`. With
`backend="torch", output="backend"`, these compact arrays remain on the
selected CUDA/MPS/CPU device.

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

data.save_npy_dir("weave_64_npy")
mmap_loaded = VoxelGridData.load_npy_dir("weave_64_npy", mmap_mode="r")
```

`.npz` is convenient for compact single-file storage. The directory-based
`.npy` format stores `metadata.json` plus one raw array file per field, so large
datasets can be loaded with `mmap_mode="r"` and avoid zip decompression CPU
overhead during training.

## Sparse Material Direction and Stiffness Fields

The material-field workflow performs orientation capture, per-yarn material
lookup, and stiffness rotation with the selected array backend. On CUDA, no
NumPy conversion occurs until an explicit save or `output="numpy"` operation:

```python
from pytexgen.material_fields import (
    isotropic_stiffness_c21,
    orthotropic_stiffness_c21,
    load_material_field_bundle,
    save_material_field_bundle,
    voxelize_textile_material_fields,
)

matrix = isotropic_stiffness_c21(3.5e9, 0.35)
yarn = orthotropic_stiffness_c21(
    150e9, 10e9, 10e9, 0.25, 0.25, 0.30, 5e9, 5e9, 3.8e9
)
data, field = voxelize_textile_material_fields(
    textile,
    nx=128, ny=128, nz=128,
    backend="torch", device="cuda", output="backend",
    matrix_stiffness=matrix,
    default_yarn_stiffness=yarn,
    yarn_stiffness_by_id={3: 1.1 * yarn},
)

c21_grid = field.to_dense_c21()  # (Nz, Ny, Nx, 21)
c66_grid = field.to_dense_voigt() # (6, 6, Nz, Ny, Nx)
acdm_grid = field.to_acdm()       # (1, 6, 6, Nz, Ny, Nx)

save_material_field_bundle("fields", data.sparse_orientation, field)
orientation, reloaded = load_material_field_bundle("fields", mmap_mode="r")
```

Engineering-Voigt component order is exactly
`(xx, yy, zz, yz, xz, xy)`. C21 is the row-major upper triangle:
`(C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66)`.
The matrix uses one shared `matrix_c21` and has no orientation entries; yarn
voxels store sorted flat indices, yarn IDs, two direction vectors, material
IDs, and rotated C21 values. Directory persistence produces raw `.npy` arrays
that support `mmap_mode="r"`; a path ending in `.npz` produces one archive.
Saving either form is an explicit GPU-to-CPU transfer boundary.

## Simulation and Training Sample Contract

`SimulationSample` composes the existing voxel, sparse direction, sparse C21,
and material-table objects without copying their arrays:

```python
import numpy as np
import torch

from pytexgen.simulation_sample import (
    MaterialTable,
    voxelize_textile_simulation_sample,
)

materials = MaterialTable(
    c21=np.stack([matrix, yarn]),
    material_ids=np.array([0, 7], dtype=np.int32),
    unit="Pa",
)
sample = voxelize_textile_simulation_sample(
    textile,
    materials=materials,
    default_yarn_material_id=7,
    nx=128, ny=128, nz=128,
    backend="torch", device="cuda", output="backend",
)

field = sample.array("stiffness.yarn_c21")
alias = torch.from_dlpack(field)
assert alias.data_ptr() == field.data_ptr()
```

Stable names cover voxel yarn/physical-material/occupancy grids, sparse
direction frames, sparse C21 rows, and the material table. Native resident
fields are zero-copy. Derived occupancy/material grids and dense
`layout="acdm"` require `copy=True`; unsupported implicit allocation raises
instead of silently copying.

The sample itself does not implement `__dlpack__` because it contains multiple
arrays. Pass one result from `sample.array(...)` to Torch, JAX, CuPy, or Warp.
CUDA stream ordering remains the array library's standard DLPack
responsibility. Within one PyTorch process, callers that mutate an existing
tensor on a different stream must establish the normal event dependency before
consumption.

Consolidated persistence stores sparse indices and yarn IDs once:

```python
from pytexgen.simulation_io import (
    load_simulation_sample,
    save_simulation_sample,
)

save_simulation_sample("weave_sample", sample)
mmap_sample = load_simulation_sample("weave_sample", mmap_mode="r")
cuda_sample = load_simulation_sample(
    "weave_sample.npz", output="torch", device="cuda"
)
```

The manifest records schema/version, filenames, shapes, dtypes, order, units,
material names, generation metadata, and PyTexGen provenance. Directory arrays
remain memory-mappable. Saving a CUDA sample is an explicit device-to-host
boundary.

## Versioned Training Dataset Pipeline

The single-sample format is suitable for exchange; `training_io` is the
high-throughput format for a labelled collection. Version 1 uses uncompressed
native `.npy` shards so each DataLoader worker can memory-map selected fields
without decompressing or loading the whole dataset:

```text
dataset/
  dataset.json
  samples.jsonl
  rejections.jsonl
  shards/shard_00000/
    voxel_material_id.npy
    orientation_primary.npy
    yarn_voxels_offsets.npy
    effective_c21.npy
    shard.json
```

`TrainingFieldSpec` fixes the role, layout, exact dtype, trailing shape, unit,
semantic, and optional ragged group of every field. A fixed field has one row
per sample. A ragged field stores concatenated values plus `int64` offsets;
fields in the same group share offsets. This keeps matrix voxels out of sparse
direction and stiffness arrays. Version 1 deliberately requires one
`grid_shape` per dataset and never silently casts, resamples, densifies, repairs
labels, infers units, or moves a CUDA sample to the host.

Create group-safe splits before generation:

```python
from pytexgen.training_data import deterministic_group_split

assignment = deterministic_group_split(
    parent_geometry_ids,
    ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    seed=2026,
    strata=weave_family_by_geometry,  # optional
)
```

The assignment is stable across input order. Give every parameter variant and
augmentation of one parent geometry the same `group_id`; the writer rejects
group leakage across splits and duplicate geometry digests.

Publish samples through a staging directory:

```python
from pytexgen.training_io import SimulationDatasetWriter

writer = SimulationDatasetWriter.create(
    "rve_dataset",
    schema=schema,
    quality=quality,
    generation={"generator_commit": generator_commit, "seed": 2026},
    resume=False,
)
if solver_converged:
    writer.append(
        cpu_numpy_sample,
        targets={"effective_c21": effective_c21},
        sample_id="rve-000042",
        group_id="parent-000017",
        split=assignment["parent-000017"],
        provenance=solver_provenance,
        metadata={"weave_family": "plain"},
    )
else:
    writer.reject(
        sample_id="rve-000042",
        stage="label",
        reason="PCG residual exceeds the configured threshold",
    )
writer.finalize()
```

Each completed shard is fsynced, checksummed, and journaled. `resume=True`
continues only when the serialized schema, quality policy, and generation
configuration digest match; incomplete trailing shards are discarded. Finalize
publishes by atomic directory rename. Existing published targets are never
overwritten.

For effective engineering-Voigt C21 labels, provenance must include solver
commit, element formulation, arithmetic dtype, tolerance, observed maximum
residual, iteration count, wall time, and target units. The quality policy can
require positive-definite targets and enforce a maximum residual. Solver
arithmetic dtype is independent of storage dtype: an FP32 Voxel-ACDM result may
be explicitly converted to an FP64 label while retaining `"float32"` in its
provenance.

Audit before training or archival:

```python
from pytexgen.training_io import audit_simulation_dataset

report = audit_simulation_dataset("rve_dataset", verify="sample")
assert report["ok"]
```

Verification levels are `manifest` (metadata structure), `shard` (checksums
when selected arrays open), and `sample` (all checksums plus geometry digests).
All manifest paths are constrained to the dataset root.

### Selective Loading, Augmentation, and CUDA Prefetch

The reader is map-style, picklable, and resets its mmap cache after process
spawn. Only selected fields are opened:

```python
from pytexgen.training_data import CubicRotation
from pytexgen.training_io import SimulationDataset

rotation = CubicRotation(seed=41, probability=0.5)
train = SimulationDataset(
    "rve_dataset",
    split="train",
    inputs=("voxel.material_id", "orientation.primary"),
    targets=("effective_c21",),
    verify="shard",
    transform=rotation,
)
train.set_epoch(3)
```

`CubicRotation` chooses deterministically from the 24 proper cube rotations and
transforms coupled spatial grids, sparse indices, directions, global stiffness
C21, AABBs, and effective C21 labels. Local material-table C21 is invariant.
No reflections or interpolated arbitrary-angle rotations are applied.

```python
from pytexgen.torch_training import CudaPrefetcher, make_simulation_dataloader

loader = make_simulation_dataloader(
    train,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    seed=41,
)

prefetch = CudaPrefetcher(loader, device="cuda")
for batch in prefetch:
    # Fixed fields are tensors; ragged fields expose .values and .offsets.
    ids = batch.inputs["voxel.material_id"]
    directions = batch.inputs["orientation.primary"]
    prediction = model((ids != 0).float().unsqueeze(1))
    loss = criterion(
        prediction, batch.targets["effective_c21"].float()
    )
    loss.backward()
```

Collation first creates one owned contiguous CPU batch. PyTorch views that
allocation without a copy, pins the custom batch, and leaves CUDA initialization
in the main process. `CudaPrefetcher` overlaps non-blocking H2D transfer on a
dedicated stream, records consumer streams for allocator safety, and reports
`transferred_bytes`, `wait_seconds`, and `recorded_tensors`. Individual tensors
can be passed directly through DLPack to JAX, CuPy, or Warp.

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
directly in C++ and returns owned contiguous numpy ndarrays through the NumPy C
API. `voxelize_snapshot_bundle_data(...)` consumes this flat bundle directly on
the numpy backend, avoiding a round trip back to per-yarn `YarnSnapshot` Python
objects. `_fastdata` deliberately does not link `TexGenCore`, because wheel
builds link `TexGenCore` statically into `_Core`; linking it again would create
a second TexGen singleton. If the SWIG wrapper is regenerated, keep the
`_fastdata_extract_snapshot_bundle_direct` shim in `_Core` or add an equivalent
explicit export.

`VoxelGridData.to_dlpack("yarn_id" | "material_id" | "occupancy")` remains a
legacy capsule helper. New integrations should pass the array returned by
`SimulationSample.array(...)` directly to a DLPack consumer.

### Fastdata Pipeline Benchmark

Use `bench_fastdata_pipeline.py` to measure the real TexGen pipeline in phases:

```powershell
.\.venv\Scripts\python.exe bench_fastdata_pipeline.py --resolution 16 --repeat 1
```

The benchmark reports `construct_assign_domain`, `build_refine`,
`snapshot_direct_core`, `snapshot_python_fallback`, and
`bundle_numpy_pack`, `voxel_centers`, `voxel_classify_numpy_flat`, and
`voxel_numpy_from_direct_total` separately. Pass `--refine` for TexGen's
refined weave path, increase `--resolution` to stress voxel classification, and
use `--json-out path.json` when comparing optimization commits.

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

Material direction/stiffness integration and acceptance benchmark:

```bash
python bench_gpu_material_fields.py \
  --resolutions 16 --repeat 1 --warmup 1 --device cuda \
  --json-out build/material_fields_smoke.json

python bench_gpu_material_fields.py \
  --resolutions 128 256 --repeat 3 --warmup 1 --device cuda \
  --json-out build/material_fields_acceptance.json
```

Each case records median/P90 time, GPU memory, RSS, output bytes, occupancy and
yarn mismatches, direction-axis agreement, stiffness error, and environment
metadata. `compute` compares with TexGen `GetPointInformation`; `practical`
compares compact GPU persistence with `SaveVoxelMesh` plus `.ori/.eld` parsing.
The large-run acceptance gate requires all two-model/two-resolution/two-mode
records to be correct and at least 5x faster.

### Simulation Interoperability Acceptance

`bench_simulation_interop.py` checks the complete in-memory sample boundary:

```bash
python bench_simulation_interop.py \
  --resolution 16 --repeat 3 --device cpu --min-speedup 0 --check \
  --json-out build/simulation_interop_cpu.json

python bench_simulation_interop.py \
  --resolution 64 --repeat 5 --device cuda --min-speedup 5.0 --check \
  --json-out build/simulation_interop_cuda.json
```

Each path has one independent warm-up. Reported `pytexgen_seconds` and
`texgen_cpu_seconds` are medians of at least three synchronized runs. The CPU
reference is the original `CRectangularVoxelMesh.SaveVoxelMesh` export plus
`.eld`/`.ori` parsing; PyTexGen produces the voxel, sparse direction, rotated
C21, and material fields directly on the requested device.

The JSON records resolution, environment and GPU metadata, resident sparse and
dense ACDM byte counts, timings, speedup, CPU field agreement, and FP32/FP64
tolerances. `handoff_host_transfer_bytes` is zero only when every resident
Torch field is consumed directly with `torch.from_dlpack`, remains on the same
device, and has the same data pointer. Validation copies used after timing to
compare phase IDs and dense ACDM values are not solver-handoff traffic.
`--check` exits nonzero on pointer/device changes, nonzero handoff transfer,
phase mismatch, stiffness tolerance failure, CPU-reference failure, or
insufficient speedup.
