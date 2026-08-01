# pytexgen

[![PyPI version](https://img.shields.io/pypi/v/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![Python](https://img.shields.io/pypi/pyversions/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

**pytexgen packages the TexGen textile geometry engine for Python and adds a
portable numpy/torch voxelization path for modern simulation workflows.**

TexGen is the open-source geometric textile modelling software developed at the
University of Nottingham. This project keeps the core TexGen modelling API
available from Python while making the package easier to install, test, and use
across Windows, Linux, and macOS.

## Version 1.2.1 Highlights

- Exact direct and batch TG3 voxelization through TexGen's authoritative C++
  classifier, parallelized with portable C++11 threads and no OpenMP runtime.
- Direct voxel data handoff with `VoxelGridData.to("numpy" | "torch")`,
  `save_npz(...)`, `load_npz(...)`, `save_npy_dir(...)`, and
  `load_npy_dir(...)`.
- Prepared `.ptgb` geometry caches and bounded batch voxelization to raw
  material, orientation, and C21 arrays.
- 2x2 weave tetrahedral mesh and small numpy/scipy FEM example scripts.
- Root `build.sh`, `build.bat`, and `build.ps1` helpers for uv-based local
  builds and installs.

## What This Project Adds

| Area | Contribution | Practical impact |
|---|---|---|
| Python packaging | `pyproject.toml` + `scikit-build-core` build path | Users can install with normal `pip` workflows instead of hand-driving CMake/SWIG |
| Stable wheel builds | Pre-generated `Python/Core.py` and `Python/Core_wrap.cxx` | Normal builds do not require a local SWIG install |
| Cross-platform defaults | GUI, renderer, native CPU flags, and p4est are off by default; CPU parallelism uses C++11 threads | Fewer Windows/MSVC/MinGW, runtime, and older-CPU build failures |
| Python voxel backend | `pytexgen.gpu_voxelizer.voxelize_textile(...)` | OpenMP-free structured voxel output through numpy or torch |
| Direct solver handoff | `pytexgen.gpu_voxelizer.voxelize_textile_data(...)` | Return numpy arrays or torch tensors without writing/parsing Abaqus files |
| GPU-ready path | Optional `backend="torch"` with CUDA/MPS/CPU devices | Larger voxel grids can use torch acceleration without changing the TexGen C++ core |
| Batch voxelization | `prepare_geometry(...)` and `voxelize_files_batch(...)` | Reuse mmap-ready geometry and stream bounded production outputs |
| Lightweight adaptive output | `adaptive=True` numpy mode | Exploratory non-uniform C3D8R voxel meshes without compiling p4est |
| Performance pruning | Conservative AABB candidate pruning | Skips yarn/translation candidates that cannot intersect the current voxel chunk |
| Tetra/FEM examples | `script/tetgen_2d_weave_tetra.py`, `script/tet_fem_solve.py` | End-to-end mesh generation, C3D4 export, PNG preview, and scipy sparse FEM smoke solve |
| Local build helpers | `build.sh`, `build.bat`, `build.ps1` | Create/use a uv virtual environment, install build dependencies, compile, and install pytexgen |
| Verification tools | Backend tests and a synthetic benchmark script | Easier to check numpy, torch, adaptive, and pruning behavior after changes |

The goal is not to replace the TexGen C++ engine. The goal is to keep the
official modelling surface usable while moving fragile optional acceleration and
adaptive-mesh dependencies behind portable Python or opt-in build paths.

## Installation

```bash
pip install pytexgen
```

The base package depends only on numpy. Install extras when you want torch,
tqdm progress bars, or the example scripts:

```bash
pip install pytexgen              # TexGen bindings + numpy voxel backend
pip install "pytexgen[gpu]"       # add torch backend support
pip install "pytexgen[progress]"  # add tqdm progress bars
pip install "pytexgen[examples]"  # add scipy/matplotlib for example scripts
```

Pip automatically selects a compatible wheel for the current Python, OS, and
CPU. If no matching wheel exists, it falls back to the source distribution and
builds PyTexGen through scikit-build-core. CMake, Ninja, NumPy, and the Python
build dependencies are resolved through pip as needed; source fallback still
requires a system C++ compiler and Python development headers (`Python.h`).

For CUDA, install a torch wheel that matches your Python version, GPU driver,
and CUDA runtime first, then install pytexgen. The `gpu` extra intentionally does
not pin a CUDA wheel because PyTorch publishes different packages for different
CUDA runtimes.

Check the install:

```python
import pytexgen

print(pytexgen.__version__)
print(pytexgen.CTextile)
```

## Quick Start

Create and save a plain weave:

```python
from pytexgen import *

weave = CTextileWeave2D(4, 4, 5.0, 2.0, False)

weave.SwapPosition(0, 3)
weave.SwapPosition(1, 2)
weave.SwapPosition(2, 1)
weave.SwapPosition(3, 0)

weave.SetYarnWidths(4.0)
weave.SetYarnHeights(0.8)
weave.AssignDefaultDomain()

name = AddTextile(weave)
SaveToXML("plain_weave.tg3", name, OUTPUT_STANDARD)
DeleteTextile(name)
```

Generate a classic TexGen rectangular voxel mesh:

```python
from pytexgen import *

textile = CTextileWeave2D(2, 2, 1.0, 0.2, True)
textile.SwapPosition(0, 1)
textile.SwapPosition(1, 0)
textile.SetYarnWidths(0.8)
textile.SetYarnHeights(0.1)
textile.AssignDefaultDomain()

voxels = CRectangularVoxelMesh("CPeriodicBoundaries")
voxels.SaveVoxelMesh(
    textile,
    "mesh_cpp.inp",
    64, 64, 32,
    True,
    True,
    5,
    0,
)
```

Use the portable numpy/torch voxelizer instead:

```python
from pytexgen import *
from pytexgen.gpu_voxelizer import voxelize_textile

textile = CTextileWeave2D(2, 2, 1.0, 0.2, True)
textile.SwapPosition(0, 1)
textile.SwapPosition(1, 0)
textile.SetYarnWidths(0.8)
textile.SetYarnHeights(0.1)
textile.AssignDefaultDomain()

info = voxelize_textile(
    textile,
    nx=64, ny=64, nz=32,
    out_inp="mesh_numpy.inp",
    backend="numpy",
    workers=4,
    aabb_pruning=True,
)

print(info["backend"], len(info["yarn_id"]))
```

The default structured path is now the OpenMP-free numpy backend. Numpy
parallelism is chunk based: the default `chunk_voxels=8192` gives a `40x40x40`
grid eight classification tasks, and `info["workers"]` reports the effective
worker count after chunk-count clamping.
Set `progress=True` to show tqdm bars for classification and `.inp` writing;
the package imports tqdm lazily so normal installs do not require it.

Return structured voxel data directly in memory for another solver, with
optional `.npz` or directory-based `.npy` persistence:

```python
from pytexgen.gpu_voxelizer import VoxelGridData, voxelize_textile_data

data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="numpy",
    workers=4,
)

yarn_grid = data.grid              # shape: (nz, ny, nx)
material_grid = data.material_id() # matrix=0, yarns=1..N
flat_yarn_ids = data.yarn_id       # ix + iy*nx + iz*nx*ny order
data.save_npz("voxel_data.npz")
data.save_npy_dir("voxel_data_npy")

loaded = VoxelGridData.load_npz("voxel_data.npz")
mmap_loaded = VoxelGridData.load_npy_dir("voxel_data_npy", mmap_mode="r")
```

For reference-compatible production data, select the original TexGen
classifier without creating `.inp/.eld/.ori` files:

```python
exact = voxelize_textile_data(
    textile,
    nx=128, ny=128, nz=128,
    classification="exact",
    workers=8,                    # portable C++ std::thread workers
    include_orientations=True,
    orientation_storage="sparse",
)
```

`classification="exact"` calls `CTextile.GetPointInformation` on the same
z-layers and in the same element order as `CRectangularVoxelMesh`. It evaluates
geometry in C++ double precision, releases the Python GIL, and has no OpenMP
dependency. `backend="torch", device="cuda"` transfers this exact result to
CUDA for subsequent stiffness/material operations; classification itself stays
in the authoritative TexGen CPU core. Use `classification="tensor"` for the
faster approximate NumPy/Torch classifier.

For an OpenMP-free classifier whose hot loop is pure NumPy, extract an exact
array snapshot once and reuse it:

```python
from pytexgen.gpu_voxelizer import (
    extract_numpy_exact_geometry,
    voxelize_numpy_exact_geometry_data,
)

geometry = extract_numpy_exact_geometry(textile)  # one-time TexGen/SWIG step
data = voxelize_numpy_exact_geometry_data(
    geometry,
    nx=128, ny=128, nz=128,
    workers=4,
    chunk_voxels=32768,
    include_orientations=True,
    orientation_storage="sparse",
)
```

The equivalent one-call form is
`voxelize_textile_data(..., classification="numpy_exact")`. Classification
uses only NumPy and Python standard-library threads; SciPy is not required.
Bezier, cubic, and linear centre lines plus constant, node-interpolated
(including mid-node), or position-interpolated sections are reproduced.
Adjusted centre lines currently raise `NotImplementedError` instead of falling
back to an approximation.

Use `.npz` for a compact single-file artifact. Use `save_npy_dir(...)` when
large training runs should avoid zip decompression overhead or memory-map
individual arrays such as `yarn_id`, `orientation1`, and `orientation2`.

For anisotropic solvers, request per-voxel yarn directions:

```python
data = voxelize_textile_data(
    textile,
    nx=64, ny=64, nz=64,
    backend="numpy",
    output="numpy",
    include_orientations=True,
)

orientation1 = data.orientation1  # yarn tangent, shape: (nz, ny, nx, 3)
orientation2 = data.orientation2  # TexGen secondary material axis, shape: (nz, ny, nx, 3)
```

Build a sparse, fully GPU-resident material direction and stiffness field in
one call:

```python
from pytexgen.material_fields import (
    isotropic_stiffness_c21,
    orthotropic_stiffness_c21,
    save_material_field_bundle,
    voxelize_textile_material_fields,
)

matrix_c21 = isotropic_stiffness_c21(E=3.5e9, nu=0.35)
yarn_c21 = orthotropic_stiffness_c21(
    150e9, 10e9, 10e9, 0.25, 0.25, 0.30, 5e9, 5e9, 3.8e9
)
data, stiffness = voxelize_textile_material_fields(
    textile,
    nx=128, ny=128, nz=128,
    backend="torch", device="cuda", output="backend",
    classification="exact",
    matrix_stiffness=matrix_c21,
    default_yarn_stiffness=yarn_c21,
    yarn_stiffness_by_id={3: 1.1 * yarn_c21},
)

dense_c21 = stiffness.to_dense_c21()  # (Nz, Ny, Nx, 21)
acdm_c66 = stiffness.to_acdm()        # (1, 6, 6, Nz, Ny, Nx)
save_material_field_bundle("material_fields", data.sparse_orientation, stiffness)
```

Engineering-Voigt order is `(xx, yy, zz, yz, xz, xy)`. C21 stores the
row-major upper triangle:
`(C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66)`.
Matrix voxels share one `matrix_c21` and have no direction entry. Dense views
are created only when requested. Saving transfers CUDA tensors to CPU; keep
`output="backend"` during computation to avoid an earlier copy.

### Batch Voxelization

Keep `.tg3` as the editable model and prepare a memory-mappable `.ptgb` cache
for repeated production runs:

```python
from pytexgen.batch import MaterialSpec, prepare_geometry, voxelize_files_batch

prepare_geometry("models/plain.tg3", "prepared/plain.ptgb")

materials = MaterialSpec(
    matrix_c21=matrix_c21,
    default_yarn_c21=yarn_c21,
    unit="Pa",
)
report = voxelize_files_batch(
    ["prepared/plain.ptgb", "models/twill.tg3"],
    resolution=(128, 128, 128),
    output_dir="voxel_output",
    fields=("material_id", "orientation", "stiffness_c21"),
    materials=materials,
    device="cuda",
    dtype="float32",
    batch_size="auto",
    memory_budget_bytes=12 << 30,
)
```

Each input produces one directory containing `metadata.json` and uncompressed
`.npy` arrays. Shapes are `(Nz, Ny, Nx)`, `(Nz, Ny, Nx, 3, 3)`, and
`(Nz, Ny, Nx, 21)` respectively. PTGB v1 stores only flattened geometry needed
for voxelization; it is not a replacement for an editable TG3 model. GPU work
is streamed one geometry at a time while bounded background writers overlap
the result transfer and disk output.

For TexGen-reference output, pass only original `.tg3` inputs together with
`classification="exact"` or `classification="numpy_exact"`. PTGB v1 intentionally
stores lossy flattened geometry for the faster tensor classifier and therefore
cannot be used by either exact mode.

Use torch when an accelerator is available:

```python
from pytexgen.gpu_voxelizer import voxelize_textile

info = voxelize_textile(
    textile,
    nx=128, ny=128, nz=64,
    out_inp="mesh_torch.inp",
    backend="torch",
    device="cuda",  # also supports "mps" or "cpu"
)
```

Create a lightweight adaptive numpy mesh:

```python
from pytexgen.gpu_voxelizer import voxelize_textile

info = voxelize_textile(
    textile,
    nx=16, ny=16, nz=8,
    out_inp="mesh_adaptive_numpy.inp",
    backend="numpy",
    adaptive=True,
    adaptive_levels=2,
)
```

Adaptive numpy mode writes non-uniform Abaqus `C3D8R` cells. It does not produce
p4est-style 2:1 balancing or hanging-node constraint equations, so keep using a
p4est-enabled `COctreeVoxelMesh` build when a downstream FEM workflow requires
those guarantees.

## Backend Choices

| Path | Entry point | Dependencies | Best use |
|---|---|---|---|
| TexGen C++ structured voxels | `CRectangularVoxelMesh.SaveVoxelMesh(...)` | bundled TexGen core | Reference-compatible structured output |
| Exact direct data | `voxelize_textile_data(..., classification="exact")` | bundled TexGen core | Reference-compatible NumPy/Torch data without text files or OpenMP |
| NumPy exact data | `voxelize_textile_data(..., classification="numpy_exact")` | NumPy | Portable compatible classifier with reusable array geometry |
| Python numpy backend | `voxelize_textile(..., backend="numpy")` | `numpy` | Portable CPU voxelization without OpenMP |
| Python direct data backend | `voxelize_textile_data(...)` | `numpy`, optional `torch` | In-memory yarn/material grids and `.npz` handoff without `.inp` |
| Python torch backend | `voxelize_textile(..., backend="torch")` | `torch` | CUDA/MPS/torch CPU acceleration for larger grids |
| Python adaptive numpy backend | `voxelize_textile(..., adaptive=True)` | `numpy` | Lightweight non-uniform exploratory meshes |
| TexGen p4est octree | `COctreeVoxelMesh.SaveVoxelMesh(...)` | local p4est/sc build | Full p4est-style adaptive octree workflows |

See the source repository's `docs/voxel_backends.md` for backend limits, p4est
build notes, and benchmark commands.

## Core TexGen API

The package re-exports the SWIG-generated TexGen core API at the `pytexgen`
package level:

```python
from pytexgen import CTextile, CTextileWeave2D, CYarn, CNode, XYZ
from pytexgen import CSectionEllipse, CYarnSectionConstant
from pytexgen import CRectangularVoxelMesh, SaveToXML, ReadFromXML
```

Common API families:

| Family | Examples |
|---|---|
| Textiles | `CTextile`, `CTextileWeave2D`, `CShearedTextileWeave2D`, `CTextileWeave3D`, `CTextileOrthogonal`, `CTextileLayerToLayer` |
| Yarn geometry | `CYarn`, `CNode`, `XYZ`, `XY`, `CInterpolationCubic`, `CInterpolationBezier` |
| Sections | `CSectionEllipse`, `CSectionLenticular`, `CSectionRectangle`, `CSectionPolygon`, `CSectionPowerEllipse` |
| Domains | `CDomainPlanes`, `AssignDefaultDomain`, `GetDefaultDomain` |
| Mesh/export | `CRectangularVoxelMesh`, `CShearedVoxelMesh`, `CStaggeredVoxelMesh`, `CRotatedVoxelMesh`, `CTetgenMesh`, `CSurfaceMesh` |
| IO | `AddTextile`, `DeleteTextile`, `SaveToXML`, `ReadFromXML` |

## Compatibility With Upstream TexGen

This repository is based on the official
[TexGen](https://github.com/louisepb/TexGen) C++ codebase and keeps the main
Python modelling interface close to the upstream SWIG interface.

Intentional differences in the default pip/wheel build:

- `COctreeVoxelMesh` is not exposed by default because it depends on p4est/sc.
- The GUI, renderer, cascade export, examples, and documentation targets are not
  part of the core Python wheel.
- CPU voxel parallelism uses the C++11 standard thread library; architecture-native compiler flags remain opt-in.
- SWIG regeneration is opt-in; generated wrappers are committed for normal
  installs.

These defaults reduce fragile compile-time dependencies. If your project needs
the official p4est octree path, build locally with p4est/sc libraries and
`-DTEXGEN_REGENERATE_SWIG=ON`.

## Building From Source

Prerequisites:

- Python 3.9+
- Python development headers (`Python.h`)
- CMake 3.17+
- A C++11 compiler
- `scikit-build-core`

Install from a checkout:

```bash
git clone https://github.com/yufangjie1643/pytexgen.git
cd pytexgen
pip install -e .
```

Build a wheel:

```bash
pip install build
python -m build
```

Useful CMake options:

| Option | Default | Description |
|---|---|---|
| `BUILD_PYTHON_INTERFACE` | `ON` | Build Python bindings |
| `BUILD_RENDERER` | `OFF` | Build the OpenGL renderer |
| `BUILD_GUI` | `OFF` | Build the wxWidgets GUI |
| `BUILD_SHARED` | `OFF` | Build shared libraries instead of static wheel libraries |
| `TEXGEN_ENABLE_NATIVE_OPTIMIZATIONS` | `OFF` | Enable local CPU flags such as `-march=native` |
| `TEXGEN_REGENERATE_SWIG` | `OFF` | Regenerate `Core.py` and `Core_wrap.cxx` from `Python/Core.i` |
| `TEXGEN_INSTALL_WORKFLOW_EXTENSIONS` | `OFF` | Install source-only solver, training, and experimental mesher modules |

SWIG is only required when `TEXGEN_REGENERATE_SWIG=ON`.
The published wheel keeps this last option off so its Python API remains
focused on textile geometry, meshing, voxelization, and material fields.

## Testing And Benchmarks

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

## Project Layout

```text
Core/                    TexGen C++ geometry, textile, mesh, and export code
Python/Core.i            SWIG interface
Python/Core.py           committed SWIG Python proxy
Python/Core_wrap.cxx      committed SWIG C++ wrapper
TexGen/gpu_voxelizer.py   portable numpy/torch voxelization backend
TexGen/material_fields.py sparse orientation and stiffness fields
TexGen/batch.py           PTGB preparation and batch voxelization
src/pytexgen/             installed Python package
docs/voxel_backends.md    backend selection and p4est notes
pyproject.toml            Python packaging and wheel build configuration
```

## Attribution

TexGen was originally developed by Louise Brown and collaborators at the
University of Nottingham Composites Research Group. For academic use, please cite
the original TexGen project:

> Lin, H., Brown, L.P. and Long, A.C. (2011). Modelling and Simulating Textile
> Structures using TexGen. *Advanced Materials Research*, Vols. 331, pp 44-47.

## License

This project is licensed under the GNU General Public License v2.0 or later. See
the source repository `LICENSE` file for details.
