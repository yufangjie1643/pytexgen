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

## What This Project Adds

| Area | Contribution | Practical impact |
|---|---|---|
| Python packaging | `pyproject.toml` + `scikit-build-core` build path | Users can install with normal `pip` workflows instead of hand-driving CMake/SWIG |
| Stable wheel builds | Pre-generated `Python/Core.py` and `Python/Core_wrap.cxx` | Normal builds do not require a local SWIG install |
| Cross-platform defaults | GUI, renderer, OpenMP, native CPU flags, and p4est are off by default | Fewer Windows/MSVC/MinGW, OpenMP runtime, and older-CPU build failures |
| Python voxel backend | `pytexgen.gpu_voxelizer.voxelize_textile(...)` | OpenMP-free structured voxel output through numpy or torch |
| GPU-ready path | Optional `backend="torch"` with CUDA/MPS/CPU devices | Larger voxel grids can use torch acceleration without changing the TexGen C++ core |
| Lightweight adaptive output | `adaptive=True` numpy mode | Exploratory non-uniform C3D8R voxel meshes without compiling p4est |
| Performance pruning | Conservative AABB candidate pruning | Skips yarn/translation candidates that cannot intersect the current voxel chunk |
| Verification tools | Backend tests and a synthetic benchmark script | Easier to check numpy, torch, adaptive, and pruning behavior after changes |

The goal is not to replace the TexGen C++ engine. The goal is to keep the
official modelling surface usable while moving fragile optional acceleration and
adaptive-mesh dependencies behind portable Python or opt-in build paths.

## Installation

```bash
pip install pytexgen
```

The base package depends only on numpy. Install the GPU extra when you want the
torch backend:

```bash
pip install pytexgen         # TexGen bindings + numpy voxel backend
pip install "pytexgen[gpu]"  # add torch backend support
```

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
| Python numpy backend | `voxelize_textile(..., backend="numpy")` | `numpy` | Portable CPU voxelization without OpenMP |
| Python torch backend | `voxelize_textile(..., backend="torch")` | `torch` | CUDA/MPS/torch CPU acceleration for larger grids |
| Python adaptive numpy backend | `voxelize_textile(..., adaptive=True)` | `numpy` | Lightweight non-uniform exploratory meshes |
| TexGen p4est octree | `COctreeVoxelMesh.SaveVoxelMesh(...)` | local p4est/sc build | Full p4est-style adaptive octree workflows |

See [docs/voxel_backends.md](docs/voxel_backends.md) for backend limits, p4est
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
- OpenMP and architecture-native compiler flags are opt-in rather than default.
- SWIG regeneration is opt-in; generated wrappers are committed for normal
  installs.

These defaults reduce fragile compile-time dependencies. If your project needs
the official p4est octree path, build locally with p4est/sc libraries and
`-DTEXGEN_REGENERATE_SWIG=ON`.

## Building From Source

Prerequisites:

- Python 3.9+
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
| `TEXGEN_ENABLE_OPENMP` | `OFF` | Enable optional C++ OpenMP loops |
| `TEXGEN_ENABLE_NATIVE_OPTIMIZATIONS` | `OFF` | Enable local CPU flags such as `-march=native` |
| `TEXGEN_REGENERATE_SWIG` | `OFF` | Regenerate `Core.py` and `Core_wrap.cxx` from `Python/Core.i` |

SWIG is only required when `TEXGEN_REGENERATE_SWIG=ON`.

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
[LICENSE](LICENSE) for details.
