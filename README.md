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

The Python voxelizer enables conservative AABB candidate pruning with
`aabb_pruning=True` by default in the examples. The synthetic benchmark is the
quick way to compare pruning behavior after changes; increase `--yarn-grid` or
`--resolution` to make candidate skipping more visible.

Torch/CUDA benchmark when torch is installed:

```bash
python bench_gpu_voxelizer_backends.py --include-torch --device cuda
```

## SiC/SiC RVE Scripts

The `script/` directory contains ready-to-run examples for shallow-cross
layer-to-layer textiles and layer-aware RVE exports. The SiC/SiC example uses
default parameters, exports layer windows from the model domain instead of
hard-coded absolute z limits, and can read overrides from `@params.json`.
By default, RVE z windows are cut between neighbouring flat yarn centre planes
instead of equally splitting the padded domain box. Each RVE can save the
matching clipped `.tg3` model and Abaqus `.inp` mesh with the same layer label.

Generate the default SiC/SiC shallow-cross straight model and all default RVE
layers:

```bash
uv run python script/sic_sic_shallow_cross_straight.py
```

Override parameters with a JSON file:

```bash
uv run python script/sic_sic_shallow_cross_straight.py @params.json
```

Minimal `params.json` example for one `128x128x128` RVE and its matching TG3:

```json
{
  "rve_export": {
    "enabled": true,
    "save_dir": "Saved_SiC_SiC_Shallow_Cross_Straight/RVE",
    "file_prefix": "sic_sic_shallow_cross_straight_rve",
    "layer_count": 5,
    "layers": "center",
    "layers_per_rve": 1,
    "window_mode": "yarn_centres",
    "save_tg3": true,
    "mesh_boundary": "CPeriodicBoundaries",
    "mesh_resolutions": [[128, 128, 128]]
  }
}
```

The SiC/SiC generator enforces the yarn-count rules used by the examples:
`z_layers >= 1`, `num_x_yarns` is the y-direction yarn count and must be a
multiple of 2, and `num_y_yarns` is the x-direction yarn count and must be a
multiple of 4. `rve_export.layer_count` must match `z_layers`.

An alternate x4/y2/z1 layout is committed as
`script/params_sic_sic_x4_y2_z1_rve.json`. The parent model keeps 5 z layers,
then the RVE export selects one central z window with `"layers": "center"`.
For 5 layers this exports `L02` at `64x32x16`, saves the matching clipped TG3
file, and writes output under
`Saved_SiC_SiC_Shallow_Cross_Straight/RVE_x4_y2_z1_center_test`.

```bash
uv run python script/sic_sic_shallow_cross_straight.py @script/params_sic_sic_x4_y2_z1_rve.json
```

Preview the x4/y2/z1 central `L02` output:

```bash
pip install plotly
uv run python script/inp_viewer.py Saved_SiC_SiC_Shallow_Cross_Straight/RVE_x4_y2_z1_center_test/sic_sic_x4_y2_z1_center_test_rve_L02_mesh_64x32x16.inp --backend plotly --output build/rve_x4_y2_z1_L02.html --background white
```

Render yarn elements from the latest generated INP file. This defaults to
auto-detecting the newest `.inp` under the common output folders:

```bash
pip install matplotlib
uv run python script/inp_viewer.py --backend matplotlib --output build/rve_view.png --background white --no-axes --no-title
```

Create an interactive HTML view:

```bash
pip install plotly
uv run python script/inp_viewer.py path/to/mesh.inp --backend plotly --output build/rve_view.html --background white
```

Run the reproducible RVE export benchmark used for local speed checks. The
default is 64 exports, 4 parallel workers, and a `64x64x64` mesh. Large
generated mesh files are deleted after each case; `progress.json` and
`summary.json` remain under `build/`.

```bash
uv run python script/bench_sic_sic_rve_parallel.py
```

Useful benchmark variants:

```bash
uv run python script/bench_sic_sic_rve_parallel.py --cases 1 --resolution 128 128 128
uv run python script/bench_sic_sic_rve_parallel.py --cases 8 --workers 2 --resolution 64
uv run python script/bench_sic_sic_rve_parallel.py --keep-output
```

On the Windows workstation used during development, the default `64x64x64`
benchmark completed 64/64 cases in 62.599 s with 4 workers. The temporary mesh
data totalled about 2.9 GB before cleanup.

## Weft-Knit Composite Script

`script/weft_knit_composite.py` creates a parametric weft-knit composite parent
model using `CTextileWeftKnit`, then exports a central RVE window. The default
parent is 8 wales by 8 courses; the RVE is the central 2-by-2 window. This keeps
the parent model larger than the rendered/exported region.

Run the default model and RVE export:

```bash
uv run python script/weft_knit_composite.py
```

Run with the committed parameter file:

```bash
uv run python script/weft_knit_composite.py @script/params_weft_knit_composite_rve.json
```

The default parameter file writes:

```text
Saved_Weft_Knit_Composite/weft_knit_composite.tg3
Saved_Weft_Knit_Composite/RVE/weft_knit_composite_rve_W03_W04_C03_C04.tg3
Saved_Weft_Knit_Composite/RVE/weft_knit_composite_rve_W03_W04_C03_C04_mesh_64x64x32.inp
```

Preview the central knit RVE:

```bash
pip install plotly
uv run python script/inp_viewer.py Saved_Weft_Knit_Composite/RVE/weft_knit_composite_rve_W03_W04_C03_C04_mesh_64x64x32.inp --backend plotly --output build/weft_knit_rve.html --background white
```

## Project Layout

```text
Core/                    TexGen C++ geometry, textile, mesh, and export code
Python/Core.i            SWIG interface
Python/Core.py           committed SWIG Python proxy
Python/Core_wrap.cxx      committed SWIG C++ wrapper
TexGen/gpu_voxelizer.py   portable numpy/torch voxelization backend
src/pytexgen/             installed Python package
script/                   TexGen examples, RVE export helpers, INP viewer, benchmarks
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
