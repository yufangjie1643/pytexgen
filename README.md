# pytexgen

[![PyPI version](https://img.shields.io/pypi/v/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![Python](https://img.shields.io/pypi/pyversions/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

**Python bindings for [TexGen](https://github.com/louisepb/TexGen)** — an open-source geometric textile modelling software package developed at the University of Nottingham for obtaining engineering properties of woven textiles and textile composites.

pytexgen brings the full power of the TexGen C++ engine to Python, enabling scripted creation, analysis, and export of textile geometries for computational mechanics workflows.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Create a 2D Plain Weave](#1-create-a-2d-plain-weave)
  - [2. Custom Yarn Paths](#2-custom-yarn-paths)
  - [3. Sheared Weave with Voxel Mesh Export](#3-sheared-weave-with-voxel-mesh-export)
  - [4. 3D Woven Textile](#4-3d-woven-textile)
  - [5. Layered Textiles](#5-layered-textiles)
- [API Overview](#api-overview)
  - [Global Functions](#global-functions)
  - [Textile Classes](#textile-classes)
  - [Yarn and Node](#yarn-and-node)
  - [Cross-Sections](#cross-sections)
  - [Yarn Section Assignment](#yarn-section-assignment)
  - [Interpolation](#interpolation)
  - [Domains](#domains)
  - [Mesh Generation](#mesh-generation)
  - [Math Utilities](#math-utilities)
- [Export Formats](#export-formats)
- [Building from Source](#building-from-source)
- [Project Structure](#project-structure)
- [Upstream Project](#upstream-project)
- [License](#license)

---

## Features

- **Comprehensive textile modelling** — 2D weaves, 3D weaves, angle interlocks, layer-to-layer, orthogonal weaves, knits, and braids.
- **Sheared geometries** — model fabrics under in-plane shear deformation with `CShearedTextileWeave2D`.
- **Flexible yarn definition** — full control over yarn paths (nodes), interpolation (cubic, linear, Bezier), and cross-sections (ellipse, lenticular, rectangle, polygon, power ellipse, and more).
- **Mesh generation** — rectangular voxel, sheared voxel, staggered voxel, rotated voxel, octree voxel, tetrahedral (`CTetgenMesh`), and surface shell meshes.
- **FEA export** — direct export to Abaqus (`.inp`) and ANSYS formats with periodic boundary conditions.
- **XML serialization** — save and reload complete textile models as `.tg3` / `.tgx` files.
- **Geometric analysis** — fibre volume fraction, yarn path queries, interference detection, and domain clipping.
- **Cross-platform** — pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).

---

## Installation

```bash
pip install pytexgen
```

Requires **Python 3.9+**. Pre-built binary wheels are provided for all major platforms — no compiler or C++ dependencies needed.

To verify the installation:

```python
import pytexgen
print(pytexgen.__version__)
```

The OpenMP-free Python voxelizer is optional:

```bash
pip install "pytexgen[voxel]"  # numpy CPU/adaptive backend
pip install "pytexgen[gpu]"    # numpy + torch backend
```

---

## Quick Start

All examples follow the same pattern: **import → build textile → set sections → assign domain → export**.

### 1. Create a 2D Plain Weave

```python
from pytexgen import *

# Create a 4×4 2D weave (4 warp yarns, 4 weft yarns)
# Parameters: num_x, num_y, spacing, thickness, refine
weave = CTextileWeave2D(4, 4, 5.0, 2.0, False)

# Define the interlacing pattern (plain weave)
weave.SwapPosition(0, 3)
weave.SwapPosition(1, 2)
weave.SwapPosition(2, 1)
weave.SwapPosition(3, 0)

# Set yarn dimensions
weave.SetYarnWidths(4.0)
weave.SetYarnHeights(0.8)

# Assign default rectangular domain
weave.AssignDefaultDomain()

# Register and export
name = AddTextile(weave)
SaveToXML("plain_weave.tg3", name, OUTPUT_STANDARD)

# Clean up
DeleteTextile(name)
```

### 2. Custom Yarn Paths

For full control over yarn geometry, build textiles from individual yarns:

```python
from pytexgen import *

textile = CTextile()

# --- Warp yarn with cubic interpolation and elliptical cross-section ---
yarn = CYarn()
yarn.AddNode(CNode(XYZ(0, 0, 0)))
yarn.AddNode(CNode(XYZ(5, 0, 2)))
yarn.AddNode(CNode(XYZ(10, 0, 0)))
yarn.AssignInterpolation(CInterpolationCubic())
yarn.AssignSection(CYarnSectionConstant(CSectionEllipse(2.0, 1.0)))
yarn.SetResolution(20)
yarn.AddRepeat(XYZ(10, 0, 0))
yarn.AddRepeat(XYZ(0, 10, 0))
textile.AddYarn(yarn)

# --- Weft yarn with varying cross-section along its path ---
yarn2 = CYarn()
yarn2.AddNode(CNode(XYZ(0, 0, 2)))
yarn2.AddNode(CNode(XYZ(0, 5, 0)))
yarn2.AddNode(CNode(XYZ(0, 10, 2)))
yarn2.AssignInterpolation(CInterpolationCubic())

# Interpolate between different sections at each node
section = CYarnSectionInterpNode()
section.AddSection(CSectionLenticular(2.0, 0.5))
section.AddSection(CSectionEllipse(2.0, 1.0))
section.AddSection(CSectionLenticular(0.5, 2.0))
yarn2.AssignSection(section)
yarn2.SetResolution(20)
yarn2.AddRepeat(XYZ(10, 0, 0))
yarn2.AddRepeat(XYZ(0, 10, 0))
textile.AddYarn(yarn2)

# Assign domain and register
textile.AssignDomain(CDomainPlanes(XYZ(0, 0, -1), XYZ(10, 10, 3)))
AddTextile("custom", textile)
```

### 3. Sheared Weave with Voxel Mesh Export

Model fabrics under in-plane shear and generate finite element meshes:

```python
import math
from pathlib import Path
from pytexgen import *

# Create a 2×2 sheared weave (15° shear angle)
shear_angle = math.radians(15)
textile = CShearedTextileWeave2D(
    2, 2,           # num_x, num_y
    1.0,            # spacing
    0.2,            # thickness
    shear_angle,    # shear angle in radians
    True,           # refine
    True            # in-plane tangents
)

# Plain weave pattern
textile.SwapPosition(1, 0)
textile.SwapPosition(0, 1)

# Yarn geometry
textile.SetYarnWidths(0.8)
textile.SetYarnHeights(0.1)
textile.AssignDefaultDomain()

name = AddTextile(textile)

# Export to XML
Path("output").mkdir(exist_ok=True)
SaveToXML("output/sheared_weave.tg3", name, OUTPUT_STANDARD)

# Generate rectangular voxel mesh with periodic boundaries
voxel = CRectangularVoxelMesh("CPeriodicBoundaries")
voxel.SaveVoxelMesh(
    textile,          # textile object
    "output/sheared_weave_48.inp",
    48, 48, 24,       # voxel resolution in x, y, z
    True, True,       # output options
    5,                # file format (Abaqus = 5)
    0                 # offset
)

DeleteTextile(name)
```

### 4. 3D Woven Textile

```python
from pytexgen import *

# 8 warp × 4 weft, spacing=5, thickness=7
textile = CTextileWeave3D(8, 4, 5.0, 7.0)

# Add yarn layers
textile.AddYLayers(0, 1)
textile.AddYLayers(2, 1)
textile.AddYLayers(4, 1)
textile.AddYLayers(6, 1)
textile.AddXLayers()
textile.AddYLayers()
textile.AddXLayers()
textile.AddYLayers()
textile.AddXLayers()
textile.AddYLayers()

# Define 3D interlacing with PushUp / PushDown
textile.PushUp(0, 0)
textile.PushUp(1, 0)
textile.PushDown(4, 0)
textile.PushUp(7, 0)

textile.PushUp(1, 1)
textile.PushUp(2, 1)
textile.PushUp(3, 1)
textile.PushDown(6, 1)

textile.PushDown(0, 2)
textile.PushUp(3, 2)
textile.PushUp(4, 2)
textile.PushUp(5, 2)

textile.PushDown(2, 3)
textile.PushUp(5, 3)
textile.PushUp(6, 3)
textile.PushUp(7, 3)

textile.SetYarnWidths(4.0)
textile.SetYarnHeights(1.0)
textile.AssignDefaultDomain()
AddTextile(textile)
```

### 5. Layered Textiles

Combine multiple textile layers into a single model:

```python
from pytexgen import *

# Layer 1: Satin weave
layer1 = CTextileWeave2D(4, 4, 1.0, 0.2, True, False)
layer1.SetGapSize(0)
layer1.SwapPosition(0, 3)
layer1.SwapPosition(1, 2)
layer1.SwapPosition(2, 1)
layer1.SwapPosition(3, 0)
layer1.SetYarnWidths(0.8)
layer1.SetYarnHeights(0.1)
layer1.AssignDefaultDomain()

# Get domain bounds for stacking
domain = layer1.GetDefaultDomain()
min_pt, max_pt = XYZ(), XYZ()
domain.GetBoxLimits(min_pt, max_pt)

# Layer 2: Plain weave
layer2 = CTextileWeave2D(2, 2, 1.0, 0.25, True, False)
layer2.SetGapSize(0)
layer2.SwapPosition(0, 1)
layer2.SwapPosition(1, 0)
layer2.SetYarnWidths(0.8)

# Combine layers with vertical offset
combined = CTextile()
for yarn in layer1.GetYarns():
    combined.AddYarn(yarn)

offset = XYZ(0, 0, 0.2)
for yarn in layer2.GetYarns():
    nodes = yarn.GetMasterNodes()
    for i, node in enumerate(nodes):
        node.SetPosition(node.GetPosition() - offset)
        yarn.ReplaceNode(i, node)
    combined.AddYarn(yarn)

min_pt -= offset
combined.AssignDomain(CDomainPlanes(min_pt, max_pt))
AddTextile("layered", combined)
```

---

## API Overview

### Global Functions

| Function | Description |
|---|---|
| `AddTextile(textile)` | Register a textile, returns auto-generated name |
| `AddTextile(name, textile)` | Register a textile with a given name |
| `GetTextile(name)` | Retrieve a registered textile by name |
| `DeleteTextile(name)` | Remove a textile and free memory |
| `DeleteTextiles()` | Remove all registered textiles |
| `GetTextiles()` | Get dict of all registered textiles |
| `SaveToXML(filename, name, output_type)` | Save textile to XML (`.tg3` / `.tgx`) |
| `ReadFromXML(filename)` | Load textiles from an XML file |

`output_type` can be `OUTPUT_MINIMAL`, `OUTPUT_STANDARD`, or `OUTPUT_FULL`.

### Textile Classes

| Class | Description |
|---|---|
| `CTextile` | Base class — build from individual yarns |
| `CTextileWeave2D` | 2D woven textile with interlacing pattern |
| `CShearedTextileWeave2D` | 2D weave under in-plane shear |
| `CTextileWeave3D` | 3D woven textile (multi-layer) |
| `CTextile3DWeave` | Alternative 3D weave definition |
| `CTextileOrthogonal` | Orthogonal 3D weave |
| `CTextileAngleInterlock` | Angle interlock weave |
| `CTextileLayerToLayer` | Layer-to-layer interlock weave |
| `CTextileDecoupledLToL` | Decoupled layer-to-layer |
| `CTextileOffsetAngleInterlock` | Offset angle interlock |
| `CTextileLayered` | Combine multiple textiles into layers |

Common weave methods: `SwapPosition(x, y)`, `SetYarnWidths(w)`, `SetYarnHeights(h)`, `SetXYarnWidths(idx, w)`, `SetYYarnWidths(idx, w)`, `SetGapSize(g)`, `AssignDefaultDomain()`.

### Yarn and Node

```python
yarn = CYarn()
yarn.AddNode(CNode(XYZ(x, y, z)))       # Add control point
yarn.AssignInterpolation(interp)          # Set path interpolation
yarn.AssignSection(section)               # Set cross-section
yarn.SetResolution(n)                     # Surface discretization
yarn.AddRepeat(XYZ(dx, dy, dz))          # Periodic repeat vector
yarn.ReplaceNode(index, node)             # Modify an existing node
yarn.GetMasterNodes()                     # Retrieve node list
```

### Cross-Sections

| Class | Parameters | Shape |
|---|---|---|
| `CSectionEllipse(w, h)` | width, height | Elliptical |
| `CSectionLenticular(w, h)` | width, height | Lens-shaped |
| `CSectionRectangle(w, h)` | width, height | Rectangular |
| `CSectionPolygon(points)` | list of XY points | Arbitrary polygon |
| `CSectionBezier(points)` | control points | Bezier curve |
| `CSectionPowerEllipse(w, h, p)` | width, height, power | Superellipse |
| `CSectionHybrid()` | composite | Multiple section types |
| `CSectionRotated(section, angle)` | base section, angle | Rotated variant |
| `CSectionScaled(section, sx, sy)` | base section, scale | Scaled variant |

### Yarn Section Assignment

| Class | Usage |
|---|---|
| `CYarnSectionConstant(section)` | Same cross-section everywhere |
| `CYarnSectionInterpNode()` | Vary section at each node (call `.AddSection()`) |
| `CYarnSectionInterpPosition()` | Vary section by parametric position |
| `CYarnSectionAdjusted()` | Fine-tune sections with per-node offsets |

### Interpolation

| Class | Description |
|---|---|
| `CInterpolationCubic()` | Natural cubic spline (recommended) |
| `CInterpolationLinear()` | Linear between nodes |
| `CInterpolationBezier()` | Bezier curve interpolation |
| `CInterpolationAdjusted()` | Manual adjustments to paths |

### Domains

```python
# Rectangular box domain
domain = CDomainPlanes(XYZ(x_min, y_min, z_min), XYZ(x_max, y_max, z_max))
textile.AssignDomain(domain)

# Or let the textile compute it automatically
textile.AssignDefaultDomain()

# Query domain bounds
min_pt, max_pt = XYZ(), XYZ()
domain.GetBoxLimits(min_pt, max_pt)
```

### Mesh Generation

| Class | Description |
|---|---|
| `CRectangularVoxelMesh(bc_type)` | Axis-aligned voxel mesh |
| `CShearedVoxelMesh(bc_type)` | Voxel mesh for sheared textiles |
| `CStaggeredVoxelMesh(bc_type)` | Staggered offset voxel mesh |
| `CRotatedVoxelMesh(bc_type)` | Rotated coordinate voxel mesh |
| `COctreeVoxelMesh(bc_type)` | Adaptive octree refinement |
| `CTetgenMesh(min_size)` | Tetrahedral mesh via TetGen |
| `CSurfaceMesh()` | Surface triangulation |
| `CShellElementExport(...)` | Shell element export |

`COctreeVoxelMesh` requires p4est/sc libraries and is not included in the default binary wheels.
For portable installs, `pytexgen.gpu_voxelizer.voxelize_textile(..., adaptive=True)`
provides lightweight numpy adaptive voxel output without p4est.

Boundary condition types: `"CPeriodicBoundaries"`, `"CMaterialPeriodic"`, etc.

Voxel mesh export example:

```python
vox = CRectangularVoxelMesh("CPeriodicBoundaries")
vox.SaveVoxelMesh(
    textile,          # textile object
    "mesh.inp",       # output file path
    nx, ny, nz,       # voxel resolution in x, y, z
    True,             # bOutputMatrix
    True,             # bOutputYarns
    5,                # file format (Abaqus = 5)
    0                 # offset
)
```

OpenMP-free Python backend alternative:

```python
from pytexgen.gpu_voxelizer import voxelize_textile

voxelize_textile(
    textile,
    nx=nx, ny=ny, nz=nz,
    out_inp="mesh_numpy.inp",
    backend="numpy",  # or "torch" for CUDA/MPS/torch CPU
    workers=4,        # numpy backend only
    aabb_pruning=True,
)

voxelize_textile(
    textile,
    nx=16, ny=16, nz=8,
    out_inp="mesh_adaptive_numpy.inp",
    backend="numpy",
    adaptive=True,
    adaptive_levels=2,
)
```

`adaptive=True` refines cells whose center/corners disagree on yarn ownership and
writes non-uniform C3D8R cells. It does not generate p4est-style 2:1 balancing or
hanging-node constraints; use a local p4est-enabled `COctreeVoxelMesh` build when
those conforming adaptive FEM features are required.
`aabb_pruning=True` skips yarn/translation candidates whose conservative bounding
boxes cannot overlap the current voxel chunk. For backend development, run
`python bench_gpu_voxelizer_backends.py` to compare pruned and unpruned numpy
classification on a synthetic workload.

See [docs/voxel_backends.md](docs/voxel_backends.md) for backend selection,
adaptive-mode limits, torch/CUDA notes, and the advanced p4est build path.

### Math Utilities

```python
# 3D point / vector
v = XYZ(1.0, 2.0, 3.0)

# 2D point
p = XY(1.0, 2.0)

# Quaternion rotation
q = WXYZ(w, x, y, z)

# Transformation matrix
m = CMatrix()
t = CLinearTransformation()
```

---

## Export Formats

| Format | Method | Use Case |
|---|---|---|
| TexGen XML (`.tg3`) | `SaveToXML()` | Save/reload textile models |
| Abaqus (`.inp`) | Voxel/Tet mesh `.SaveVoxelMesh()` / `.SaveTetgenMesh()` | Finite element analysis |
| ANSYS | `pytexgen.Ansys` module | ANSYS FEA |
| Surface mesh | `CSurfaceMesh` | Visualization, CFD |
| WiseTex | `pytexgen.WiseTex` module | Permeability analysis |
| FlowTex | `pytexgen.FlowTex` module | Flow simulation |

---

## Building from Source

If you need to build pytexgen from source (e.g., for development or unsupported platforms):

### Prerequisites

- **CMake** 3.17+
- **C++ compiler** with C++11 support (GCC, Clang, or MSVC)
- **Python** 3.9+ with `scikit-build-core`

SWIG is not required for normal wheel builds because generated bindings are
committed in the repository. Install SWIG only when changing `Python/Core.i`
and configure with `-DTEXGEN_REGENERATE_SWIG=ON`.

### Build Steps

```bash
git clone https://github.com/yufangjie1643/pytexgen.git
cd pytexgen

# Install build dependencies
pip install scikit-build-core

# Build and install in development mode
pip install -e .

# Or build a wheel
pip install build
python -m build
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_PYTHON_INTERFACE` | ON | Build Python bindings |
| `BUILD_RENDERER` | OFF | Build OpenGL renderer |
| `BUILD_GUI` | OFF | Build wxWidgets GUI |
| `BUILD_SHARED` | OFF | Shared libs (OFF for wheels) |
| `TEXGEN_ENABLE_OPENMP` | OFF | Enable optional C++ OpenMP point-classification loops |
| `TEXGEN_ENABLE_NATIVE_OPTIMIZATIONS` | OFF | Enable local CPU-specific flags such as `-march=native` |
| `TEXGEN_REGENERATE_SWIG` | OFF | Regenerate Python bindings from `Python/Core.i` |

`COctreeVoxelMesh` is disabled in default wheel/SKBUILD builds. To expose it,
provide local p4est/sc libraries and configure a legacy CMake build with
`-DTEXGEN_REGENERATE_SWIG=ON`; see [docs/voxel_backends.md](docs/voxel_backends.md).

---

## Project Structure

```
TexGen/
├── Core/                   # C++ engine (textile geometry, meshing, export)
├── Python/
│   ├── Core.i              # SWIG interface — C++ to Python bindings
│   ├── Core.py             # Committed SWIG-generated Python proxy
│   ├── Core_wrap.cxx       # Committed SWIG-generated C++ wrapper
│   ├── Scripts/            # Example scripts
│   └── Lib/                # Python utility modules (Abaqus, Ansys, ...)
├── src/pytexgen/           # Python package (installed via pip)
│   ├── __init__.py         # Package entry point
│   └── Core (compiled)     # SWIG-generated binary module
├── Triangle/               # Constrained Delaunay triangulation library
├── tetgenlib/              # Tetrahedral mesh generator
├── OctreeRefinement/       # Adaptive octree mesh refinement
├── CSparse/                # Sparse matrix library
├── tinyxml/                # XML parser
├── CMakeLists.txt          # Build system
├── pyproject.toml          # Python packaging metadata
└── LICENSE                 # GPL-2.0
```

---

## Upstream Project

pytexgen is the Python packaging of [**TexGen**](https://github.com/louisepb/TexGen), originally developed by **Louise Brown** and collaborators at the **University of Nottingham**, Composites Research Group.

TexGen is a geometric textile modelling software package used for obtaining engineering properties of woven textiles and textile composites. The original project provides both a GUI application and a C++ library; pytexgen makes the core engine accessible as a standard Python package via `pip install`.

For academic use, please cite the original TexGen project:

> Lin, H., Brown, L.P. and Long, A.C. (2011). Modelling and Simulating Textile Structures using TexGen. *Advanced Materials Research*, Vols. 331, pp 44-47.

---

## License

This project is licensed under the **GNU General Public License v2.0 or later** — see the [LICENSE](LICENSE) file for details.

Copyright (C) University of Nottingham. TexGen Contributors.
