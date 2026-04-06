# pytexgen

[![PyPI version](https://img.shields.io/pypi/v/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![Python](https://img.shields.io/pypi/pyversions/pytexgen.svg)](https://pypi.org/project/pytexgen/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

**Python bindings for [TexGen](https://github.com/louisepb/TexGen)** — an open-source geometric textile modelling software package developed at the University of Nottingham for obtaining engineering properties of woven textiles and textile composites.

pytexgen brings the full power of the TexGen C++ engine to Python, enabling scripted creation, analysis, and export of textile geometries for computational mechanics workflows.

## Features

- **Comprehensive textile modelling** — 2D weaves, 3D weaves, angle interlocks, layer-to-layer, orthogonal weaves, knits, and braids
- **Sheared geometries** — model fabrics under in-plane shear deformation with `CShearedTextileWeave2D`
- **Flexible yarn definition** — full control over yarn paths, interpolation (cubic, linear, Bezier), and cross-sections (ellipse, lenticular, rectangle, polygon, power ellipse, and more)
- **Mesh generation** — rectangular voxel, sheared voxel, staggered voxel, rotated voxel, octree voxel, tetrahedral (`CTetgenMesh`), and surface shell meshes
- **FEA export** — direct export to Abaqus (`.inp`) and ANSYS formats with periodic boundary conditions
- **XML serialization** — save and reload complete textile models as `.tg3` / `.tgx` files
- **Geometric analysis** — fibre volume fraction, yarn path queries, interference detection, and domain clipping
- **Cross-platform** — pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64)

## Installation

```bash
pip install pytexgen
```

Requires **Python 3.8+**. Pre-built binary wheels are provided for all major platforms — no compiler or C++ dependencies needed.

## Quick Start

### 2D Plain Weave

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
```

### Sheared Weave with Voxel Mesh Export

```python
import math
from pytexgen import *

shear_angle = math.radians(15)
textile = CShearedTextileWeave2D(2, 2, 1.0, 0.2, shear_angle, True, True)
textile.SwapPosition(1, 0)
textile.SwapPosition(0, 1)
textile.SetYarnWidths(0.8)
textile.SetYarnHeights(0.1)
textile.AssignDefaultDomain()

name = AddTextile(textile)

# Generate voxel mesh with periodic boundaries for Abaqus
voxel = CRectangularVoxelMesh("CPeriodicBoundaries")
voxel.SaveVoxelMesh(textile, "sheared_weave.inp", 48, 48, 24, True, True, 5, 0)
```

### Custom Yarn Paths

```python
from pytexgen import *

textile = CTextile()

yarn = CYarn()
yarn.AddNode(CNode(XYZ(0, 0, 0)))
yarn.AddNode(CNode(XYZ(5, 0, 2)))
yarn.AddNode(CNode(XYZ(10, 0, 0)))
yarn.AssignInterpolation(CInterpolationCubic())
yarn.AssignSection(CYarnSectionConstant(CSectionEllipse(2.0, 1.0)))
yarn.SetResolution(20)
yarn.AddRepeat(XYZ(10, 0, 0))
textile.AddYarn(yarn)

textile.AssignDomain(CDomainPlanes(XYZ(0, 0, -1), XYZ(10, 10, 3)))
AddTextile("custom", textile)
```

## API Overview

### Textile Classes

| Class | Description |
|---|---|
| `CTextile` | Base class — build from individual yarns |
| `CTextileWeave2D` | 2D woven textile with interlacing pattern |
| `CShearedTextileWeave2D` | 2D weave under in-plane shear |
| `CTextileWeave3D` | 3D woven textile (multi-layer) |
| `CTextileOrthogonal` | Orthogonal 3D weave |
| `CTextileAngleInterlock` | Angle interlock weave |
| `CTextileLayerToLayer` | Layer-to-layer interlock weave |
| `CTextileLayered` | Combine multiple textiles into layers |

### Cross-Sections

| Class | Shape |
|---|---|
| `CSectionEllipse(w, h)` | Elliptical |
| `CSectionLenticular(w, h)` | Lens-shaped |
| `CSectionRectangle(w, h)` | Rectangular |
| `CSectionPolygon(points)` | Arbitrary polygon |
| `CSectionPowerEllipse(w, h, p)` | Superellipse |
| `CSectionRotated(section, angle)` | Rotated variant |

### Mesh Generation

| Class | Description |
|---|---|
| `CRectangularVoxelMesh(bc)` | Axis-aligned voxel mesh |
| `CShearedVoxelMesh(bc)` | Voxel mesh for sheared textiles |
| `CStaggeredVoxelMesh(bc)` | Staggered offset voxel mesh |
| `CRotatedVoxelMesh(bc)` | Rotated coordinate voxel mesh |
| `COctreeVoxelMesh(bc)` | Adaptive octree refinement |
| `CTetgenMesh(min_size)` | Tetrahedral mesh via TetGen |
| `CSurfaceMesh()` | Surface triangulation |

### Export Formats

| Format | Method | Use Case |
|---|---|---|
| TexGen XML (`.tg3`) | `SaveToXML()` | Save/reload textile models |
| Abaqus (`.inp`) | `SaveVoxelMesh()` / `SaveTetgenMesh()` | Finite element analysis |
| ANSYS | `pytexgen.Ansys` module | ANSYS FEA |
| WiseTex | `pytexgen.WiseTex` module | Permeability analysis |
| FlowTex | `pytexgen.FlowTex` module | Flow simulation |

## Building from Source

```bash
git clone https://github.com/yufangjie1643/pytexgen.git
cd pytexgen
pip install scikit-build-core
pip install -e .
```

Requires CMake 3.17+, a C++ compiler, and SWIG.

## Upstream Project

pytexgen is the Python packaging of [TexGen](https://github.com/louisepb/TexGen), originally developed by Louise Brown and collaborators at the University of Nottingham, Composites Research Group.

For academic use, please cite:

> Lin, H., Brown, L.P. and Long, A.C. (2011). Modelling and Simulating Textile Structures using TexGen. *Advanced Materials Research*, Vols. 331, pp 44-47.

## License

GPL-2.0-or-later — see [LICENSE](https://github.com/yufangjie1643/pytexgen/blob/main/LICENSE) for details.

Full documentation and more examples: [GitHub](https://github.com/yufangjie1643/pytexgen)
