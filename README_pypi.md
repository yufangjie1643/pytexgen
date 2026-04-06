# pytexgen

Python bindings for **TexGen**, a geometric textile modeller developed at the University of Nottingham.

pytexgen provides a comprehensive toolkit for modelling textile structures including 2D/3D weaves, knits, braids, and layered fabrics. It supports mesh generation (voxel, tetrahedral, surface), finite element export (Abaqus, ANSYS), and geometric analysis.

## Installation

```bash
pip install pytexgen
```

Requires Python 3.8 or later. Pre-built wheels are available for:

- **Linux**: x86_64, aarch64
- **macOS**: x86_64, arm64 (Apple Silicon)
- **Windows**: x86_64

## Quick Start

```python
import pytexgen

# Create a 2D plain weave
weave = pytexgen.CTextileWeave2D(4, 4, 1.0, 0.2)
weave.SwapPosition(0, 0)
weave.SwapPosition(1, 1)
weave.SwapPosition(2, 2)
weave.SwapPosition(3, 3)

# Assign to the global TexGen instance
pytexgen.AddTextile("PlainWeave", weave)

# Export to XML
pytexgen.SaveToXML("plain_weave.tgx")
```

## Features

- Model 2D and 3D woven, knitted, and braided textile structures
- Generate voxel, tetrahedral, and surface meshes
- Export to Abaqus and ANSYS for finite element analysis
- Define yarn paths, cross-sections, and material properties
- Compute fibre volume fractions and geometric properties

## License

GPL-2.0-or-later. See [LICENSE](https://github.com/your-org/TexGen/blob/main/LICENSE) for details.
