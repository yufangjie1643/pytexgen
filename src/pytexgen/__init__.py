"""
pytexgen - Python bindings for the TexGen textile geometry modeller.

TexGen is a geometric textile modeller capable of modelling a range of
textile structures, from simple 2D weaves to complex 3D woven, knitted,
and braided fabrics.

Usage:
    import pytexgen
    from pytexgen import CTextile, CTextileWeave2D, XYZ
"""

__version__ = "0.1.0"

# Import the SWIG-generated Core module (the compiled C++ bindings)
# This makes all Core classes/functions available directly on pytexgen,
# e.g. pytexgen.CTextile, pytexgen.CTextileWeave2D, pytexgen.XYZ, etc.
from .Core import *  # noqa: F401, F403

# Re-export convenience functions at package level
from .Core import (  # noqa: F401
    GetTextile,
    AddTextile,
    DeleteTextile,
    SaveToXML,
    ReadFromXML,
    DeleteTextiles,
    GetTextiles,
)

# Import bundled Python utility modules (optional, may have extra deps)
_optional_modules = [
    "Abaqus", "Ansys", "FlowTex", "GridFile",
    "WiseTex", "WeavePattern",
]

for _mod in _optional_modules:
    try:
        __import__(f"{__name__}.{_mod}")
    except ImportError:
        pass

del _optional_modules, _mod
