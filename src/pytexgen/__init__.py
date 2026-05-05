"""
pytexgen - Python bindings for the TexGen textile geometry modeller.

TexGen is a geometric textile modeller capable of modelling a range of
textile structures, from simple 2D weaves to complex 3D woven, knitted,
and braided fabrics.

Usage:
    import pytexgen
    from pytexgen import CTextile, CTextileWeave2D, XYZ
"""

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _dist_version
except ImportError:  # pragma: no cover - Python < 3.8 compatibility guard
    _PackageNotFoundError = Exception
    _dist_version = None

try:
    __version__ = _dist_version("pytexgen") if _dist_version else "1.1.0"
except _PackageNotFoundError:
    __version__ = "1.1.0"

_CORE_IMPORT_ERROR = None
_CORE_AVAILABLE = False

try:
    # Import the SWIG-generated Core module (the compiled C++ bindings), attach
    # runtime docstrings, then re-export the Core API at package level.
    from . import Core as _CoreModule
except ImportError as _error:
    _CoreModule = None
    _CORE_IMPORT_ERROR = _error
else:
    _CORE_AVAILABLE = True
    try:
        from ._core_docs import apply_core_docstrings as _apply_core_docstrings

        _apply_core_docstrings(_CoreModule)
    except Exception:
        # Documentation must never make the compiled extension fail to import.
        pass

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


def _core_unavailable_message() -> str:
    return (
        "pytexgen was imported from the Python source tree, but the compiled "
        "TexGen Core extension is not available. Build or install the package "
        "first, for example `python -m pip install .`, then import pytexgen "
        "from that environment."
    )


def __getattr__(name):
    if not _CORE_AVAILABLE:
        raise ImportError(_core_unavailable_message()) from _CORE_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Import bundled Python utility modules (optional, may have extra deps)
_optional_modules = [
    "Abaqus", "Ansys", "FlowTex", "GridFile",
    "WiseTex", "WeavePattern",
]

if _CORE_AVAILABLE:
    for _mod in _optional_modules:
        try:
            __import__(f"{__name__}.{_mod}")
        except ImportError:
            pass

del _optional_modules, _CoreModule
try:
    del _apply_core_docstrings
except NameError:
    pass
