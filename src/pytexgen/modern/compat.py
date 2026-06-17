"""Compatibility helpers for reusing pytexgen's existing Python voxelizer."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_gpu_voxelizer():
    """Load the existing Python voxelizer from an install or source checkout."""
    try:
        from pytexgen import gpu_voxelizer

        return gpu_voxelizer
    except Exception:
        pass

    try:
        from TexGen import gpu_voxelizer

        return gpu_voxelizer
    except Exception:
        return _load_source_gpu_voxelizer()


def _load_source_gpu_voxelizer():
    root = Path(__file__).resolve().parents[3]
    path = root / "TexGen" / "gpu_voxelizer.py"
    if not path.exists():
        raise ImportError("modern voxelization requires pytexgen.gpu_voxelizer")

    pkg = sys.modules.get("TexGen")
    if pkg is None:
        pkg = types.ModuleType("TexGen")
        pkg.__path__ = [str(root / "TexGen")]
        sys.modules["TexGen"] = pkg

    core = sys.modules.get("TexGen.Core")
    if core is None or not hasattr(core, "CYarn") or not hasattr(core, "CTextile"):
        core = types.ModuleType("TexGen.Core")

        class CYarn:
            LINE = 1
            SURFACE = 2
            VOLUME = 4

        class CTextile:
            pass

        core.CYarn = CYarn
        core.CTextile = CTextile
        sys.modules["TexGen.Core"] = core

    spec = importlib.util.spec_from_file_location("TexGen.gpu_voxelizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gpu_voxelizer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
