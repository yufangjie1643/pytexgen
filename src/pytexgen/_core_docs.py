"""Runtime docstrings for SWIG-wrapped TexGen Core objects.

The generated ``Core.py`` file is owned by SWIG and should not be edited by
hand. This module attaches Python docstrings after import so ``help()`` and IDE
hover text are useful without modifying generated files.
"""

from __future__ import annotations

import inspect
import re
from typing import Any


_MODULE_FUNCTION_DOCS = {
    "GetTextile": """Return a textile stored in TexGen's global registry.

Parameters
----------
TextileName : str, optional
    Name of the registered textile. When omitted, TexGen returns the current
    default textile.

Returns
-------
CTextile or None
    The registered textile object, or ``None`` if no matching textile exists.
""",
    "AddTextile": """Add a textile to TexGen's global registry.

Call signatures
---------------
AddTextile(textile) -> str
    Register ``textile`` under its default name and return the assigned name.
AddTextile(name, textile, bOverwrite=False) -> bool
    Register ``textile`` under ``name``. If ``bOverwrite`` is false, an existing
    textile with the same name is left unchanged.
""",
    "DeleteTextile": """Remove one textile from TexGen's global registry.

Parameters
----------
TextileName : str
    Name returned by :func:`AddTextile` or assigned by TexGen.

Returns
-------
bool
    ``True`` when a textile was removed.
""",
    "SaveToXML": """Save a registered textile collection to a TexGen XML file.

Parameters
----------
FileName : str
    Output ``.tg3`` path.
TextileName : str, optional
    Textile to save. The empty string saves the default/current textile.
OutputType : int, default=OUTPUT_STANDARD
    TexGen output mode constant.
""",
    "ReadFromXML": """Read TexGen textiles from a ``.tg3`` XML file.

Parameters
----------
FileName : str
    Input ``.tg3`` path.

Returns
-------
bool
    ``True`` when the file was loaded successfully.
""",
    "DeleteTextiles": """Remove all textiles from TexGen's global registry.""",
    "GetTextiles": """Return TexGen's global textile registry.

Returns
-------
TextileMap
    Mapping from textile names to ``CTextile`` pointers.
""",
}


_CLASS_DOCS = {
    "CTextile": """Container for yarns, domain, material properties and mesh export.

Typical use
-----------
Create a concrete textile such as ``CTextileWeave2D`` or ``CTextileLayerToLayer``,
set yarn widths/heights/resolution, call ``AssignDefaultDomain()``, then export
with ``SaveToXML`` or a voxel/surface mesh writer.
""",
    "CYarn": """Single yarn path with master nodes, interpolation, section shape and repeats.

Use ``AddNode`` to define the centerline, ``AssignSection`` to define the cross
section, ``AssignInterpolation`` for the path interpolation, and
``SetResolution`` before meshing.
""",
    "CTextileWeave": """Base class for woven textile unit cells.

Provides shared setters/getters for yarn widths, heights, spacings, resolution,
section meshes and default domains.
""",
    "CTextileWeave2D": """Two-dimensional woven textile unit cell.

Constructor
-----------
CTextileWeave2D(iWidth, iHeight, dSpacing, dThickness,
                bRefine=True, bInPlaneTangents=True)

Parameters
----------
iWidth, iHeight : int
    Number of cells/yarns in the two weave directions.
dSpacing : float
    Yarn spacing in model units.
dThickness : float
    Fabric thickness in model units.
bRefine : bool, default=True
    Refine generated yarns to reduce interference.
bInPlaneTangents : bool, default=True
    Force master-node tangents into the x-y plane.

Examples
--------
>>> weave = CTextileWeave2D(4, 4, 1.0, 0.2)
>>> weave.SetYarnWidths(0.8)
>>> weave.SetYarnHeights(0.1)
>>> weave.AssignDefaultDomain()
""",
    "CTextile3DWeave": """Three-dimensional woven textile base class.

Use layer editing methods such as ``AddXLayers``, ``AddYLayers``, ``PushUp`` and
``PushDown`` to control yarn ordering through the thickness.
""",
    "CTextileLayerToLayer": """Layer-to-layer 3D weave with binder yarn control.

The constructor defines warp/weft counts, yarn dimensions and binder layer
count. Use ``SetBinderPosition`` to route binder yarns through layers.
""",
    "CTextileOrthogonal": """Orthogonal 3D weave model with optional binder shaping/refinement.""",
    "CShearedTextileWeave2D": """Sheared 2D weave unit cell.

Use this when the textile plane is sheared by an angle instead of remaining
rectangular.
""",
    "CTextileLayered": """Textile assembled by stacking multiple textile layers with offsets.""",
    "CTextileKnit": """Base knit textile representation.""",
    "CTextileWeftKnit": """Weft-knit textile representation.""",
    "XYZ": """3D vector or point with ``x``, ``y`` and ``z`` components.

Supports basic arithmetic in Python through SWIG extension methods.
""",
    "XY": """2D vector or point with ``x`` and ``y`` components.""",
    "WXYZ": """Quaternion-like rotation value used by TexGen rotations.""",
    "CMesh": """Unstructured mesh container for nodes, elements and mesh data.""",
    "CNode": """Master node on a yarn centerline, including position and local frame.""",
    "CSlaveNode": """Interpolated yarn node used for surface and volume mesh generation.""",
    "CDomainPlanes": """Planar domain/bounding volume used for clipping and periodic repeats.""",
    "CDomainPrism": """Prismatic domain built from a 2D polygon cross-section.""",
    "CVoxelMesh": """Base class for TexGen C++ voxel mesh exporters.""",
    "CRectangularVoxelMesh": """Structured rectangular voxel mesh exporter.

Typical call
------------
>>> vox = CRectangularVoxelMesh("CPeriodicBoundaries")
>>> vox.SaveVoxelMesh(textile, "mesh.inp", 64, 64, 32, True, True, 5, 0)
""",
    "CSurfaceMesh": """Surface mesh exporter for textile yarns and domains.""",
    "CTetgenMesh": """Tetrahedral mesh exporter backed by TetGen.""",
    "CSectionEllipse": """Elliptical yarn cross-section.""",
    "CSectionPowerEllipse": """Power-ellipse yarn cross-section with adjustable shape exponent.""",
    "CSectionRectangle": """Rectangular yarn cross-section.""",
    "CSectionPolygon": """Polygon-defined yarn cross-section.""",
    "CYarnSectionConstant": """Yarn section that keeps one cross-section shape along the full yarn.""",
    "CYarnSectionInterpNode": """Yarn section interpolated between node-associated sections.""",
    "CYarnSectionInterpPosition": """Yarn section interpolated between sections placed by yarn position.""",
}


_METHOD_DOCS = {
    ("CTextile", "AddYarn"): """Add a yarn to the textile and return its yarn index.

Parameters
----------
Yarn : CYarn
    Yarn to copy into the textile.
""",
    ("CTextile", "DeleteYarn"): """Delete one yarn by index.

Returns ``False`` when the index is outside the yarn list.
""",
    ("CTextile", "DeleteYarns"): """Remove all yarns from this textile.""",
    ("CTextile", "AssignDomain"): """Assign a domain used for clipping, repeats and mesh export.""",
    ("CTextile", "RemoveDomain"): """Remove the currently assigned textile domain.""",
    ("CTextile", "SetResolution"): """Set mesh resolution for every yarn in the textile.

Call signature
--------------
SetResolution(iNumSectionPoints, iNumSlaveNodes=0) -> bool
""",
    ("CTextile", "GetPointInformation"): """Classify points relative to yarns.

The C++ implementation fills a ``PointInfoVector`` with yarn index, tangent,
local yarn coordinates, volume fraction, surface distance and orientation.
""",
    ("CTextile", "GetNumYarns"): """Return the number of yarns after building the textile if needed.""",
    ("CTextile", "GetYarn"): """Return a yarn by zero-based index.""",
    ("CTextile", "GetYarns"): """Return the textile yarn container.""",
    ("CTextile", "GetDomain"): """Return the assigned textile domain, or ``None`` if no domain is assigned.""",
    ("CTextile", "Translate"): """Translate the whole textile by an ``XYZ`` vector.""",
    ("CTextile", "Rotate"): """Rotate the whole textile by a ``WXYZ`` rotation around an optional origin.""",
    ("CTextile", "GetYarnLength"): """Return total yarn length in the requested units.""",
    ("CTextile", "GetYarnVolume"): """Return total yarn volume in the requested units.""",
    ("CTextile", "GetDomainVolumeFraction"): """Calculate fibre/yarn volume fraction inside the assigned domain.""",
    ("CTextileWeave", "SetYarnWidths"): """Set all warp and weft yarn widths to the same value.""",
    ("CTextileWeave", "SetYarnHeights"): """Set all warp and weft yarn heights to the same value.""",
    ("CTextileWeave", "SetYarnSpacings"): """Set all warp and weft yarn spacings to the same value.""",
    ("CTextileWeave", "SetXYarnWidths"): """Set x-direction yarn width for one index or for all x-yarns.""",
    ("CTextileWeave", "SetYYarnWidths"): """Set y-direction yarn width for one index or for all y-yarns.""",
    ("CTextileWeave", "SetXYarnHeights"): """Set x-direction yarn height for one index or for all x-yarns.""",
    ("CTextileWeave", "SetYYarnHeights"): """Set y-direction yarn height for one index or for all y-yarns.""",
    ("CTextileWeave", "SetXYarnSpacings"): """Set x-direction yarn spacing for one index or for all x-yarns.""",
    ("CTextileWeave", "SetYYarnSpacings"): """Set y-direction yarn spacing for one index or for all y-yarns.""",
    ("CTextileWeave", "AssignDefaultDomain"): """Create and assign TexGen's default domain for this weave.""",
    ("CTextileWeave", "AssignSectionMesh"): """Assign the same section mesh strategy to all yarn sections.""",
    ("CTextileWeave2D", "SwapPosition"): """Swap the over/under order at one weave cell ``(x, y)``.""",
    ("CTextileWeave2D", "SwapAll"): """Swap the over/under order in every weave cell.""",
    ("CTextileWeave2D", "RefineTextile"): """Refine the 2D weave after pattern or dimension changes.""",
    ("CTextileWeave2D", "SetInPlaneTangents"): """Enable or disable forcing master-node tangents into the x-y plane.""",
    ("CYarn", "AddNode"): """Append a master node to the yarn centerline.""",
    ("CYarn", "InsertNode"): """Insert a master node before another node or at a zero-based index.""",
    ("CYarn", "ReplaceNode"): """Replace one master node by zero-based index.""",
    ("CYarn", "DeleteNode"): """Delete one master node by zero-based index.""",
    ("CYarn", "GetNode"): """Return a master node by zero-based index.""",
    ("CYarn", "SetNodes"): """Replace all master nodes with an ordered node vector.""",
    ("CYarn", "AssignInterpolation"): """Assign the interpolation curve used between master nodes.""",
    ("CYarn", "AssignSection"): """Assign the yarn cross-section model.""",
    ("CYarn", "AssignFibreDistribution"): """Assign the fibre distribution model inside the yarn section.""",
    ("CYarn", "SetResolution"): """Set slave-node and section-point resolution for generated meshes.""",
    ("CYarn", "SetEquiSpacedSectionMesh"): """Choose whether generated section mesh points are equispaced.""",
    ("CYarn", "AddRepeat"): """Add one repeat vector for periodic copies of this yarn.""",
    ("CYarn", "ClearRepeats"): """Remove all repeat vectors from this yarn.""",
    ("CYarn", "AddSurfaceToMesh"): """Generate yarn surface elements and append them to a ``CMesh``.""",
    ("CYarn", "AddVolumeToMesh"): """Generate yarn volume elements and append them to a ``CMesh``.""",
    ("CYarn", "PointInsideYarn"): """Return whether a point lies inside this yarn.

Optional output arguments receive tangent, local section coordinates, volume
fraction, surface distance, orientation and up vector.
""",
    ("CVoxelMesh", "SaveVoxelMesh"): """Write a voxel mesh for a textile.

Concrete subclasses such as ``CRectangularVoxelMesh`` define the element
layout. For pure Python numpy/torch acceleration, use
``pytexgen.gpu_voxelizer.voxelize_textile``.
""",
    ("CSurfaceMesh", "SaveSurfaceMesh"): """Build a textile surface mesh in memory.""",
    ("CSurfaceMesh", "SaveToSTL"): """Write the current surface mesh to STL.""",
    ("CSurfaceMesh", "SaveToVTK"): """Write the current surface mesh to VTK.""",
}


_VERB_DOCS = {
    "Get": "Return {subject} from {owner}.",
    "Set": "Set {subject} on {owner}.",
    "Add": "Add {subject} to {owner}.",
    "Delete": "Delete {subject} from {owner}.",
    "Remove": "Remove {subject} from {owner}.",
    "Clear": "Clear {subject} on {owner}.",
    "Save": "Save {subject} for {owner}.",
    "Read": "Read {subject} into {owner}.",
    "Create": "Create {subject} for {owner}.",
    "Build": "Build {subject} for {owner}.",
    "Assign": "Assign {subject} to {owner}.",
    "Output": "Output {subject} from {owner}.",
    "Rotate": "Rotate {owner}.",
    "Translate": "Translate {owner}.",
    "Copy": "Return a copy of {owner}.",
    "Populate": "Populate {subject} from {owner}.",
    "Insert": "Insert {subject} into {owner}.",
    "Replace": "Replace {subject} on {owner}.",
    "Swap": "Swap {subject} on {owner}.",
    "Move": "Move {subject} on {owner}.",
    "Push": "Push {subject} on {owner}.",
    "Convert": "Convert {subject} for {owner}.",
    "Detect": "Detect {subject} for {owner}.",
    "Flatten": "Flatten {subject} for {owner}.",
    "Correct": "Correct {subject} for {owner}.",
    "Valid": "Return whether {owner} is valid.",
}


def _humanize(name: str) -> str:
    """Convert a CamelCase API name into readable words."""
    if not name:
        return "the value"
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip().lower()
    return text or name


def _set_doc(obj: Any, doc: str) -> None:
    """Best-effort assignment of ``__doc__`` on SWIG proxy objects."""
    try:
        obj.__doc__ = doc.strip() + "\n"
    except (AttributeError, TypeError):
        pass


def _generic_doc(name: str, owner: str | None = None) -> str:
    """Create a readable fallback docstring from a TexGen API name."""
    owner_text = owner or "TexGen Core"
    if name == "__init__":
        return (
            f"Create a new {owner_text} instance.\n\n"
            "Exact constructor overloads are provided by the SWIG-generated "
            "wrapper and the corresponding C++ header."
        )
    if name.startswith("__"):
        return f"Python special method for {owner_text}."

    for verb, template in _VERB_DOCS.items():
        if name.startswith(verb):
            subject = _humanize(name[len(verb):])
            doc = template.format(subject=subject, owner=owner_text)
            break
    else:
        doc = f"Call the SWIG-wrapped TexGen C++ API ``{name}`` on {owner_text}."

    return (
        f"{doc}\n\n"
        "Notes\n"
        "-----\n"
        "This is a SWIG-wrapped C++ function. If an overload is called with the "
        "wrong argument types, SWIG's ``TypeError`` message lists the accepted "
        "C++ prototypes. The authoritative implementation is in ``Core/*.h`` "
        "and ``Core/*.cpp``."
    )


def _is_user_callable(name: str, value: Any) -> bool:
    """Return whether a class/module member should receive a docstring."""
    if name in {"thisown", "__swig_destroy__"}:
        return False
    if name.startswith("_") and name not in {"__init__", "__repr__"}:
        return False
    return callable(value)


def apply_core_docstrings(core_module: Any) -> None:
    """Attach docstrings to SWIG Core classes, methods and module functions.

    Parameters
    ----------
    core_module : module
        Imported ``pytexgen.Core`` module.
    """
    for name, doc in _MODULE_FUNCTION_DOCS.items():
        obj = getattr(core_module, name, None)
        if obj is not None:
            _set_doc(obj, doc)

    for name in dir(core_module):
        obj = getattr(core_module, name, None)
        if obj is None:
            continue

        if inspect.isclass(obj):
            _set_doc(
                obj,
                _CLASS_DOCS.get(
                    name,
                    f"SWIG wrapper for the TexGen C++ class ``{name}``.\n\n"
                    "Use ``help(instance.method)`` for method-level notes. "
                    "The exact overloads are defined in the matching C++ header."
                ),
            )
            for attr_name, attr_value in vars(obj).items():
                if not _is_user_callable(attr_name, attr_value):
                    continue
                doc = _METHOD_DOCS.get((name, attr_name))
                if doc is None:
                    doc = _generic_doc(attr_name, owner=name)
                _set_doc(attr_value, doc)
        elif _is_user_callable(name, obj) and name not in _MODULE_FUNCTION_DOCS:
            _set_doc(obj, _generic_doc(name))


__all__ = ["apply_core_docstrings"]
