"""
Python voxelization bypass for TexGen.

Drop-in replacement for ``CRectangularVoxelMesh.SaveVoxelMesh(...)`` that:
  1. Takes a fully-built ``CTextile`` from TexGen (all refine / interference /
     section-mesh work already done by TexGen's C++ core).
  2. Snapshots each yarn's slave-node frame + section polygon into plain arrays.
  3. Classifies every voxel center by "point-in-swept-polygon" test using a
     portable numpy CPU backend or an optional torch backend.
  4. Writes an Abaqus ``.inp`` file compatible with TexGen's own output format
     (hex elements + per-element yarn index).

Design goals:
  * Avoid OpenMP as a required compiled dependency for wheel builds.
  * Works with any ``CTextile`` subclass (2D/3D/sheared/orthogonal/...).
  * Numpy backend: portable CPU vectorization with no compiler/runtime OpenMP.
  * Adaptive numpy mode: lightweight linear-octree refinement without p4est.
  * Conservative AABB pruning avoids testing yarns that cannot hit a voxel chunk.
  * Torch backend: optional CUDA, Metal (MPS), or CPU acceleration.

Usage:
    from pytexgen import *
    from pytexgen.gpu_voxelizer import voxelize_textile, voxelize_textile_data

    T = CShearedTextileWeave2D(3,3,5.0,2.0,0.2618,True,True)
    T.SetYarnWidths(2.0); T.SetYarnHeights(0.8); T.AssignDefaultDomain()

    voxelize_textile(T, nx=64, ny=64, nz=64, out_inp="out.inp", backend="numpy")
    data = voxelize_textile_data(T, nx=64, ny=64, nz=64, backend="torch")
    material_grid = data.material_id()
    voxelize_textile(T, nx=16, ny=16, nz=8, out_inp="adaptive.inp",
                     backend="numpy", adaptive=True, adaptive_levels=2)
    voxelize_textile(T, nx=64, ny=64, nz=64, out_inp="out_torch.inp", backend="torch")
"""

from __future__ import annotations

import math
import importlib
import json
import os
import time
from collections.abc import Mapping as MappingABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

DEFAULT_NUMPY_CHUNK_VOXELS = 8192


def _progress_iter(iterable: Iterable[Any],
                   progress: Any = False,
                   *,
                   total: Optional[int] = None,
                   desc: Optional[str] = None,
                   unit: str = "it") -> Iterable[Any]:
    """Wrap an iterable with tqdm when progress reporting is requested.

    ``progress`` may be ``True`` to use ``tqdm.auto.tqdm`` or a callable with a
    tqdm-compatible signature for tests and host applications. Keeping tqdm
    lazy avoids adding it to the base runtime dependency set.
    """
    if not progress:
        return iterable
    if callable(progress):
        return progress(iterable, total=total, desc=desc, unit=unit)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "progress=True requires tqdm. Install it with `pip install tqdm` "
            'or `pip install "pytexgen[progress]"`.'
        ) from exc
    return tqdm(iterable, total=total, desc=desc, unit=unit)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from .Core import CTextile, CYarn  # type: ignore
except ImportError:
    from TexGen.Core import CTextile, CYarn  # type: ignore

try:
    from .material_fields import SparseOrientationField
except ImportError:
    from TexGen.material_fields import SparseOrientationField

# BUILD_TYPE bitmask constants from CYarn. SWIG exposes them as CYarn.SURFACE etc.
# Fallback to raw values if the enum binding differs.
try:
    _LINE    = CYarn.LINE
    _SURFACE = CYarn.SURFACE
    _VOLUME  = CYarn.VOLUME
except AttributeError:
    _LINE, _SURFACE, _VOLUME = 1 << 0, 1 << 1, 1 << 2


def _require_torch():
    """Return the imported torch module or raise a user-facing install hint.

    Returns
    -------
    module
        The already-imported PyTorch module.

    Raises
    ------
    ImportError
        If PyTorch is not installed and the torch backend was requested.
    """
    if torch is None:
        raise ImportError(
            'Torch backend requested but PyTorch is not installed. '
            'Install with `pip install "pytexgen[gpu]"` or use backend="numpy".'
        )
    return torch


# ---------------------------------------------------------------------------
# Geometry snapshot: extract plain numpy arrays from TexGen's C++ objects.
# ---------------------------------------------------------------------------


@dataclass
class YarnSnapshot:
    """Per-yarn array-friendly geometry snapshot."""
    positions: np.ndarray      # (M, 3) slave node world positions
    tangents:  np.ndarray      # (M, 3) unit tangent along yarn length
    ups:       np.ndarray      # (M, 3) unit up (perpendicular to tangent)
    sides:     np.ndarray      # (M, 3) unit side = tangent x up (right-handed frame)
    section:   np.ndarray      # (N, 2) 2D polygon (u=side, v=up) at each slave node
    translations: np.ndarray   # (K, 3) periodic-image translations (includes origin)


def _snapshot_matrix(name: str, value: Any, width: int) -> np.ndarray:
    """Coerce a provider array to a 2D numpy matrix with a fixed column count."""
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    return arr


def _snapshot_offsets(name: str, value: Any, total: int) -> np.ndarray:
    """Coerce and validate a monotonic offset table for flat snapshot arrays."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D offset array")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer offsets")
    offsets = raw.astype(np.int64, copy=False)
    if len(offsets) == 0:
        raise ValueError(f"{name} must contain at least the initial zero offset")
    if int(offsets[0]) != 0:
        raise ValueError(f"{name} must start at 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be nondecreasing")
    if int(offsets[-1]) != int(total):
        raise ValueError(
            f"{name} final offset {int(offsets[-1])} does not match flat array "
            f"length {int(total)}"
        )
    return offsets


@dataclass
class AdaptiveVoxelCells:
    """Leaf cells for lightweight linear-octree voxel output."""
    lows: np.ndarray       # (E, 3) lower corner per cell
    sizes: np.ndarray      # (E, 3) cell dimensions
    levels: np.ndarray     # (E,) refinement level relative to the base grid
    yarn_id: np.ndarray    # (E,) owning yarn index (-1 = matrix)


@dataclass
class SnapshotBundle:
    """Structure-of-arrays snapshot for high-throughput C++/Python handoff.

    The existing public voxelizer operates on ``YarnSnapshot`` objects for
    readability. Future C++/nanobind providers can return this flatter layout
    instead, avoiding thousands of small SWIG object reads while keeping the
    downstream numpy/torch voxelizer API stable.
    """
    positions: np.ndarray
    tangents: np.ndarray
    ups: np.ndarray
    sides: np.ndarray
    node_offsets: np.ndarray
    sections: np.ndarray
    section_offsets: np.ndarray
    translations: np.ndarray
    translation_offsets: np.ndarray
    aabb: np.ndarray

    def __post_init__(self) -> None:
        self.positions = _snapshot_matrix("positions", self.positions, 3)
        self.tangents = _snapshot_matrix("tangents", self.tangents, 3)
        self.ups = _snapshot_matrix("ups", self.ups, 3)
        self.sides = _snapshot_matrix("sides", self.sides, 3)
        self.sections = _snapshot_matrix("sections", self.sections, 2)
        self.translations = _snapshot_matrix("translations", self.translations, 3)
        self.aabb = np.asarray(self.aabb, dtype=np.float64)
        if self.aabb.shape != (2, 3):
            raise ValueError("aabb must have shape (2, 3)")

        self.node_offsets = _snapshot_offsets(
            "node_offsets", self.node_offsets, self.positions.shape[0]
        )
        self.section_offsets = _snapshot_offsets(
            "section_offsets", self.section_offsets, self.sections.shape[0]
        )
        self.translation_offsets = _snapshot_offsets(
            "translation_offsets",
            self.translation_offsets,
            self.translations.shape[0],
        )

        for name in ("tangents", "ups", "sides"):
            arr = getattr(self, name)
            if arr.shape != self.positions.shape:
                raise ValueError(
                    f"{name} must have the same shape as positions "
                    f"{self.positions.shape}"
                )

        offset_lengths = {
            len(self.node_offsets),
            len(self.section_offsets),
            len(self.translation_offsets),
        }
        if len(offset_lengths) != 1:
            raise ValueError(
                "node_offsets, section_offsets, and translation_offsets must "
                "have the same length"
            )

    @property
    def num_yarns(self) -> int:
        """Return the number of yarn snapshots stored in this bundle."""
        return int(len(self.node_offsets) - 1)

    @classmethod
    def from_snapshots(cls,
                       snapshots: List[YarnSnapshot],
                       aabb: np.ndarray) -> "SnapshotBundle":
        """Pack per-yarn snapshots into flat arrays plus offset tables."""
        node_offsets = [0]
        section_offsets = [0]
        translation_offsets = [0]
        positions = []
        tangents = []
        ups = []
        sides = []
        sections = []
        translations = []

        for snap in snapshots:
            positions.append(np.asarray(snap.positions))
            tangents.append(np.asarray(snap.tangents))
            ups.append(np.asarray(snap.ups))
            sides.append(np.asarray(snap.sides))
            sections.append(np.asarray(snap.section))
            translations.append(np.asarray(snap.translations))
            node_offsets.append(node_offsets[-1] + int(snap.positions.shape[0]))
            section_offsets.append(section_offsets[-1] + int(snap.section.shape[0]))
            translation_offsets.append(
                translation_offsets[-1] + int(snap.translations.shape[0])
            )

        coord_dtype = (
            np.asarray(snapshots[0].positions).dtype if snapshots else np.float64
        )
        section_dtype = (
            np.asarray(snapshots[0].section).dtype if snapshots else coord_dtype
        )

        return cls(
            positions=(
                np.concatenate(positions, axis=0)
                if positions else np.empty((0, 3), dtype=coord_dtype)
            ),
            tangents=(
                np.concatenate(tangents, axis=0)
                if tangents else np.empty((0, 3), dtype=coord_dtype)
            ),
            ups=(
                np.concatenate(ups, axis=0)
                if ups else np.empty((0, 3), dtype=coord_dtype)
            ),
            sides=(
                np.concatenate(sides, axis=0)
                if sides else np.empty((0, 3), dtype=coord_dtype)
            ),
            node_offsets=np.asarray(node_offsets, dtype=np.int64),
            sections=(
                np.concatenate(sections, axis=0)
                if sections else np.empty((0, 2), dtype=section_dtype)
            ),
            section_offsets=np.asarray(section_offsets, dtype=np.int64),
            translations=(
                np.concatenate(translations, axis=0)
                if translations else np.empty((0, 3), dtype=coord_dtype)
            ),
            translation_offsets=np.asarray(translation_offsets, dtype=np.int64),
            aabb=np.asarray(aabb, dtype=np.float64),
        )

    def to_snapshots(self) -> List[YarnSnapshot]:
        """Unpack this bundle into the existing ``YarnSnapshot`` representation."""
        snapshots: List[YarnSnapshot] = []
        for index in range(self.num_yarns):
            n0, n1 = self.node_offsets[index:index + 2]
            s0, s1 = self.section_offsets[index:index + 2]
            t0, t1 = self.translation_offsets[index:index + 2]
            snapshots.append(
                YarnSnapshot(
                    positions=self.positions[n0:n1],
                    tangents=self.tangents[n0:n1],
                    ups=self.ups[n0:n1],
                    sides=self.sides[n0:n1],
                    section=self.sections[s0:s1],
                    translations=self.translations[t0:t1],
                )
            )
        return snapshots


@dataclass
class BackendSelection:
    """Resolved numerical backend settings."""
    backend: str
    device: str
    workers: int
    np_dtype: Optional[type] = None
    torch_dtype: Optional[object] = None
    torch_module: Optional[object] = None


@dataclass
class VoxelGridData:
    """Structured voxel data for direct numpy/torch solver handoff.

    ``yarn_id`` is stored in TexGen element order:
    ``ix + iy*nx + iz*nx*ny``. The ``grid`` property exposes the same data as a
    ``(nz, ny, nx)`` view without copying for both numpy arrays and torch
    tensors.
    """
    yarn_id: Any
    aabb: Any
    resolution: Tuple[int, int, int]
    backend: str
    device: str
    workers: int
    dtype: str
    timings: Dict[str, float]
    centers: Optional[Any] = None
    orientation1: Optional[Any] = None
    orientation2: Optional[Any] = None
    sparse_orientation: Optional[SparseOrientationField] = None
    aabb_pruning: bool = True
    storage: str = "numpy"
    order: str = "ix + iy*nx + iz*nx*ny"

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the structured grid shape as ``(nz, ny, nx)``."""
        nx, ny, nz = self.resolution
        return (nz, ny, nx)

    @property
    def grid(self):
        """Return ``yarn_id`` as a zero-copy ``(nz, ny, nx)`` view."""
        return self.yarn_id.reshape(self.shape)

    @property
    def voxel_size(self):
        """Return voxel spacing along x, y and z in the current array backend."""
        nx, ny, nz = self.resolution
        if _is_torch_tensor(self.aabb):
            denom = self.aabb.new_tensor([nx, ny, nz], dtype=self.aabb.dtype)
        else:
            aabb_np = np.asarray(self.aabb)
            denom = np.asarray([nx, ny, nz], dtype=aabb_np.dtype)
        return (self.aabb[1] - self.aabb[0]) / denom

    def occupancy(self):
        """Return a boolean grid where yarn voxels are true."""
        return self.grid >= 0

    def material_id(self, matrix_id: int = 0, yarn_offset: int = 1):
        """Return solver-friendly material ids as a ``(nz, ny, nx)`` grid.

        Matrix voxels are assigned ``matrix_id``. Yarn ids are shifted by
        ``yarn_offset`` so TexGen yarn 0 can map to material id 1 by default.
        """
        grid = self.grid
        if _is_torch_tensor(grid):
            torch_mod = _require_torch()
            matrix = grid.new_full(grid.shape, int(matrix_id))
            return torch_mod.where(grid >= 0, grid + int(yarn_offset), matrix)
        return np.where(grid >= 0, grid + int(yarn_offset), int(matrix_id))

    def to(self,
           storage: Optional[str] = None,
           *,
           device: Optional[str] = None,
           dtype: Optional[object] = None,
           copy: bool = False) -> "VoxelGridData":
        """Return this voxel data in another array storage backend.

        This follows the spirit of ``torch.Tensor.to(...)`` at the container
        level: it converts the stored arrays while preserving metadata and the
        public ``VoxelGridData`` API. Use ``data.grid`` or ``data.material_id()``
        to pass individual arrays/tensors to downstream code.

        Parameters
        ----------
        storage : {"numpy", "torch", None}
            Target array backend. ``None`` keeps the current backend and only
            applies device/copy handling when the data is torch-backed.
        device : str or None
            Target torch device. Ignored for numpy output.
        dtype : object or None
            Target floating-point dtype, for example ``"float32"``,
            ``np.float32`` or ``torch.float32``. Integer label arrays such as
            ``yarn_id`` are intentionally kept as integers.
        copy : bool
            Force a copy/clone even when the current storage already matches.
        """
        target = self.storage if storage is None else storage.lower()
        if target not in {"numpy", "torch"}:
            raise ValueError('storage must be "numpy", "torch", or None')
        if target == "numpy":
            return self.to_numpy(copy=copy, dtype=dtype)
        return self.to_torch(device=device, copy=copy, dtype=dtype)

    def to_numpy(self, copy: bool = False,
                 dtype: Optional[object] = None) -> "VoxelGridData":
        """Return an equivalent data object backed by numpy arrays.

        Prefer ``data.to("numpy")`` in new code.
        """
        np_dtype = _resolve_numpy_array_dtype(dtype)
        yarn_id = _array_to_numpy(self.yarn_id, copy=copy)
        aabb = _array_to_numpy(self.aabb, copy=copy)
        centers = None if self.centers is None else _array_to_numpy(self.centers, copy=copy)
        orientation1 = (
            None if self.orientation1 is None
            else _array_to_numpy(self.orientation1, copy=copy)
        )
        orientation2 = (
            None if self.orientation2 is None
            else _array_to_numpy(self.orientation2, copy=copy)
        )
        sparse_orientation = (
            None
            if self.sparse_orientation is None
            else self.sparse_orientation.to(
                "numpy", dtype=np_dtype, copy=copy
            )
        )
        if np_dtype is not None:
            aabb = aabb.astype(np_dtype, copy=copy or aabb.dtype != np_dtype)
            if centers is not None:
                centers = centers.astype(np_dtype, copy=copy or centers.dtype != np_dtype)
            if orientation1 is not None:
                orientation1 = orientation1.astype(
                    np_dtype, copy=copy or orientation1.dtype != np_dtype
                )
            if orientation2 is not None:
                orientation2 = orientation2.astype(
                    np_dtype, copy=copy or orientation2.dtype != np_dtype
                )
        return VoxelGridData(
            yarn_id=yarn_id,
            aabb=aabb,
            resolution=self.resolution,
            backend=self.backend,
            device="cpu",
            workers=self.workers,
            dtype=self.dtype if np_dtype is None else np_dtype.name,
            timings=dict(self.timings),
            centers=centers,
            orientation1=orientation1,
            orientation2=orientation2,
            sparse_orientation=sparse_orientation,
            aabb_pruning=self.aabb_pruning,
            storage="numpy",
            order=self.order,
        )

    def to_torch(self, device: Optional[str] = None,
                 copy: bool = False,
                 dtype: Optional[object] = None) -> "VoxelGridData":
        """Return an equivalent data object backed by torch tensors.

        Prefer ``data.to("torch", device=...)`` in new code.
        """
        torch_mod = _require_torch()
        yarn_id = _array_to_torch(self.yarn_id, torch_mod, device=device, copy=copy)
        torch_dtype = _resolve_torch_array_dtype(torch_mod, dtype)
        aabb = _array_to_torch(
            self.aabb, torch_mod, device=str(yarn_id.device), copy=copy
        )
        if torch_dtype is not None:
            aabb = aabb.to(dtype=torch_dtype, copy=copy)
        centers = None
        if self.centers is not None:
            centers = _array_to_torch(
                self.centers, torch_mod, device=str(yarn_id.device), copy=copy
            )
            if torch_dtype is not None:
                centers = centers.to(dtype=torch_dtype, copy=copy)
        orientation1 = None
        if self.orientation1 is not None:
            orientation1 = _array_to_torch(
                self.orientation1, torch_mod, device=str(yarn_id.device), copy=copy
            )
            if torch_dtype is not None:
                orientation1 = orientation1.to(dtype=torch_dtype, copy=copy)
        orientation2 = None
        if self.orientation2 is not None:
            orientation2 = _array_to_torch(
                self.orientation2, torch_mod, device=str(yarn_id.device), copy=copy
            )
            if torch_dtype is not None:
                orientation2 = orientation2.to(dtype=torch_dtype, copy=copy)
        sparse_orientation = (
            None
            if self.sparse_orientation is None
            else self.sparse_orientation.to(
                "torch",
                device=str(yarn_id.device),
                dtype=torch_dtype,
                copy=copy,
            )
        )
        return VoxelGridData(
            yarn_id=yarn_id,
            aabb=aabb,
            resolution=self.resolution,
            backend=self.backend,
            device=str(yarn_id.device),
            workers=self.workers,
            dtype=self.dtype if torch_dtype is None else _torch_dtype_name(torch_dtype),
            timings=dict(self.timings),
            centers=centers,
            orientation1=orientation1,
            orientation2=orientation2,
            sparse_orientation=sparse_orientation,
            aabb_pruning=self.aabb_pruning,
            storage="torch",
            order=self.order,
        )

    def save_npz(self, path: str,
                 *,
                 compressed: bool = True,
                 include_centers: bool = True) -> None:
        """Save voxel data to a portable numpy ``.npz`` file.

        Tensor-backed data is copied to CPU numpy arrays before writing. The
        saved file is intended for fast reload into numpy or torch without
        going through Abaqus ``.inp`` text output.
        """
        payload = {
            "yarn_id": _array_to_numpy(self.yarn_id, copy=False),
            "aabb": _array_to_numpy(self.aabb, copy=False),
            "format": np.asarray("pytexgen.voxel_grid_npz"),
            "format_version": np.asarray(2, dtype=np.int64),
            "resolution": np.asarray(self.resolution, dtype=np.int64),
            "aabb_pruning": np.asarray(bool(self.aabb_pruning)),
            "backend": np.asarray(self.backend),
            "device": np.asarray(self.device),
            "dtype": np.asarray(self.dtype),
            "storage": np.asarray(self.storage),
            "order": np.asarray(self.order),
            "timings_json": np.asarray(json.dumps(self.timings)),
        }
        if include_centers and self.centers is not None:
            payload["centers"] = _array_to_numpy(self.centers, copy=False)
        if self.orientation1 is not None:
            payload["orientation1"] = _array_to_numpy(self.orientation1, copy=False)
        if self.orientation2 is not None:
            payload["orientation2"] = _array_to_numpy(self.orientation2, copy=False)
        if self.sparse_orientation is not None:
            sparse = self.sparse_orientation
            payload.update(
                orientation_voxel_indices=_array_to_numpy(
                    sparse.voxel_indices, copy=False
                ),
                orientation_yarn_ids=_array_to_numpy(
                    sparse.yarn_ids, copy=False
                ),
                sparse_orientation1=_array_to_numpy(
                    sparse.orientation1, copy=False
                ),
                sparse_orientation2=_array_to_numpy(
                    sparse.orientation2, copy=False
                ),
                orientation_order=np.asarray(sparse.order),
            )

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if compressed else np.savez
        saver(out_path, **payload)

    def save_npy_dir(self, path: str,
                     *,
                     include_centers: bool = True) -> None:
        """Save voxel data as a directory of raw ``.npy`` arrays.

        Compared with ``.npz``, this avoids zip packaging/decompression and
        allows :meth:`load_npy_dir` to memory-map individual arrays. Tensor-
        backed data is copied to CPU numpy arrays before writing.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        arrays = {
            "yarn_id": ("yarn_id.npy", self.yarn_id),
            "aabb": ("aabb.npy", self.aabb),
        }
        if include_centers and self.centers is not None:
            arrays["centers"] = ("centers.npy", self.centers)
        if self.orientation1 is not None:
            arrays["orientation1"] = ("orientation1.npy", self.orientation1)
        if self.orientation2 is not None:
            arrays["orientation2"] = ("orientation2.npy", self.orientation2)
        if self.sparse_orientation is not None:
            sparse = self.sparse_orientation
            arrays.update(
                orientation_voxel_indices=(
                    "orientation_voxel_indices.npy", sparse.voxel_indices
                ),
                orientation_yarn_ids=(
                    "orientation_yarn_ids.npy", sparse.yarn_ids
                ),
                sparse_orientation1=(
                    "sparse_orientation1.npy", sparse.orientation1
                ),
                sparse_orientation2=(
                    "sparse_orientation2.npy", sparse.orientation2
                ),
            )

        for filename, value in arrays.values():
            np.save(
                out_dir / filename,
                _array_to_numpy(value, copy=False),
                allow_pickle=False,
            )

        metadata = {
            "format": "pytexgen.voxel_grid_npy_dir",
            "format_version": 2,
            "resolution": list(self.resolution),
            "aabb_pruning": bool(self.aabb_pruning),
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "storage": self.storage,
            "order": self.order,
            "workers": int(self.workers),
            "timings": dict(self.timings),
            "arrays": {
                field: filename for field, (filename, _value) in arrays.items()
            },
        }
        if self.sparse_orientation is not None:
            metadata["orientation_order"] = self.sparse_orientation.order
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_npz(cls, path: str,
                 *,
                 output: str = "numpy",
                 device: Optional[str] = None) -> "VoxelGridData":
        """Load voxel data saved by :meth:`save_npz`.

        Parameters
        ----------
        output : {"numpy", "torch"}
            Storage backend for the returned object.
        device : str or None
            Torch device when ``output="torch"``.
        """
        output = output.lower()
        if output not in {"numpy", "torch"}:
            raise ValueError('output must be "numpy" or "torch"')

        with np.load(path, allow_pickle=False) as data:
            timings_raw = str(data["timings_json"].item())
            centers = data["centers"].copy() if "centers" in data.files else None
            orientation1 = (
                data["orientation1"].copy() if "orientation1" in data.files else None
            )
            orientation2 = (
                data["orientation2"].copy() if "orientation2" in data.files else None
            )
            sparse_names = {
                "orientation_voxel_indices",
                "orientation_yarn_ids",
                "sparse_orientation1",
                "sparse_orientation2",
            }
            sparse_present = sparse_names.intersection(data.files)
            if sparse_present and sparse_present != sparse_names:
                missing = sorted(sparse_names - sparse_present)
                raise ValueError(
                    f"sparse orientation archive is missing arrays: {missing}"
                )
            resolution = tuple(
                int(v) for v in data["resolution"].tolist()
            )
            sparse_orientation = None
            if sparse_present:
                sparse_orientation = SparseOrientationField(
                    voxel_indices=data["orientation_voxel_indices"].copy(),
                    yarn_ids=data["orientation_yarn_ids"].copy(),
                    orientation1=data["sparse_orientation1"].copy(),
                    orientation2=data["sparse_orientation2"].copy(),
                    grid_shape=(resolution[2], resolution[1], resolution[0]),
                    order=(
                        str(data["orientation_order"].item())
                        if "orientation_order" in data.files
                        else str(data["order"].item())
                    ),
                )
            obj = cls(
                yarn_id=data["yarn_id"].copy(),
                aabb=data["aabb"].copy(),
                resolution=resolution,
                backend=str(data["backend"].item()),
                device="cpu",
                workers=1,
                dtype=str(data["dtype"].item()),
                timings=json.loads(timings_raw),
                centers=centers,
                orientation1=orientation1,
                orientation2=orientation2,
                sparse_orientation=sparse_orientation,
                aabb_pruning=bool(data["aabb_pruning"].item()),
                storage="numpy",
                order=str(data["order"].item()),
            )

        if output == "torch":
            return obj.to_torch(device=device, copy=False)
        return obj

    @classmethod
    def load_npy_dir(cls, path: str,
                     *,
                     output: str = "numpy",
                     device: Optional[str] = None,
                     mmap_mode: Optional[str] = None) -> "VoxelGridData":
        """Load voxel data saved by :meth:`save_npy_dir`.

        Parameters
        ----------
        output : {"numpy", "torch"}
            Storage backend for the returned object.
        device : str or None
            Torch device when ``output="torch"``.
        mmap_mode : str or None
            Passed to ``numpy.load`` for numpy output. Use ``"r"`` for
            read-only memory-mapped numpy arrays, which reduces CPU and memory
            pressure for large grids.
        """
        output = output.lower()
        if output not in {"numpy", "torch"}:
            raise ValueError('output must be "numpy" or "torch"')
        array_mmap_mode = mmap_mode if output == "numpy" else None

        in_dir = Path(path)
        metadata_path = in_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "pytexgen.voxel_grid_npy_dir":
            raise ValueError("unsupported voxel grid directory format")
        format_version = int(metadata.get("format_version", 1))
        if format_version not in {1, 2}:
            raise ValueError(
                f"unsupported voxel grid directory format version {format_version}"
            )
        arrays = metadata.get("arrays", {})

        def load_array(field: str, *, required: bool = False):
            filename = arrays.get(field)
            if filename is None:
                if required:
                    raise ValueError(f"metadata.json does not list {field!r}")
                return None
            return np.load(
                in_dir / filename,
                allow_pickle=False,
                mmap_mode=array_mmap_mode,
            )

        sparse_names = {
            "orientation_voxel_indices",
            "orientation_yarn_ids",
            "sparse_orientation1",
            "sparse_orientation2",
        }
        sparse_present = sparse_names.intersection(arrays)
        if sparse_present and sparse_present != sparse_names:
            missing = sorted(sparse_names - sparse_present)
            raise ValueError(
                f"sparse orientation metadata is missing arrays: {missing}"
            )
        resolution = tuple(int(v) for v in metadata["resolution"])
        sparse_orientation = None
        if sparse_present:
            sparse_orientation = SparseOrientationField(
                voxel_indices=load_array(
                    "orientation_voxel_indices", required=True
                ),
                yarn_ids=load_array("orientation_yarn_ids", required=True),
                orientation1=load_array("sparse_orientation1", required=True),
                orientation2=load_array("sparse_orientation2", required=True),
                grid_shape=(resolution[2], resolution[1], resolution[0]),
                order=str(metadata.get(
                    "orientation_order",
                    metadata.get("order", "ix + iy*nx + iz*nx*ny"),
                )),
            )

        obj = cls(
            yarn_id=load_array("yarn_id", required=True),
            aabb=load_array("aabb", required=True),
            resolution=resolution,
            backend=str(metadata.get("backend", "numpy")),
            device="cpu",
            workers=int(metadata.get("workers", 1)),
            dtype=str(metadata.get("dtype", "float64")),
            timings=dict(metadata.get("timings", {})),
            centers=load_array("centers"),
            orientation1=load_array("orientation1"),
            orientation2=load_array("orientation2"),
            sparse_orientation=sparse_orientation,
            aabb_pruning=bool(metadata.get("aabb_pruning", True)),
            storage="numpy",
            order=str(metadata.get("order", "ix + iy*nx + iz*nx*ny")),
        )

        if output == "torch":
            return obj.to_torch(device=device, copy=False)
        return obj

    def to_dlpack(self, field: str = "yarn_id"):
        """Export one voxel data field as a legacy DLPack capsule.

        New simulation integrations should prefer
        ``SimulationSample.array(name)`` and pass the returned NumPy array or
        Torch tensor directly to a consumer such as ``torch.from_dlpack``.
        This method remains available for callers that require a
        single-consumption capsule.

        Parameters
        ----------
        field : {"yarn_id", "material_id", "occupancy"}
            Field to expose. ``material_id`` uses the default
            :meth:`material_id` mapping and ``occupancy`` exports the boolean
            yarn mask.

        Returns
        -------
        PyCapsule
            DLPack capsule consumable by torch, CuPy, JAX, and other tensor
            libraries that support the DLPack Python protocol.
        """
        if torch is None:
            raise ImportError(
                "DLPack export requires PyTorch. Install with "
                '`pip install "pytexgen[gpu]"` or convert through numpy.'
            )

        field = field.lower()
        if field == "yarn_id":
            value = self.yarn_id
        elif field == "material_id":
            value = self.material_id()
        elif field == "occupancy":
            value = self.occupancy()
        else:
            raise ValueError(
                'field must be one of "yarn_id", "material_id", or "occupancy"'
            )

        tensor = value if _is_torch_tensor(value) else torch.as_tensor(value)
        return torch.utils.dlpack.to_dlpack(tensor)


@dataclass
class VoxelizationCache:
    """Cached TexGen geometry snapshots for repeated voxelization.

    Use this when the same textile is voxelized at multiple resolutions or sent
    to multiple array backends. It avoids repeating the SWIG/C++ object walk in
    :func:`extract_snapshots`.
    """
    snapshots: List["YarnSnapshot"]
    aabb: np.ndarray

    @classmethod
    def from_textile(cls, textile: CTextile) -> "VoxelizationCache":
        """Extract and cache yarn snapshots from a built textile."""
        bundle = extract_snapshot_bundle(textile)
        snapshots, aabb = bundle.to_snapshots(), bundle.aabb
        if len(snapshots) == 0:
            raise RuntimeError("No yarns extracted - textile may be empty or unbuilt")
        return cls(snapshots=snapshots, aabb=aabb)

    @classmethod
    def from_bundle(cls, bundle: SnapshotBundle) -> "VoxelizationCache":
        """Create a cache from a structure-of-arrays snapshot bundle."""
        bundle = _coerce_snapshot_bundle(bundle)
        return cls(snapshots=bundle.to_snapshots(), aabb=bundle.aabb)

    def voxelize(self, nx: int = 64, ny: int = 64, nz: int = 64,
                 **kwargs) -> VoxelGridData:
        """Voxelize cached snapshots without re-reading the TexGen textile."""
        return voxelize_snapshots_data(
            self.snapshots, self.aabb, nx=nx, ny=ny, nz=nz, **kwargs
        )


def _is_torch_tensor(value) -> bool:
    """Return true when ``value`` behaves like a torch tensor."""
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "device")


def _array_to_numpy(value, copy: bool = False):
    """Convert numpy-like or torch tensor data to a numpy array."""
    if _is_torch_tensor(value):
        array = value.detach().cpu().numpy()
        return array.copy() if copy else array
    return np.array(value, copy=copy)


def _array_to_torch(value, torch_mod, device: Optional[str] = None, copy: bool = False):
    """Convert numpy-like or torch tensor data to a torch tensor."""
    if _is_torch_tensor(value):
        tensor = value.to(device=device) if device is not None else value
    else:
        tensor = torch_mod.as_tensor(value)
        if device is not None:
            tensor = tensor.to(device=device)
    return tensor.clone() if copy else tensor


def _resolve_numpy_array_dtype(dtype):
    """Resolve a dtype accepted by ``VoxelGridData.to('numpy', dtype=...)``."""
    if dtype is None:
        return None
    if hasattr(dtype, "is_floating_point"):
        name = str(dtype).replace("torch.", "")
        if name == "bfloat16":
            raise ValueError("numpy output does not support torch.bfloat16 dtype")
        dtype = name
    np_dtype = np.dtype(dtype)
    if not np.issubdtype(np_dtype, np.floating):
        raise ValueError(f"numpy conversion dtype must be floating, got {np_dtype}")
    return np_dtype


def _resolve_torch_array_dtype(torch_mod, dtype):
    """Resolve a dtype accepted by ``VoxelGridData.to('torch', dtype=...)``."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        dtype = dtype.replace("torch.", "")
        try:
            return {
                "float16": torch_mod.float16,
                "float32": torch_mod.float32,
                "float64": torch_mod.float64,
                "bfloat16": torch_mod.bfloat16,
            }[dtype.lower()]
        except KeyError:
            raise ValueError(
                'torch dtype string must be "float16", "float32", '
                '"float64", or "bfloat16"'
            )
    if hasattr(dtype, "is_floating_point"):
        if not dtype.is_floating_point:
            raise ValueError(f"torch conversion dtype must be floating, got {dtype}")
        return dtype
    # Accept numpy dtype-like objects for convenience.
    np_dtype = np.dtype(dtype)
    if not np.issubdtype(np_dtype, np.floating):
        raise ValueError(f"torch conversion dtype must be floating, got {np_dtype}")
    return {
        np.dtype("float16"): torch_mod.float16,
        np.dtype("float32"): torch_mod.float32,
        np.dtype("float64"): torch_mod.float64,
    }.get(np_dtype) or (_raise_unsupported_torch_dtype(np_dtype))


def _raise_unsupported_torch_dtype(np_dtype):
    """Raise a consistent error for unsupported torch conversion dtypes."""
    raise ValueError(f"unsupported torch floating dtype: {np_dtype}")


def _torch_dtype_name(dtype) -> str:
    """Return a compact dtype metadata name from a torch dtype."""
    return str(dtype).replace("torch.", "")


def _coerce_voxel_grid_output(data: VoxelGridData,
                              output: str,
                              device: Optional[str]) -> VoxelGridData:
    """Convert a ``VoxelGridData`` object to the requested storage backend."""
    if output == "backend":
        return data
    if output == "numpy":
        return data.to("numpy", copy=False)
    if output == "torch":
        return data.to("torch", device=device, copy=False)
    raise ValueError('output must be one of "backend", "numpy", or "torch"')


def _xyz(v) -> np.ndarray:
    """Convert a TexGen 3D point/vector object to a float64 array.

    Parameters
    ----------
    v : object
        SWIG object exposing ``x``, ``y`` and ``z`` attributes.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3,)`` with dtype ``float64``.
    """
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _xy(v) -> np.ndarray:
    """Convert a TexGen 2D point/vector object to a float64 array.

    Parameters
    ----------
    v : object
        SWIG object exposing ``x`` and ``y`` attributes.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2,)`` with dtype ``float64``.
    """
    return np.array([v.x, v.y], dtype=np.float64)


def _extract_yarn(yarn: CYarn, translations_xyz) -> Optional[YarnSnapshot]:
    """Extract one TexGen yarn into array-friendly geometry.

    Parameters
    ----------
    yarn : CYarn
        TexGen yarn object. The function asks TexGen to build line, surface and
        volume data when the SWIG build exposes ``BuildYarnIfNeeded``.
    translations_xyz : array-like of shape ``(K, 3)``
        Periodic image translations for this yarn. The origin translation
        should be included by the caller when no repeats are present.

    Returns
    -------
    YarnSnapshot or None
        Snapshot containing slave-node frames, section polygon and periodic
        translations. ``None`` is returned for degenerate yarns with fewer than
        two slave nodes.
    """
    # Some SWIG builds do not expose BuildYarnIfNeeded; GetSlaveNodes and
    # section access still trigger/return built geometry for normal textiles.
    build_if_needed = getattr(yarn, "BuildYarnIfNeeded", None)
    if build_if_needed is not None:
        build_if_needed(_LINE | _SURFACE | _VOLUME)

    slaves = yarn.GetSlaveNodes(_SURFACE)
    # SWIG-wrapped std::vector supports both __len__ and .size().
    M = len(slaves) if hasattr(slaves, "__len__") else slaves.size()
    if M < 2:
        return None

    positions = np.empty((M, 3), dtype=np.float64)
    tangents  = np.empty((M, 3), dtype=np.float64)
    ups       = np.empty((M, 3), dtype=np.float64)
    sides     = np.empty((M, 3), dtype=np.float64)

    # Section polygon: sampled once from the first slave node.
    # CSlaveNode exposes the 2D section points via GetSectionPoints().
    sec_pts = None

    for i in range(M):
        node = slaves[i]
        positions[i] = _xyz(node.GetPosition())
        t = _xyz(node.GetTangent())
        u = _xyz(node.GetUp())
        s = np.cross(t, u)  # right-handed (side, up, tangent) frame
        # Renormalize in case TexGen's projection left residual non-unit lengths.
        t /= max(np.linalg.norm(t), 1e-12)
        u /= max(np.linalg.norm(u), 1e-12)
        s /= max(np.linalg.norm(s), 1e-12)
        tangents[i] = t
        ups[i] = u
        sides[i] = s

        if sec_pts is None:
            try:
                pts2d = node.Get2DSectionPoints()
                if len(pts2d) >= 3:
                    sec_pts = np.array([[p.x, p.y] for p in pts2d], dtype=np.float64)
            except Exception:
                sec_pts = None

    if sec_pts is None or len(sec_pts) < 3:
        # Fallback: ask the yarn-section object at parameter 0.
        ys = yarn.GetYarnSection()
        section_obj = ys.GetSection(0.0)
        pts = section_obj.GetPoints(40, True)
        sec_pts = np.array([[p.x, p.y] for p in pts], dtype=np.float64)

    # Ensure polygon is closed & has consistent orientation (CCW).
    if not np.allclose(sec_pts[0], sec_pts[-1]):
        sec_pts = np.vstack([sec_pts, sec_pts[:1]])
    # Shoelace -> if negative, reverse.
    x, y = sec_pts[:, 0], sec_pts[:, 1]
    area2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if area2 < 0:
        sec_pts = sec_pts[::-1].copy()

    return YarnSnapshot(
        positions=positions, tangents=tangents, ups=ups, sides=sides,
        section=sec_pts, translations=np.asarray(translations_xyz, dtype=np.float64),
    )


def extract_snapshots(textile: CTextile) -> Tuple[List[YarnSnapshot], np.ndarray]:
    """Snapshot all yarns and the textile domain bounding box.

    Parameters
    ----------
    textile : CTextile
        Built TexGen textile object with an assigned domain.

    Returns
    -------
    snapshots : list of YarnSnapshot
        One array snapshot per non-degenerate yarn.
    aabb : numpy.ndarray
        Domain axis-aligned bounding box with shape ``(2, 3)``. ``aabb[0]`` is
        the lower corner and ``aabb[1]`` is the upper corner.

    Notes
    -----
    Domain translations are pulled from TexGen for each yarn so periodic images
    are tested by the numpy/torch classifiers.
    """
    domain = textile.GetDomain()
    # Domain translations are per-yarn (periodic images).
    snapshots: List[YarnSnapshot] = []
    for i in range(textile.GetNumYarns()):
        yarn = textile.GetYarn(i)
        try:
            trans = domain.GetTranslations(yarn)
            trans_np = np.array([[t.x, t.y, t.z] for t in trans], dtype=np.float64)
            if len(trans_np) == 0:
                trans_np = np.zeros((1, 3), dtype=np.float64)
        except Exception:
            trans_np = np.zeros((1, 3), dtype=np.float64)
        snap = _extract_yarn(yarn, trans_np)
        if snap is not None:
            snapshots.append(snap)

    # Domain AABB: ask the domain mesh.
    dmesh = domain.GetMesh()
    nodes = dmesh.GetNodes()
    pts = np.array([[n.x, n.y, n.z] for n in nodes], dtype=np.float64)
    if len(pts) == 0:
        # Fallback: bounding box of all slave nodes.
        all_pos = np.vstack([s.positions for s in snapshots])
        lo, hi = all_pos.min(0), all_pos.max(0)
    else:
        lo, hi = pts.min(0), pts.max(0)
    aabb = np.stack([lo, hi])  # (2, 3)
    return snapshots, aabb


def _coerce_snapshot_bundle(value: Any) -> SnapshotBundle:
    """Return ``value`` as a ``SnapshotBundle``.

    Fast C++ providers may return the dataclass directly or a mapping of arrays
    with the same field names. Keeping the coercion here gives external
    providers a small, stable contract.
    """
    if isinstance(value, SnapshotBundle):
        return value
    if isinstance(value, MappingABC):
        required = (
            "positions", "tangents", "ups", "sides", "node_offsets",
            "sections", "section_offsets", "translations",
            "translation_offsets", "aabb",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(
                "snapshot bundle mapping is missing required field(s): "
                + ", ".join(missing)
            )
        return SnapshotBundle(
            positions=np.asarray(value["positions"]),
            tangents=np.asarray(value["tangents"]),
            ups=np.asarray(value["ups"]),
            sides=np.asarray(value["sides"]),
            node_offsets=np.asarray(value["node_offsets"], dtype=np.int64),
            sections=np.asarray(value["sections"]),
            section_offsets=np.asarray(value["section_offsets"], dtype=np.int64),
            translations=np.asarray(value["translations"]),
            translation_offsets=np.asarray(value["translation_offsets"], dtype=np.int64),
            aabb=np.asarray(value["aabb"], dtype=np.float64),
        )
    raise TypeError(
        "fastdata provider must return SnapshotBundle or a mapping of bundle arrays"
    )


def _fastdata_provider_candidates() -> List[str]:
    """Return module names to probe for the optional compiled provider."""
    candidates = []
    package = __package__
    if package:
        candidates.append(f"{package}._fastdata")
    candidates.append("TexGen._fastdata")
    return list(dict.fromkeys(candidates))


def _load_fastdata_provider():
    """Return an optional compiled fastdata provider module, if installed."""
    for module_name in _fastdata_provider_candidates():
        try:
            provider = importlib.import_module(module_name)
        except (ImportError, OSError):
            continue
        if hasattr(provider, "extract_snapshot_bundle"):
            return provider
    return None


def _fastdata_provider_info(provider: Any) -> Dict[str, Any]:
    """Return optional provider metadata in a normalized dictionary."""
    info_func = getattr(provider, "provider_info", None)
    if info_func is None:
        return {"capabilities": []}
    info = info_func()
    if not isinstance(info, MappingABC):
        raise TypeError("fastdata provider_info() must return a mapping")
    normalized = dict(info)
    capabilities = normalized.get("capabilities", [])
    normalized["capabilities"] = list(capabilities)
    return normalized


def fastdata_provider_status() -> Dict[str, Any]:
    """Report whether the optional compiled snapshot provider is available.

    This is intentionally side-effect free apart from normal Python imports; it
    gives applications and benchmarks a clear signal that the high-throughput
    Python/C handoff is active.
    """
    checked = []
    errors = {}
    for module_name in _fastdata_provider_candidates():
        checked.append(module_name)
        try:
            provider = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            errors[module_name] = str(exc)
            continue
        if not hasattr(provider, "extract_snapshot_bundle"):
            errors[module_name] = "missing extract_snapshot_bundle"
            continue
        provider_info = _fastdata_provider_info(provider)
        return {
            "available": True,
            "module": module_name,
            "checked": checked,
            "error": None,
            "errors": errors,
            "capabilities": provider_info.get("capabilities", []),
            "provider_info": provider_info,
        }

    return {
        "available": False,
        "module": None,
        "checked": checked,
        "error": "no _fastdata provider with extract_snapshot_bundle found",
        "errors": errors,
        "capabilities": [],
        "provider_info": {},
    }


def extract_snapshot_bundle(textile: CTextile,
                            provider: Optional[Any] = None) -> SnapshotBundle:
    """Extract textile geometry as a structure-of-arrays bundle.

    A provider module can be passed explicitly for tests or supplied by an
    installed ``_fastdata`` extension. Without one, the function falls back to
    the existing SWIG-based :func:`extract_snapshots` path.
    """
    provider = _load_fastdata_provider() if provider is None else provider
    if provider is not None:
        extractor = getattr(provider, "extract_snapshot_bundle", None)
        if extractor is None:
            raise TypeError("fastdata provider lacks extract_snapshot_bundle")
        return _coerce_snapshot_bundle(extractor(textile))

    snapshots, aabb = extract_snapshots(textile)
    return SnapshotBundle.from_snapshots(snapshots, aabb)


# ---------------------------------------------------------------------------
# Classification: for every voxel center, find owning yarn.
# ---------------------------------------------------------------------------


def _pack_yarns(snapshots: List[YarnSnapshot], device, dtype):
    """Pack yarn snapshots into padded torch tensors.

    Parameters
    ----------
    snapshots : list of YarnSnapshot
        Yarn snapshots returned by :func:`extract_snapshots`.
    device : str or torch.device
        Torch device used for the packed tensors, for example ``"cuda"`` or
        ``"cpu"``.
    dtype : torch.dtype
        Floating point dtype used for positions, frames and polygons.

    Returns
    -------
    dict
        Padded tensor bundle. Keys ``P``, ``T``, ``U`` and ``S`` store
        positions, tangents, up vectors and side vectors with shape
        ``(num_yarns, max_nodes, 3)``. ``Sec`` stores section polygons,
        ``Tr`` stores translations, and ``BoundsLo``/``BoundsHi`` store
        per-translation pruning boxes.
    """
    torch_mod = _require_torch()
    num_yarns = len(snapshots)
    M_max = max(s.positions.shape[0] for s in snapshots)
    N_max = max(s.section.shape[0] for s in snapshots)
    K_max = max(s.translations.shape[0] for s in snapshots)

    P = torch_mod.zeros((num_yarns, M_max, 3), device=device, dtype=dtype)
    T = torch_mod.zeros_like(P)
    U = torch_mod.zeros_like(P)
    S = torch_mod.zeros_like(P)
    M_len = torch_mod.zeros(num_yarns, device=device, dtype=torch_mod.int32)
    Sec = torch_mod.zeros((num_yarns, N_max, 2), device=device, dtype=dtype)
    N_len = torch_mod.zeros(num_yarns, device=device, dtype=torch_mod.int32)
    Tr = torch_mod.zeros((num_yarns, K_max, 3), device=device, dtype=dtype)
    K_len = torch_mod.zeros(num_yarns, device=device, dtype=torch_mod.int32)
    BoundsLo = torch_mod.zeros((num_yarns, K_max, 3), device=device, dtype=dtype)
    BoundsHi = torch_mod.zeros((num_yarns, K_max, 3), device=device, dtype=dtype)

    for i, s in enumerate(snapshots):
        m = s.positions.shape[0]
        P[i, :m] = torch_mod.from_numpy(s.positions).to(device=device, dtype=dtype)
        T[i, :m] = torch_mod.from_numpy(s.tangents).to(device=device, dtype=dtype)
        U[i, :m] = torch_mod.from_numpy(s.ups).to(device=device, dtype=dtype)
        S[i, :m] = torch_mod.from_numpy(s.sides).to(device=device, dtype=dtype)
        M_len[i] = m
        n = s.section.shape[0]
        Sec[i, :n] = torch_mod.from_numpy(s.section).to(device=device, dtype=dtype)
        N_len[i] = n
        k = s.translations.shape[0]
        Tr[i, :k] = torch_mod.from_numpy(s.translations).to(device=device, dtype=dtype)
        K_len[i] = k
        bounds_lo, bounds_hi = _snapshot_translation_bounds(s)
        BoundsLo[i, :k] = torch_mod.from_numpy(bounds_lo).to(device=device, dtype=dtype)
        BoundsHi[i, :k] = torch_mod.from_numpy(bounds_hi).to(device=device, dtype=dtype)

    return dict(
        P=P, T=T, U=U, S=S, M=M_len, Sec=Sec, N=N_len, Tr=Tr, K=K_len,
        BoundsLo=BoundsLo, BoundsHi=BoundsHi,
    )


def _point_in_polygon_batch(points_uv: torch.Tensor,
                            polygon: torch.Tensor,
                            poly_len: int) -> torch.Tensor:
    """Classify torch points against a 2D polygon with ray casting.

    Parameters
    ----------
    points_uv : torch.Tensor
        Query points in local section coordinates. Shape is ``(..., 2)`` where
        the last axis is ``(u, v)``.
    polygon : torch.Tensor
        Padded polygon vertex array with shape ``(N_max, 2)``.
    poly_len : int
        Number of valid vertices in ``polygon``. The polygon is expected to be
        closed by repeating the first point.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``points_uv.shape[:-1]``. ``True`` means the
        query point is inside the polygon.
    """
    torch_mod = _require_torch()
    vertices = polygon[:poly_len]                      # (N, 2), closed
    poly = vertices[:-1]                               # valid polygon edges
    p_next = vertices[1:]
    # Broadcast: points (..., 1, 2)  vs edges (N-1, 2)
    u = points_uv[..., None, 0]                        # (..., 1)
    v = points_uv[..., None, 1]
    x1 = poly[:, 0]; y1 = poly[:, 1]
    x2 = p_next[:, 0]; y2 = p_next[:, 1]
    # Crossing test: edge straddles v and ray going +u crosses it.
    cond1 = (y1 > v) != (y2 > v)
    # x-intercept of edge at height v
    denom = (y2 - y1)
    denom = torch_mod.where(denom.abs() < 1e-12, torch_mod.full_like(denom, 1e-12), denom)
    xi = x1 + (v - y1) * (x2 - x1) / denom
    cond2 = u < xi
    hits = (cond1 & cond2).sum(dim=-1)                 # (...,)
    return (hits % 2) == 1


def _distance_to_polygon_edges(points_uv: torch.Tensor,
                               polygon: torch.Tensor,
                               poly_len: int) -> torch.Tensor:
    """Return the shortest local-plane distance to a polygon boundary."""
    torch_mod = _require_torch()
    vertices = polygon[:poly_len]
    edge_start = vertices[:-1]
    edge_delta = vertices[1:] - edge_start
    edge_length2 = edge_delta.square().sum(dim=1).clamp_min(
        torch_mod.finfo(points_uv.dtype).eps
    )
    point_dot_delta = points_uv @ edge_delta.transpose(0, 1)
    start_dot_delta = (edge_start * edge_delta).sum(dim=1)
    alpha = (
        point_dot_delta - start_dot_delta.unsqueeze(0)
    ) / edge_length2.unsqueeze(0)
    alpha = alpha.clamp(0.0, 1.0)
    point_dot_start = points_uv @ edge_start.transpose(0, 1)
    closest_length2 = (
        edge_start.square().sum(dim=1).unsqueeze(0)
        + 2.0 * alpha * start_dot_delta.unsqueeze(0)
        + alpha.square() * edge_length2.unsqueeze(0)
    )
    point_dot_closest = point_dot_start + alpha * point_dot_delta
    distance2 = (
        points_uv.square().sum(dim=1, keepdim=True)
        + closest_length2
        - 2.0 * point_dot_closest
    ).clamp_min_(0.0)
    return torch_mod.sqrt(distance2.amin(dim=1))


def _classify_voxels_torch(centers: torch.Tensor,
                           packed: dict,
                           chunk: int = 65536,
                           aabb_pruning: bool = True,
                           progress: Any = False,
                           include_orientations: bool = False,
                           orientation_storage: str = "dense") -> Any:
    """Classify voxel centers with the torch backend.

    Parameters
    ----------
    centers : torch.Tensor
        Voxel center coordinates with shape ``(V, 3)``.
    packed : dict
        Tensor bundle returned by :func:`_pack_yarns`.
    chunk : int, default=65536
        Number of voxel centers processed at once. Increase for faster large
        GPU runs when memory allows; decrease to reduce VRAM/RAM usage.
    aabb_pruning : bool, default=True
        Skip yarn/translation candidates whose conservative bounding boxes
        cannot contain the current chunk.
    progress : bool or callable, default=False
        Show a tqdm progress bar over voxel chunks when true.
    include_orientations : bool, default=False
        Capture the winning yarn tangent and up vector for every yarn voxel.
    orientation_storage : {"dense", "sparse"}, default="dense"
        Return full per-voxel direction arrays or compact arrays containing
        only yarn voxels.

    Returns
    -------
    torch.Tensor or tuple
        ``int32`` yarn indices, optionally followed by dense direction arrays
        or a compact sparse orientation payload.
    """
    torch_mod = _require_torch()
    device = centers.device
    V = centers.shape[0]
    yarn_id = torch_mod.full((V,), -1, device=device, dtype=torch_mod.int32)
    dense_orientation1 = None
    dense_orientation2 = None
    sparse_indices = []
    sparse_yarn_ids = []
    sparse_orientation1 = []
    sparse_orientation2 = []
    if include_orientations and orientation_storage == "dense":
        dense_orientation1 = torch_mod.zeros(
            (V, 3), device=device, dtype=centers.dtype
        )
        dense_orientation2 = torch_mod.zeros(
            (V, 3), device=device, dtype=centers.dtype
        )

    P, T, U, S = packed["P"], packed["T"], packed["U"], packed["S"]
    M_len = packed["M"]
    Sec, N_len = packed["Sec"], packed["N"]
    Tr, K_len = packed["Tr"], packed["K"]
    BoundsLo = packed.get("BoundsLo")
    BoundsHi = packed.get("BoundsHi")
    num_yarns = P.shape[0]

    # Process voxels in chunks to cap VRAM.
    chunk_starts = range(0, V, chunk)
    for v0 in _progress_iter(
        chunk_starts, progress, total=math.ceil(V / chunk),
        desc="classify torch voxels", unit="chunk"
    ):
        v1 = min(v0 + chunk, V)
        pts = centers[v0:v1]                           # (C, 3)
        C = pts.shape[0]
        chunk_lo = pts.amin(dim=0)
        chunk_hi = pts.amax(dim=0)
        best_dist = torch_mod.full((C,), float("inf"), device=device)
        best_yarn = torch_mod.full((C,), -1, device=device, dtype=torch_mod.int32)
        if include_orientations:
            best_orientation1 = torch_mod.zeros(
                (C, 3), device=device, dtype=centers.dtype
            )
            best_orientation2 = torch_mod.zeros(
                (C, 3), device=device, dtype=centers.dtype
            )

        for y_idx in range(num_yarns):
            m = int(M_len[y_idx].item())
            k = int(K_len[y_idx].item())
            n = int(N_len[y_idx].item())

            Py = P[y_idx, :m]                           # (M, 3)
            Ty = T[y_idx, :m]
            Uy = U[y_idx, :m]
            Sy = S[y_idx, :m]
            poly = Sec[y_idx]                           # (N_max, 2)

            for t_idx in range(k):
                offset = Tr[y_idx, t_idx]               # (3,)
                Pt = Py + offset                        # (M, 3)
                active_idx = None
                active_pts = pts
                if aabb_pruning and BoundsLo is not None and BoundsHi is not None:
                    lo = BoundsLo[y_idx, t_idx]
                    hi = BoundsHi[y_idx, t_idx]
                    if bool(((chunk_hi < lo) | (chunk_lo > hi)).any().item()):
                        continue
                    candidate = ((pts >= lo) & (pts <= hi)).all(dim=1)
                    if not bool(candidate.any().item()):
                        continue
                    active_idx = candidate.nonzero(as_tuple=False).flatten()
                    active_pts = pts[active_idx]

                # Project onto the closest slave-node segment. Interpolating
                # the centerline and frame avoids the finite slabs produced by
                # a nearest-node approximation, especially around crossovers.
                segment_start = Pt[:-1]                 # (M-1, 3)
                segment_delta = Pt[1:] - segment_start
                segment_length2 = segment_delta.square().sum(dim=1).clamp_min_(
                    torch_mod.finfo(centers.dtype).eps
                )
                point_dot_delta = (
                    active_pts @ segment_delta.transpose(0, 1)
                )
                start_dot_delta = (
                    segment_start * segment_delta
                ).sum(dim=1)
                raw_alpha = (
                    point_dot_delta - start_dot_delta.unsqueeze(0)
                ) / segment_length2.unsqueeze(0)
                alpha = raw_alpha.clamp(0.0, 1.0)
                point_dot_start = (
                    active_pts @ segment_start.transpose(0, 1)
                )
                start_length2 = segment_start.square().sum(dim=1)
                closest_length2 = (
                    start_length2.unsqueeze(0)
                    + 2.0 * alpha * start_dot_delta.unsqueeze(0)
                    + alpha.square() * segment_length2.unsqueeze(0)
                )
                point_dot_closest = (
                    point_dot_start + alpha * point_dot_delta
                )
                d2 = (
                    active_pts.square().sum(dim=1, keepdim=True)
                    + closest_length2
                    - 2.0 * point_dot_closest
                ).clamp_min_(0.0)
                nn = d2.argmin(dim=1)
                selected_alpha = alpha.gather(
                    1, nn[:, None]
                ).squeeze(1)
                selected_raw_alpha = raw_alpha.gather(
                    1, nn[:, None]
                ).squeeze(1)
                centerline = (
                    segment_start[nn]
                    + selected_alpha[:, None] * segment_delta[nn]
                )

                rel = active_pts - centerline
                tan = (
                    (1.0 - selected_alpha)[:, None] * Ty[nn]
                    + selected_alpha[:, None] * Ty[nn + 1]
                )
                tan = tan / tan.norm(dim=1, keepdim=True).clamp_min(
                    torch_mod.finfo(centers.dtype).eps
                )
                up = (
                    (1.0 - selected_alpha)[:, None] * Uy[nn]
                    + selected_alpha[:, None] * Uy[nn + 1]
                )
                up = up - (up * tan).sum(dim=1, keepdim=True) * tan
                up = up / up.norm(dim=1, keepdim=True).clamp_min(
                    torch_mod.finfo(centers.dtype).eps
                )
                sid = torch_mod.linalg.cross(tan, up, dim=1)
                u_coord = (rel * sid).sum(-1)           # (C,)
                v_coord = (rel * up ).sum(-1)
                t_coord = (rel * tan).sum(-1)

                # Point-in-polygon in (u, v) plane.
                uv = torch_mod.stack([u_coord, v_coord], dim=-1)  # (C, 2)
                inside = _point_in_polygon_batch(uv, poly, n)
                inside = inside & ~(
                    ((nn == 0) & (selected_raw_alpha < 0.0))
                    | (
                        (nn == m - 2)
                        & (selected_raw_alpha > 1.0)
                    )
                )

                # TexGen resolves overlapping yarns with the most negative
                # surface distance. For interior points, the negative shortest
                # distance to the section polygon is the equivalent local
                # quantity and avoids centerline-distance bias for flat yarns.
                dist = -_distance_to_polygon_edges(uv, poly, n)

                if active_idx is None:
                    upd = inside & (dist < best_dist)
                    best_dist = torch_mod.where(upd, dist, best_dist)
                    best_yarn = torch_mod.where(upd, torch_mod.full_like(best_yarn, y_idx), best_yarn)
                    if include_orientations:
                        best_orientation1 = torch_mod.where(
                            upd[:, None], tan, best_orientation1
                        )
                        best_orientation2 = torch_mod.where(
                            upd[:, None], up, best_orientation2
                        )
                else:
                    upd = inside & (dist < best_dist[active_idx])
                    if bool(upd.any().item()):
                        target = active_idx[upd]
                        best_dist[target] = dist[upd]
                        best_yarn[target] = y_idx
                        if include_orientations:
                            best_orientation1[target] = tan[upd]
                            best_orientation2[target] = up[upd]

        yarn_id[v0:v1] = best_yarn
        if include_orientations and orientation_storage == "dense":
            dense_orientation1[v0:v1] = best_orientation1
            dense_orientation2[v0:v1] = best_orientation2
        elif include_orientations:
            yarn_mask = best_yarn >= 0
            if bool(yarn_mask.any().item()):
                sparse_indices.append(
                    torch_mod.arange(
                        v0, v1, device=device, dtype=torch_mod.int64
                    )[yarn_mask]
                )
                sparse_yarn_ids.append(best_yarn[yarn_mask])
                sparse_orientation1.append(best_orientation1[yarn_mask])
                sparse_orientation2.append(best_orientation2[yarn_mask])

    if not include_orientations:
        return yarn_id
    if orientation_storage == "dense":
        return yarn_id, dense_orientation1, dense_orientation2

    if sparse_indices:
        payload = (
            torch_mod.cat(sparse_indices),
            torch_mod.cat(sparse_yarn_ids),
            torch_mod.cat(sparse_orientation1),
            torch_mod.cat(sparse_orientation2),
        )
    else:
        payload = (
            torch_mod.empty((0,), device=device, dtype=torch_mod.int64),
            torch_mod.empty((0,), device=device, dtype=torch_mod.int32),
            torch_mod.empty((0, 3), device=device, dtype=centers.dtype),
            torch_mod.empty((0, 3), device=device, dtype=centers.dtype),
        )
    return yarn_id, payload


def _snapshots_as_dtype(snapshots: List[YarnSnapshot], dtype) -> List[YarnSnapshot]:
    """Cast snapshot arrays once for a selected backend dtype.

    Parameters
    ----------
    snapshots : list of YarnSnapshot
        Original float64 geometry snapshots.
    dtype : numpy dtype
        Target dtype, usually ``np.float32`` or ``np.float64``.

    Returns
    -------
    list of YarnSnapshot
        New snapshot objects whose arrays use ``dtype`` where possible. NumPy
        may reuse the original arrays when the dtype already matches.
    """
    return [
        YarnSnapshot(
            positions=s.positions.astype(dtype, copy=False),
            tangents=s.tangents.astype(dtype, copy=False),
            ups=s.ups.astype(dtype, copy=False),
            sides=s.sides.astype(dtype, copy=False),
            section=s.section.astype(dtype, copy=False),
            translations=s.translations.astype(dtype, copy=False),
        )
        for s in snapshots
    ]


def _bundle_as_dtype(bundle: SnapshotBundle, dtype) -> SnapshotBundle:
    """Cast flat snapshot bundle arrays once for a selected numpy dtype."""
    return SnapshotBundle(
        positions=bundle.positions.astype(dtype, copy=False),
        tangents=bundle.tangents.astype(dtype, copy=False),
        ups=bundle.ups.astype(dtype, copy=False),
        sides=bundle.sides.astype(dtype, copy=False),
        node_offsets=bundle.node_offsets,
        sections=bundle.sections.astype(dtype, copy=False),
        section_offsets=bundle.section_offsets,
        translations=bundle.translations.astype(dtype, copy=False),
        translation_offsets=bundle.translation_offsets,
        aabb=bundle.aabb,
    )


def _snapshot_search_radius(snap: YarnSnapshot) -> float:
    """Estimate a conservative search radius for one yarn snapshot.

    Parameters
    ----------
    snap : YarnSnapshot
        Yarn geometry snapshot.

    Returns
    -------
    float
        Radius used to inflate the slave-node position bounding box. It covers
        the section polygon radius plus half the longest slave-node segment.
    """
    section_radius = float(np.sqrt(np.max(np.einsum("ij,ij->i", snap.section, snap.section))))
    if snap.positions.shape[0] > 1:
        segment_lengths = np.linalg.norm(np.diff(snap.positions, axis=0), axis=1)
        segment_margin = float(segment_lengths.max(initial=0.0)) * 0.5
    else:
        segment_margin = 0.0
    return section_radius + segment_margin + 1e-6


def _snapshot_translation_bounds(snap: YarnSnapshot) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-translation AABBs for fast candidate pruning.

    Parameters
    ----------
    snap : YarnSnapshot
        Yarn geometry snapshot containing positions and periodic translations.

    Returns
    -------
    bounds_lo, bounds_hi : tuple of numpy.ndarray
        Lower and upper corners with shape ``(K, 3)``, where ``K`` is the number
        of translations in ``snap.translations``.
    """
    radius = _snapshot_search_radius(snap)
    base_lo = snap.positions.min(axis=0) - radius
    base_hi = snap.positions.max(axis=0) + radius
    translations = np.asarray(snap.translations, dtype=snap.positions.dtype)
    return base_lo[None, :] + translations, base_hi[None, :] + translations


def _bundle_yarn_slices(bundle: SnapshotBundle, index: int) -> Tuple[slice, slice, slice]:
    """Return node, section, and translation slices for one flat bundle yarn."""
    n0, n1 = bundle.node_offsets[index:index + 2]
    s0, s1 = bundle.section_offsets[index:index + 2]
    t0, t1 = bundle.translation_offsets[index:index + 2]
    return slice(int(n0), int(n1)), slice(int(s0), int(s1)), slice(int(t0), int(t1))


def _bundle_search_radius(bundle: SnapshotBundle, index: int) -> float:
    """Estimate a conservative search radius for one flat-bundle yarn."""
    node_slice, section_slice, _ = _bundle_yarn_slices(bundle, index)
    section = bundle.sections[section_slice]
    positions = bundle.positions[node_slice]
    section_radius = float(np.sqrt(np.max(np.einsum("ij,ij->i", section, section))))
    if positions.shape[0] > 1:
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        segment_margin = float(segment_lengths.max(initial=0.0)) * 0.5
    else:
        segment_margin = 0.0
    return section_radius + segment_margin + 1e-6


def _bundle_translation_bounds(bundle: SnapshotBundle) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build per-yarn, per-translation AABBs for a flat snapshot bundle."""
    bounds = []
    for index in range(bundle.num_yarns):
        node_slice, _, translation_slice = _bundle_yarn_slices(bundle, index)
        positions = bundle.positions[node_slice]
        translations = bundle.translations[translation_slice]
        radius = _bundle_search_radius(bundle, index)
        base_lo = positions.min(axis=0) - radius
        base_hi = positions.max(axis=0) + radius
        bounds.append((base_lo[None, :] + translations, base_hi[None, :] + translations))
    return bounds


def _point_in_polygon_batch_numpy(points_uv: np.ndarray,
                                  polygon: np.ndarray,
                                  poly_len: int) -> np.ndarray:
    """Classify numpy points against a 2D polygon with ray casting.

    Parameters
    ----------
    points_uv : numpy.ndarray
        Query points in local section coordinates, shape ``(N, 2)``.
    polygon : numpy.ndarray
        Polygon vertices with shape ``(M, 2)``. The polygon should be closed by
        repeating the first point at the end.
    poly_len : int
        Number of valid vertices from ``polygon`` to use.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(N,)``. ``True`` means the point is inside.
    """
    vertices = polygon[:poly_len]
    poly = vertices[:-1]
    p_next = vertices[1:]

    u = points_uv[:, None, 0]
    v = points_uv[:, None, 1]
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = p_next[:, 0]
    y2 = p_next[:, 1]

    cond1 = (y1 > v) != (y2 > v)
    denom = y2 - y1
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    xi = x1 + (v - y1) * (x2 - x1) / denom
    hits = (cond1 & (u < xi)).sum(axis=-1)
    return (hits % 2) == 1


def _classify_voxel_chunk_bundle_numpy(
    pts: np.ndarray,
    bundle: SnapshotBundle,
    bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    aabb_pruning: bool = True,
    include_orientations: bool = False,
) -> Any:
    """Classify a center chunk directly from flat snapshot bundle arrays."""
    C = pts.shape[0]
    best_dist = np.full(C, np.inf, dtype=pts.dtype)
    best_yarn = np.full(C, -1, dtype=np.int32)
    orientation1 = np.zeros((C, 3), dtype=pts.dtype) if include_orientations else None
    orientation2 = np.zeros((C, 3), dtype=pts.dtype) if include_orientations else None
    chunk_lo = pts.min(axis=0)
    chunk_hi = pts.max(axis=0)

    for y_idx in range(bundle.num_yarns):
        node_slice, section_slice, translation_slice = _bundle_yarn_slices(bundle, y_idx)
        Py = bundle.positions[node_slice]
        if Py.shape[0] < 2:
            continue
        Ty = bundle.tangents[node_slice]
        Uy = bundle.ups[node_slice]
        Sy = bundle.sides[node_slice]
        poly = bundle.sections[section_slice]
        translations = bundle.translations[translation_slice]
        n = poly.shape[0]
        bounds_lo = bounds_hi = None
        if aabb_pruning and bounds is not None:
            bounds_lo, bounds_hi = bounds[y_idx]

        for t_idx, offset in enumerate(translations):
            active_idx = None
            active_pts = pts
            if bounds_lo is not None and bounds_hi is not None:
                lo = bounds_lo[t_idx]
                hi = bounds_hi[t_idx]
                if np.any(chunk_hi < lo) or np.any(chunk_lo > hi):
                    continue
                mask = np.all((pts >= lo) & (pts <= hi), axis=1)
                if not np.any(mask):
                    continue
                active_idx = np.nonzero(mask)[0]
                active_pts = pts[active_idx]

            Pt = Py + offset
            local_count = active_pts.shape[0]
            d2 = (
                np.einsum("ij,ij->i", active_pts, active_pts)[:, None]
                + np.einsum("ij,ij->i", Pt, Pt)[None, :]
                - 2.0 * (active_pts @ Pt.T)
            )
            np.maximum(d2, 0.0, out=d2)
            nn = np.argmin(d2, axis=1)

            rel = active_pts - Pt[nn]
            tan = Ty[nn]
            up = Uy[nn]
            sid = Sy[nn]
            u_coord = np.einsum("cd,cd->c", rel, sid)
            v_coord = np.einsum("cd,cd->c", rel, up)
            t_coord = np.einsum("cd,cd->c", rel, tan)

            uv = np.stack([u_coord, v_coord], axis=-1)
            inside = _point_in_polygon_batch_numpy(uv, poly, n)

            nearest_d2 = d2[np.arange(local_count), nn]
            dist = np.sqrt(nearest_d2) + np.abs(t_coord) * 0.1
            if active_idx is None:
                update = inside & (dist < best_dist)
                best_dist[update] = dist[update]
                best_yarn[update] = y_idx
                if include_orientations:
                    orientation1[update] = tan[update]
                    orientation2[update] = up[update]
            else:
                update = inside & (dist < best_dist[active_idx])
                target = active_idx[update]
                best_dist[target] = dist[update]
                best_yarn[target] = y_idx
                if include_orientations:
                    orientation1[target] = tan[update]
                    orientation2[target] = up[update]

    if include_orientations:
        return best_yarn, orientation1, orientation2
    return best_yarn


def _classify_voxel_chunk_numpy(pts: np.ndarray,
                                snapshots: List[YarnSnapshot],
                                bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                                aabb_pruning: bool = True,
                                include_orientations: bool = False) -> Any:
    """Classify a contiguous numpy voxel-center chunk.

    Parameters
    ----------
    pts : numpy.ndarray
        Voxel center coordinates for one chunk, shape ``(C, 3)``.
    snapshots : list of YarnSnapshot
        Yarn snapshots to test against.
    bounds : list of tuple of numpy.ndarray, optional
        Per-yarn bounds returned by :func:`_snapshot_translation_bounds`. Pass
        ``None`` when AABB pruning is disabled.
    aabb_pruning : bool, default=True
        Whether to use the precomputed AABBs before running geometric tests.

    Returns
    -------
    numpy.ndarray
        ``int32`` yarn index for each point in ``pts``. ``-1`` is
        matrix/background.
    """
    C = pts.shape[0]
    best_dist = np.full(C, np.inf, dtype=pts.dtype)
    best_yarn = np.full(C, -1, dtype=np.int32)
    orientation1 = np.zeros((C, 3), dtype=pts.dtype) if include_orientations else None
    orientation2 = np.zeros((C, 3), dtype=pts.dtype) if include_orientations else None
    chunk_lo = pts.min(axis=0)
    chunk_hi = pts.max(axis=0)

    for y_idx, snap in enumerate(snapshots):
        Py = snap.positions
        Ty = snap.tangents
        Uy = snap.ups
        Sy = snap.sides
        poly = snap.section
        n = snap.section.shape[0]
        bounds_lo = bounds_hi = None
        if aabb_pruning and bounds is not None:
            bounds_lo, bounds_hi = bounds[y_idx]

        for t_idx, offset in enumerate(snap.translations):
            active_idx = None
            active_pts = pts
            if bounds_lo is not None and bounds_hi is not None:
                lo = bounds_lo[t_idx]
                hi = bounds_hi[t_idx]
                if np.any(chunk_hi < lo) or np.any(chunk_lo > hi):
                    continue
                mask = np.all((pts >= lo) & (pts <= hi), axis=1)
                if not np.any(mask):
                    continue
                active_idx = np.nonzero(mask)[0]
                active_pts = pts[active_idx]

            Pt = Py + offset
            local_count = active_pts.shape[0]
            d2 = (
                np.einsum("ij,ij->i", active_pts, active_pts)[:, None]
                + np.einsum("ij,ij->i", Pt, Pt)[None, :]
                - 2.0 * (active_pts @ Pt.T)
            )
            np.maximum(d2, 0.0, out=d2)
            nn = np.argmin(d2, axis=1)

            rel = active_pts - Pt[nn]
            tan = Ty[nn]
            up = Uy[nn]
            sid = Sy[nn]
            u_coord = np.einsum("cd,cd->c", rel, sid)
            v_coord = np.einsum("cd,cd->c", rel, up)
            t_coord = np.einsum("cd,cd->c", rel, tan)

            uv = np.stack([u_coord, v_coord], axis=-1)
            inside = _point_in_polygon_batch_numpy(uv, poly, n)

            nearest_d2 = d2[np.arange(local_count), nn]
            dist = np.sqrt(nearest_d2) + np.abs(t_coord) * 0.1
            if active_idx is None:
                update = inside & (dist < best_dist)
                best_dist[update] = dist[update]
                best_yarn[update] = y_idx
                if include_orientations:
                    orientation1[update] = tan[update]
                    orientation2[update] = up[update]
            else:
                update = inside & (dist < best_dist[active_idx])
                target = active_idx[update]
                best_dist[target] = dist[update]
                best_yarn[target] = y_idx
                if include_orientations:
                    orientation1[target] = tan[update]
                    orientation2[target] = up[update]

    if include_orientations:
        return best_yarn, orientation1, orientation2
    return best_yarn


def _classify_voxels_bundle_numpy(
    centers: np.ndarray,
    bundle: SnapshotBundle,
    chunk: int = DEFAULT_NUMPY_CHUNK_VOXELS,
    workers: Optional[int] = None,
    aabb_pruning: bool = True,
    progress: Any = False,
    include_orientations: bool = False,
) -> Any:
    """Classify voxel centers directly from a flat ``SnapshotBundle``."""
    V = centers.shape[0]
    yarn_id = np.full(V, -1, dtype=np.int32)
    orientation1 = (
        np.zeros((V, 3), dtype=centers.dtype) if include_orientations else None
    )
    orientation2 = (
        np.zeros((V, 3), dtype=centers.dtype) if include_orientations else None
    )
    ranges = [(v0, min(v0 + chunk, V)) for v0 in range(0, V, chunk)]
    bounds_list = _bundle_translation_bounds(bundle) if aabb_pruning else None
    worker_count = _effective_numpy_workers(V, chunk, workers)

    def classify_range(range_bounds):
        v0, v1 = range_bounds
        result = _classify_voxel_chunk_bundle_numpy(
            centers[v0:v1],
            bundle,
            bounds=bounds_list,
            aabb_pruning=aabb_pruning,
            include_orientations=include_orientations,
        )
        if include_orientations:
            ids, ori1, ori2 = result
            return v0, v1, ids, ori1, ori2
        ids = result
        return v0, v1, ids

    if worker_count == 1:
        for range_bounds in _progress_iter(
            ranges, progress, total=len(ranges),
            desc="classify numpy bundle voxels", unit="chunk"
        ):
            result = classify_range(range_bounds)
            if include_orientations:
                v0, v1, ids, ori1, ori2 = result
                yarn_id[v0:v1] = ids
                orientation1[v0:v1] = ori1
                orientation2[v0:v1] = ori2
            else:
                v0, v1, ids = result
                yarn_id[v0:v1] = ids
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(classify_range, ranges)
            for result in _progress_iter(
                results, progress, total=len(ranges),
                desc="classify numpy bundle voxels", unit="chunk"
            ):
                if include_orientations:
                    v0, v1, ids, ori1, ori2 = result
                    yarn_id[v0:v1] = ids
                    orientation1[v0:v1] = ori1
                    orientation2[v0:v1] = ori2
                else:
                    v0, v1, ids = result
                    yarn_id[v0:v1] = ids
    if include_orientations:
        return yarn_id, orientation1, orientation2
    return yarn_id


def _default_numpy_workers() -> int:
    """Return the conservative default number of numpy worker threads.

    Returns
    -------
    int
        Between 1 and 4, capped to avoid oversubscription with BLAS, torch, or
        host applications embedding TexGen.
    """
    return max(1, min(os.cpu_count() or 1, 4))


def _effective_numpy_workers(num_items: int,
                             chunk: int,
                             workers: Optional[int]) -> int:
    """Return the actual numpy worker count after chunk-count clamping."""
    worker_count = _default_numpy_workers() if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be >= 1")
    task_count = max(1, math.ceil(num_items / chunk))
    return min(worker_count, task_count)


def _classify_voxels_numpy(centers: np.ndarray,
                           snapshots: List[YarnSnapshot],
                           chunk: int = DEFAULT_NUMPY_CHUNK_VOXELS,
                           workers: Optional[int] = None,
                           aabb_pruning: bool = True,
                           progress: Any = False,
                           include_orientations: bool = False) -> Any:
    """Classify all voxel centers with the numpy backend.

    Parameters
    ----------
    centers : numpy.ndarray
        Voxel center coordinates, shape ``(V, 3)``.
    snapshots : list of YarnSnapshot
        Yarn snapshots to test.
    chunk : int, default=8192
        Number of voxel centers processed per task.
    workers : int or None, default=None
        Number of Python worker threads. ``None`` uses
        :func:`_default_numpy_workers`.
    aabb_pruning : bool, default=True
        Skip yarn/translation candidates whose conservative bounding boxes
        cannot contain the current voxel chunk.
    progress : bool or callable, default=False
        Show a tqdm progress bar over voxel chunks when true.

    Returns
    -------
    numpy.ndarray
        ``int32`` yarn index for each center, shape ``(V,)``. ``-1`` is
        matrix/background.
    """
    V = centers.shape[0]
    yarn_id = np.full(V, -1, dtype=np.int32)
    orientation1 = (
        np.zeros((V, 3), dtype=centers.dtype) if include_orientations else None
    )
    orientation2 = (
        np.zeros((V, 3), dtype=centers.dtype) if include_orientations else None
    )
    ranges = [(v0, min(v0 + chunk, V)) for v0 in range(0, V, chunk)]
    bounds_list = [_snapshot_translation_bounds(s) for s in snapshots] if aabb_pruning else None

    worker_count = _effective_numpy_workers(V, chunk, workers)

    def classify_range(range_bounds):
        """Classify one ``(start, stop)`` center slice for executor.map."""
        v0, v1 = range_bounds
        result = _classify_voxel_chunk_numpy(
            centers[v0:v1],
            snapshots,
            bounds=bounds_list,
            aabb_pruning=aabb_pruning,
            include_orientations=include_orientations,
        )
        if include_orientations:
            ids, ori1, ori2 = result
            return v0, v1, ids, ori1, ori2
        ids = result
        return v0, v1, ids

    if worker_count == 1:
        for range_bounds in _progress_iter(
            ranges, progress, total=len(ranges),
            desc="classify numpy voxels", unit="chunk"
        ):
            result = classify_range(range_bounds)
            if include_orientations:
                v0, v1, ids, ori1, ori2 = result
                yarn_id[v0:v1] = ids
                orientation1[v0:v1] = ori1
                orientation2[v0:v1] = ori2
            else:
                v0, v1, ids = result
                yarn_id[v0:v1] = ids
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(classify_range, ranges)
            for result in _progress_iter(
                results, progress, total=len(ranges),
                desc="classify numpy voxels", unit="chunk"
            ):
                if include_orientations:
                    v0, v1, ids, ori1, ori2 = result
                    yarn_id[v0:v1] = ids
                    orientation1[v0:v1] = ori1
                    orientation2[v0:v1] = ori2
                else:
                    v0, v1, ids = result
                    yarn_id[v0:v1] = ids

    if include_orientations:
        return yarn_id, orientation1, orientation2
    return yarn_id


# ---------------------------------------------------------------------------
# Lightweight adaptive numpy mesh: linear-octree cells without p4est.
# ---------------------------------------------------------------------------


_CHILD_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

_HEX_NODE_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

_ADAPTIVE_SAMPLE_OFFSETS = np.vstack(
    [
        np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        _CHILD_OFFSETS,
    ]
)


def _structured_cell_lows_sizes(lo: np.ndarray,
                                hi: np.ndarray,
                                nx: int, ny: int, nz: int,
                                dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create base-grid adaptive cell arrays.

    Parameters
    ----------
    lo, hi : numpy.ndarray
        Domain lower and upper corners, each shape ``(3,)``.
    nx, ny, nz : int
        Base grid resolution.
    dtype : numpy dtype
        Floating point dtype for output coordinate arrays.

    Returns
    -------
    lows : numpy.ndarray
        Lower corner of each base cell, shape ``(nx*ny*nz, 3)`` in TexGen
        element order.
    sizes : numpy.ndarray
        Cell dimensions, same shape as ``lows``.
    levels : numpy.ndarray
        Refinement level per cell, shape ``(nx*ny*nz,)``. Base cells are level
        zero.
    """
    cell_size = ((hi - lo) / np.array([nx, ny, nz], dtype=np.float64)).astype(dtype)
    xs = np.asarray(lo[0], dtype=dtype) + np.arange(nx, dtype=dtype) * cell_size[0]
    ys = np.asarray(lo[1], dtype=dtype) + np.arange(ny, dtype=dtype) * cell_size[1]
    zs = np.asarray(lo[2], dtype=dtype) + np.arange(nz, dtype=dtype) * cell_size[2]

    lows_grid = np.empty((nz, ny, nx, 3), dtype=dtype)
    lows_grid[..., 0] = xs
    lows_grid[..., 1] = ys[None, :, None]
    lows_grid[..., 2] = zs[:, None, None]
    lows = lows_grid.reshape(-1, 3)

    sizes = np.empty_like(lows)
    sizes[:] = cell_size
    levels = np.zeros(lows.shape[0], dtype=np.int16)
    return lows, sizes, levels


def _subdivide_cells(lows: np.ndarray,
                     sizes: np.ndarray,
                     levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split parent hex cells into eight children each.

    Parameters
    ----------
    lows, sizes : numpy.ndarray
        Parent lower corners and dimensions, each shape ``(N, 3)``.
    levels : numpy.ndarray
        Parent refinement levels, shape ``(N,)``.

    Returns
    -------
    child_lows, child_sizes, child_levels : tuple of numpy.ndarray
        Child arrays with ``8*N`` rows. Child levels are parent level plus one.
    """
    offsets = _CHILD_OFFSETS.astype(lows.dtype, copy=False)
    half_sizes = sizes * np.asarray(0.5, dtype=sizes.dtype)
    child_lows = (
        lows[:, None, :] + half_sizes[:, None, :] * offsets[None, :, :]
    ).reshape(-1, 3)
    child_sizes = np.broadcast_to(
        half_sizes[:, None, :], (lows.shape[0], offsets.shape[0], 3)
    ).reshape(-1, 3).copy()
    child_levels = np.repeat(levels + 1, 8)
    return child_lows, child_sizes, child_levels


def _cell_sample_points(lows: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return center plus corner samples for adaptive refinement.

    Parameters
    ----------
    lows, sizes : numpy.ndarray
        Cell lower corners and dimensions, each shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Sample coordinates with shape ``(9*N, 3)``. For each cell, the first
        sample is the center and the remaining eight samples are corners.
    """
    offsets = _ADAPTIVE_SAMPLE_OFFSETS.astype(lows.dtype, copy=False)
    return (lows[:, None, :] + sizes[:, None, :] * offsets[None, :, :]).reshape(-1, 3)


def _refine_adaptive_cells(lows: np.ndarray,
                           sizes: np.ndarray,
                           levels: np.ndarray,
                           snapshots: List[YarnSnapshot],
                           adaptive_levels: int,
                           chunk_voxels: int,
                           workers: Optional[int],
                           max_adaptive_cells: int,
                           aabb_pruning: bool = True,
                           progress: Any = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine cells whose center and corner labels disagree.

    Parameters
    ----------
    lows, sizes, levels : numpy.ndarray
        Current leaf-cell arrays.
    snapshots : list of YarnSnapshot
        Yarn snapshots used to classify sample points.
    adaptive_levels : int
        Maximum number of refinement passes.
    chunk_voxels : int
        Maximum sample points classified at once.
    workers : int or None
        Numpy classifier worker count.
    max_adaptive_cells : int
        Safety cap on generated leaf-cell count.
    aabb_pruning : bool, default=True
        Whether to use yarn AABB pruning during sample classification.
    progress : bool or callable, default=False
        Show a tqdm progress bar over adaptive cell chunks when true.

    Returns
    -------
    lows, sizes, levels : tuple of numpy.ndarray
        Refined leaf-cell arrays.
    """
    sample_count = _ADAPTIVE_SAMPLE_OFFSETS.shape[0]
    cell_chunk = max(1, chunk_voxels // sample_count)

    for level in range(adaptive_levels):
        refine_parts = []
        cell_starts = range(0, lows.shape[0], cell_chunk)
        for c0 in _progress_iter(
            cell_starts, progress, total=math.ceil(lows.shape[0] / cell_chunk),
            desc=f"refine adaptive level {level + 1}/{adaptive_levels}",
            unit="chunk"
        ):
            c1 = min(c0 + cell_chunk, lows.shape[0])
            samples = _cell_sample_points(lows[c0:c1], sizes[c0:c1])
            sample_ids = _classify_voxels_numpy(
                samples, snapshots, chunk=chunk_voxels, workers=workers,
                aabb_pruning=aabb_pruning
            )
            labels = sample_ids.reshape((c1 - c0, sample_count))
            refine_parts.append(np.any(labels != labels[:, :1], axis=1))

        refine_mask = np.concatenate(refine_parts) if refine_parts else np.zeros(0, dtype=bool)
        refine_count = int(refine_mask.sum())
        if refine_count == 0:
            break

        next_count = lows.shape[0] + refine_count * 7
        if next_count > max_adaptive_cells:
            raise RuntimeError(
                f"Adaptive refinement would create {next_count:,} cells, "
                f"above max_adaptive_cells={max_adaptive_cells:,}"
            )

        keep_mask = ~refine_mask
        child_lows, child_sizes, child_levels = _subdivide_cells(
            lows[refine_mask], sizes[refine_mask], levels[refine_mask]
        )
        lows = np.concatenate([lows[keep_mask], child_lows], axis=0)
        sizes = np.concatenate([sizes[keep_mask], child_sizes], axis=0)
        levels = np.concatenate([levels[keep_mask], child_levels], axis=0)

    return lows, sizes, levels


def _classify_adaptive_cells_numpy(lows: np.ndarray,
                                   sizes: np.ndarray,
                                   snapshots: List[YarnSnapshot],
                                   chunk_voxels: int,
                                   workers: Optional[int],
                                   aabb_pruning: bool = True,
                                   progress: Any = False) -> np.ndarray:
    """Classify adaptive leaf cells by center point ownership.

    Parameters
    ----------
    lows, sizes : numpy.ndarray
        Leaf-cell lower corners and dimensions, each shape ``(N, 3)``.
    snapshots : list of YarnSnapshot
        Yarn snapshots to test.
    chunk_voxels : int
        Maximum centers classified at once.
    workers : int or None
        Numpy classifier worker count.
    aabb_pruning : bool, default=True
        Whether to use yarn AABB pruning.
    progress : bool or callable, default=False
        Show a tqdm progress bar over adaptive leaf-cell chunks when true.

    Returns
    -------
    numpy.ndarray
        ``int32`` yarn index per cell, shape ``(N,)``.
    """
    centers = lows + sizes * np.asarray(0.5, dtype=sizes.dtype)
    return _classify_voxels_numpy(
        centers, snapshots, chunk=chunk_voxels, workers=workers,
        aabb_pruning=aabb_pruning, progress=progress
    )


# ---------------------------------------------------------------------------
# Abaqus .inp writer (hex elements, per-element yarn index).
# ---------------------------------------------------------------------------


def _write_inp(path: Path, lo, hi, nx, ny, nz, yarn_id: np.ndarray,
               textile_name: str = "TexGenPython",
               progress: Any = False):
    """Write a structured Abaqus ``.inp`` hex mesh.

    Parameters
    ----------
    path : pathlib.Path
        Output file path. Parent directories must already exist.
    lo, hi : array-like of shape ``(3,)``
        Domain lower and upper corners.
    nx, ny, nz : int
        Structured grid resolution.
    yarn_id : numpy.ndarray
        ``int32`` yarn index per element, shape ``(nx*ny*nz,)``. ``-1`` is
        written as the ``Matrix`` element set.
    textile_name : str, default="TexGenPython"
        Name written to the Abaqus heading.
    progress : bool or callable, default=False
        Show tqdm progress bars for node, element, and element-set writing.
    """
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz

    nnx, nny, nnz = nx + 1, ny + 1, nz + 1

    def nid(ix, iy, iz):
        """Return the Abaqus node id for integer grid coordinates."""
        return 1 + ix + iy * nnx + iz * nnx * nny

    def flush_lines(file_obj, pending):
        """Write and clear buffered text lines."""
        if pending:
            file_obj.writelines(pending)
            pending.clear()

    with path.open("w", encoding="utf-8", newline="\n") as f:
        lines = []

        def emit(line: str):
            """Append one line to the local write buffer."""
            lines.append(line)
            if len(lines) >= 8192:
                flush_lines(f, lines)

        f.write("*Heading\n")
        f.write(f"TexGen Python voxel mesh: {textile_name}\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("**\n*Part, name=TexGenPart\n*Node\n")
        for iz in _progress_iter(
            range(nnz), progress, total=nnz, desc="write nodes", unit="z-slice"
        ):
            for iy in range(nny):
                for ix in range(nnx):
                    x = lo[0] + ix * dx
                    y = lo[1] + iy * dy
                    z = lo[2] + iz * dz
                    emit(f"{nid(ix,iy,iz)}, {x:.6g}, {y:.6g}, {z:.6g}\n")

        flush_lines(f, lines)
        f.write("*Element, type=C3D8R\n")
        eid = 0
        for iz in _progress_iter(
            range(nz), progress, total=nz, desc="write elements", unit="z-slice"
        ):
            for iy in range(ny):
                for ix in range(nx):
                    eid += 1
                    n1 = nid(ix,   iy,   iz)
                    n2 = nid(ix+1, iy,   iz)
                    n3 = nid(ix+1, iy+1, iz)
                    n4 = nid(ix,   iy+1, iz)
                    n5 = nid(ix,   iy,   iz+1)
                    n6 = nid(ix+1, iy,   iz+1)
                    n7 = nid(ix+1, iy+1, iz+1)
                    n8 = nid(ix,   iy+1, iz+1)
                    emit(f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")

        flush_lines(f, lines)
        # ELSETs per yarn (including -1 = matrix). Avoid storing Python int
        # lists for every element; scan the compact numpy yarn_id array instead.
        unique_yarns = np.unique(yarn_id)
        for yidx in _progress_iter(
            unique_yarns, progress, total=len(unique_yarns),
            desc="write element sets", unit="set"
        ):
            ids = np.nonzero(yarn_id == yidx)[0] + 1
            name = "Matrix" if yidx < 0 else f"Yarn{yidx}"
            f.write(f"*Elset, elset={name}\n")
            # Abaqus: 16 ids per line.
            for i in range(0, len(ids), 16):
                emit(", ".join(str(int(e)) for e in ids[i:i+16]) + ",\n")

        flush_lines(f, lines)
        f.write("*End Part\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=TexGenInstance, part=TexGenPart\n*End Instance\n")
        f.write("*End Assembly\n")


def _write_adaptive_inp(path: Path,
                        cells: AdaptiveVoxelCells,
                        textile_name: str = "TexGenAdaptivePython",
                        progress: Any = False) -> dict:
    """Write adaptive non-uniform hex cells as an Abaqus input deck.

    Parameters
    ----------
    path : pathlib.Path
        Output file path. Parent directories must already exist.
    cells : AdaptiveVoxelCells
        Leaf-cell geometry and yarn ownership arrays.
    textile_name : str, default="TexGenAdaptivePython"
        Name written to the Abaqus heading.
    progress : bool or callable, default=False
        Show tqdm progress bars for node deduplication and mesh writing.

    Returns
    -------
    dict
        Mesh counts with keys ``"nodes"`` and ``"elements"``.

    Notes
    -----
    Nodes are deduplicated by rounded coordinates. This keeps adjacent adaptive
    cells connected when their corner coordinates match.
    """
    node_offsets = _HEX_NODE_OFFSETS.astype(cells.lows.dtype, copy=False)
    node_ids = {}
    node_coords = []

    def node_key(coord: np.ndarray) -> tuple:
        """Return a hashable rounded coordinate key for node deduplication."""
        return tuple(np.round(coord.astype(np.float64, copy=False), 12))

    cell_pairs = zip(cells.lows, cells.sizes)
    for low, size in _progress_iter(
        cell_pairs, progress, total=int(cells.lows.shape[0]),
        desc="deduplicate adaptive nodes", unit="cell"
    ):
        for offset in node_offsets:
            coord = low + size * offset
            key = node_key(coord)
            if key not in node_ids:
                node_ids[key] = len(node_coords) + 1
                node_coords.append(coord.astype(np.float64, copy=False))

    def flush_lines(file_obj, pending):
        """Write and clear buffered text lines."""
        if pending:
            file_obj.writelines(pending)
            pending.clear()

    with path.open("w", encoding="utf-8", newline="\n") as f:
        lines = []

        def emit(line: str):
            """Append one line to the local write buffer."""
            lines.append(line)
            if len(lines) >= 8192:
                flush_lines(f, lines)

        f.write("*Heading\n")
        f.write(f"TexGen Python adaptive voxel mesh: {textile_name}\n")
        f.write("** Lightweight linear-octree mesh generated by numpy backend.\n")
        f.write("** Hanging-node constraints and p4est-style 2:1 balancing are not generated.\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("**\n*Part, name=TexGenPart\n*Node\n")
        node_iter = enumerate(node_coords, start=1)
        for node_id, coord in _progress_iter(
            node_iter, progress, total=len(node_coords),
            desc="write adaptive nodes", unit="node"
        ):
            emit(f"{node_id}, {coord[0]:.6g}, {coord[1]:.6g}, {coord[2]:.6g}\n")

        flush_lines(f, lines)
        f.write("*Element, type=C3D8R\n")
        elem_iter = enumerate(zip(cells.lows, cells.sizes), start=1)
        for elem_id, (low, size) in _progress_iter(
            elem_iter, progress, total=int(cells.lows.shape[0]),
            desc="write adaptive elements", unit="element"
        ):
            conn = []
            for offset in node_offsets:
                conn.append(node_ids[node_key(low + size * offset)])
            emit(f"{elem_id}, " + ", ".join(str(node_id) for node_id in conn) + "\n")

        flush_lines(f, lines)
        unique_yarns = np.unique(cells.yarn_id)
        for yidx in _progress_iter(
            unique_yarns, progress, total=len(unique_yarns),
            desc="write adaptive element sets", unit="set"
        ):
            ids = np.nonzero(cells.yarn_id == yidx)[0] + 1
            name = "Matrix" if yidx < 0 else f"Yarn{yidx}"
            f.write(f"*Elset, elset={name}\n")
            for i in range(0, len(ids), 16):
                emit(", ".join(str(int(e)) for e in ids[i:i+16]) + ",\n")

        flush_lines(f, lines)
        f.write("*End Part\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=TexGenInstance, part=TexGenPart\n*End Instance\n")
        f.write("*End Assembly\n")

    return dict(nodes=len(node_coords), elements=int(cells.lows.shape[0]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_voxelizer_args(nx: int, ny: int, nz: int,
                             backend: str, dtype: str, chunk_voxels: int,
                             adaptive_levels: int,
                             max_adaptive_cells: int) -> int:
    """Validate public voxelizer arguments.

    Parameters
    ----------
    nx, ny, nz : int
        Structured base-grid resolution.
    backend : str
        Requested backend name: ``"auto"``, ``"numpy"`` or ``"torch"``.
    dtype : str
        Requested floating dtype: ``"float32"`` or ``"float64"``.
    chunk_voxels : int
        Number of voxels processed per classifier chunk.
    adaptive_levels : int
        Number of adaptive refinement passes.
    max_adaptive_cells : int
        Maximum allowed adaptive leaf cells.

    Returns
    -------
    int
        Base cell count ``nx*ny*nz``.

    Raises
    ------
    ValueError
        If any argument is outside the accepted range.
    """
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError('backend must be one of "auto", "numpy", or "torch"')
    if dtype not in {"float32", "float64"}:
        raise ValueError('dtype must be "float32" or "float64"')
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, and nz must be >= 1")
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be >= 1")
    if adaptive_levels < 0:
        raise ValueError("adaptive_levels must be >= 0")
    base_cell_count = nx * ny * nz
    if max_adaptive_cells < base_cell_count:
        raise ValueError("max_adaptive_cells must be at least nx*ny*nz")
    return base_cell_count


def _validate_orientation_storage(orientation_storage: str) -> str:
    """Normalize and validate the requested orientation field layout."""
    value = str(orientation_storage).lower()
    if value not in {"dense", "sparse"}:
        raise ValueError('orientation_storage must be "dense" or "sparse"')
    return value


def _sparse_orientation_from_dense(yarn_id: Any,
                                   orientation1: Any,
                                   orientation2: Any,
                                   grid_shape: Tuple[int, int, int]
                                   ) -> SparseOrientationField:
    """Compact dense flat orientations to yarn voxels without changing backend."""
    if _is_torch_tensor(yarn_id):
        indices = (yarn_id >= 0).nonzero(as_tuple=False).flatten()
    else:
        indices = np.flatnonzero(np.asarray(yarn_id) >= 0).astype(
            np.int64, copy=False
        )
    return SparseOrientationField(
        voxel_indices=indices,
        yarn_ids=yarn_id[indices],
        orientation1=orientation1[indices],
        orientation2=orientation2[indices],
        grid_shape=grid_shape,
    )


def _resolve_backend(backend: str,
                     device: Optional[str],
                     dtype: str,
                     workers: Optional[int],
                     adaptive: bool) -> BackendSelection:
    """Resolve user backend options into concrete execution settings.

    Parameters
    ----------
    backend : {"auto", "numpy", "torch"}
        Requested backend.
    device : str or None
        Requested torch device. ``None`` allows automatic selection.
    dtype : {"float32", "float64"}
        Numerical precision.
    workers : int or None
        Requested numpy worker count. Ignored by torch.
    adaptive : bool
        Whether adaptive voxelization is active. Adaptive mode currently forces
        numpy.

    Returns
    -------
    BackendSelection
        Concrete backend/device/dtype/worker configuration.
    """
    if adaptive:
        if backend == "torch":
            raise ValueError("adaptive=True currently supports only the numpy backend")
        if backend == "auto":
            backend = "numpy"

    torch_mod = torch
    if backend == "auto":
        if device is not None:
            backend = "torch"
        elif torch_mod is not None:
            has_cuda = torch_mod.cuda.is_available()
            has_mps = getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available()
            backend = "torch" if (has_cuda or has_mps) else "numpy"
        else:
            backend = "numpy"

    if backend == "torch":
        torch_mod = _require_torch()
        if device is None:
            if torch_mod.cuda.is_available():
                device = "cuda"
            elif getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        torch_dtype = {"float32": torch_mod.float32, "float64": torch_mod.float64}[dtype]
        return BackendSelection(
            backend="torch", device=device, workers=1,
            torch_dtype=torch_dtype, torch_module=torch_mod,
        )

    workers_used = _default_numpy_workers() if workers is None else workers
    if workers_used < 1:
        raise ValueError("workers must be >= 1")
    np_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    return BackendSelection(
        backend="numpy", device="cpu", workers=workers_used, np_dtype=np_dtype
    )


def _structured_voxel_centers(lo: np.ndarray, hi: np.ndarray,
                              nx: int, ny: int, nz: int,
                              dtype=np.float64) -> np.ndarray:
    """Build structured voxel centers in TexGen element order.

    Parameters
    ----------
    lo, hi : numpy.ndarray
        Domain lower and upper corners, each shape ``(3,)``.
    nx, ny, nz : int
        Structured grid resolution.
    dtype : numpy dtype, default=np.float64
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Voxel centers with shape ``(nx*ny*nz, 3)``. Flattening order is
        ``ix + iy*nx + iz*nx*ny``, matching Abaqus element output.
    """
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz
    xs = np.asarray(lo[0], dtype=dtype) + (np.arange(nx, dtype=dtype) + 0.5) * np.asarray(dx, dtype=dtype)
    ys = np.asarray(lo[1], dtype=dtype) + (np.arange(ny, dtype=dtype) + 0.5) * np.asarray(dy, dtype=dtype)
    zs = np.asarray(lo[2], dtype=dtype) + (np.arange(nz, dtype=dtype) + 0.5) * np.asarray(dz, dtype=dtype)

    centers = np.empty((nz, ny, nx, 3), dtype=dtype)
    centers[..., 0] = xs
    centers[..., 1] = ys[None, :, None]
    centers[..., 2] = zs[:, None, None]
    return centers.reshape(-1, 3)  # outer-to-inner: z,y,x


def _textile_name(textile: CTextile) -> str:
    """Return a safe display name for a TexGen textile.

    Parameters
    ----------
    textile : CTextile
        TexGen textile object.

    Returns
    -------
    str
        ``textile.GetName()`` when available, otherwise ``"Textile"``.
    """
    return getattr(textile, "GetName", lambda: "Textile")()


def _sync_torch_backend(torch_mod, device: Optional[str]) -> None:
    """Synchronize asynchronous torch devices before timing or copying results.

    Parameters
    ----------
    torch_mod : module
        Imported torch module.
    device : str or None
        Device name. CUDA and MPS are synchronized; CPU is a no-op.
    """
    if device == "cuda":
        torch_mod.cuda.synchronize()
    elif device == "mps" and hasattr(torch_mod, "mps"):
        torch_mod.mps.synchronize()


def voxelize_snapshots_data(snapshots: List[YarnSnapshot],
                            aabb: np.ndarray,
                            nx: int = 64, ny: int = 64, nz: int = 64,
                            backend: str = "numpy",
                            device: Optional[str] = None,
                            dtype: str = "float32",
                            chunk_voxels: int = DEFAULT_NUMPY_CHUNK_VOXELS,
                            workers: Optional[int] = None,
                            verbose: bool = True,
                            include_centers: bool = False,
                            include_orientations: bool = False,
                            orientation_storage: str = "dense",
                            output: str = "backend",
                            aabb_pruning: bool = True,
                            progress: Any = False) -> VoxelGridData:
    """Voxelize pre-extracted yarn snapshots and return numpy/torch data.

    This skips the TexGen/SWIG object traversal in :func:`extract_snapshots`.
    It is the preferred path for repeated voxelization of the same textile, for
    example resolution sweeps or emitting both numpy and torch outputs.
    """
    backend = backend.lower()
    output = output.lower()
    orientation_storage = _validate_orientation_storage(orientation_storage)
    if output not in {"backend", "numpy", "torch"}:
        raise ValueError('output must be one of "backend", "numpy", or "torch"')

    _validate_voxelizer_args(
        nx, ny, nz, backend, dtype, chunk_voxels,
        adaptive_levels=0, max_adaptive_cells=nx * ny * nz
    )
    if len(snapshots) == 0:
        raise RuntimeError("No yarn snapshots provided")

    backend_cfg = _resolve_backend(backend, device, dtype, workers, adaptive=False)

    def log(msg):
        """Print one timing/status line when verbose output is enabled."""
        if verbose:
            print(f"[voxelizer] {msg}")

    aabb_np = np.asarray(aabb, dtype=np.float64)
    if aabb_np.shape != (2, 3):
        raise ValueError(f"aabb must have shape (2, 3), got {aabb_np.shape}")

    lo, hi = aabb_np[0], aabb_np[1]
    centers_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    centers_np = _structured_voxel_centers(lo, hi, nx, ny, nz, dtype=centers_dtype)
    centers_out = None
    orientation1 = None
    orientation2 = None
    sparse_orientation = None

    t0 = time.perf_counter()
    if backend_cfg.backend == "torch":
        torch_mod = backend_cfg.torch_module
        packed = _pack_yarns(
            snapshots, device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        centers = torch_mod.from_numpy(centers_np).to(
            device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        classified = _classify_voxels_torch(
            centers, packed, chunk=chunk_voxels, aabb_pruning=aabb_pruning,
            progress=progress,
            include_orientations=include_orientations,
            orientation_storage=orientation_storage,
        )
        if not include_orientations:
            yarn_id = classified
        elif orientation_storage == "dense":
            yarn_id, orientation1_flat, orientation2_flat = classified
            orientation1 = orientation1_flat.reshape(nz, ny, nx, 3)
            orientation2 = orientation2_flat.reshape(nz, ny, nx, 3)
        else:
            yarn_id, sparse_payload = classified
            sparse_orientation = SparseOrientationField(
                voxel_indices=sparse_payload[0],
                yarn_ids=sparse_payload[1],
                orientation1=sparse_payload[2],
                orientation2=sparse_payload[3],
                grid_shape=(nz, ny, nx),
            )
        _sync_torch_backend(torch_mod, backend_cfg.device)
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers.shape[0]:,} cached voxels with torch/"
            f"{backend_cfg.device} in {t_classify:.3f}s"
        )
        if include_centers:
            centers_out = centers
        aabb_out = torch_mod.as_tensor(
            aabb_np, device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        actual_workers = 1
    else:
        snapshots_np = _snapshots_as_dtype(snapshots, backend_cfg.np_dtype)
        centers_np = centers_np.astype(backend_cfg.np_dtype, copy=False)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        actual_workers = _effective_numpy_workers(
            centers_np.shape[0], chunk_voxels, backend_cfg.workers
        )
        classified = _classify_voxels_numpy(
            centers_np, snapshots_np, chunk=chunk_voxels, workers=backend_cfg.workers,
            aabb_pruning=aabb_pruning, progress=progress,
            include_orientations=include_orientations,
        )
        if include_orientations:
            yarn_id, orientation1_flat, orientation2_flat = classified
            if orientation_storage == "dense":
                orientation1 = orientation1_flat.reshape(nz, ny, nx, 3)
                orientation2 = orientation2_flat.reshape(nz, ny, nx, 3)
            else:
                sparse_orientation = _sparse_orientation_from_dense(
                    yarn_id,
                    orientation1_flat,
                    orientation2_flat,
                    (nz, ny, nx),
                )
        else:
            yarn_id = classified
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers_np.shape[0]:,} cached voxels with numpy/"
            f"{actual_workers} workers in {t_classify:.3f}s"
        )
        if include_centers:
            centers_out = centers_np
        aabb_out = aabb_np

    data = VoxelGridData(
        yarn_id=yarn_id,
        aabb=aabb_out,
        resolution=(nx, ny, nz),
        backend=backend_cfg.backend,
        device=backend_cfg.device,
        workers=actual_workers,
        dtype=dtype,
        timings=dict(extract=0.0, pack=t_pack, classify=t_classify),
        centers=centers_out,
        orientation1=orientation1,
        orientation2=orientation2,
        sparse_orientation=sparse_orientation,
        aabb_pruning=aabb_pruning,
        storage="torch" if backend_cfg.backend == "torch" else "numpy",
    )
    return _coerce_voxel_grid_output(data, output, device=device)


def voxelize_snapshot_bundle_data(bundle: SnapshotBundle,
                                  nx: int = 64, ny: int = 64, nz: int = 64,
                                  backend: str = "numpy",
                                  device: Optional[str] = None,
                                  dtype: str = "float32",
                                  chunk_voxels: int = DEFAULT_NUMPY_CHUNK_VOXELS,
                                  workers: Optional[int] = None,
                                  verbose: bool = True,
                                  include_centers: bool = False,
                                  include_orientations: bool = False,
                                  orientation_storage: str = "dense",
                                  output: str = "backend",
                                  aabb_pruning: bool = True,
                                  progress: Any = False) -> VoxelGridData:
    """Voxelize a structure-of-arrays snapshot bundle.

    This is the stable entry point for future compiled ``_fastdata`` providers:
    they only need to return a ``SnapshotBundle``-compatible layout, while the
    existing numpy/torch classification and data-return code remains unchanged.
    """
    bundle = _coerce_snapshot_bundle(bundle)
    backend = backend.lower()
    output = output.lower()
    orientation_storage = _validate_orientation_storage(orientation_storage)
    if output not in {"backend", "numpy", "torch"}:
        raise ValueError('output must be one of "backend", "numpy", or "torch"')

    _validate_voxelizer_args(
        nx, ny, nz, backend, dtype, chunk_voxels,
        adaptive_levels=0, max_adaptive_cells=nx * ny * nz
    )
    if bundle.num_yarns == 0:
        raise RuntimeError("No yarn snapshots provided")

    backend_cfg = _resolve_backend(backend, device, dtype, workers, adaptive=False)

    if backend_cfg.backend == "torch":
        t0 = time.perf_counter()
        snapshots = bundle.to_snapshots()
        t_unpack = time.perf_counter() - t0
        data = voxelize_snapshots_data(
            snapshots,
            bundle.aabb,
            nx=nx,
            ny=ny,
            nz=nz,
            backend=backend_cfg.backend,
            device=backend_cfg.device,
            dtype=dtype,
            chunk_voxels=chunk_voxels,
            workers=workers,
            verbose=verbose,
            include_centers=include_centers,
            include_orientations=include_orientations,
            orientation_storage=orientation_storage,
            output=output,
            aabb_pruning=aabb_pruning,
            progress=progress,
        )
        data.timings["unpack"] = t_unpack
        return data

    def log(msg):
        """Print one timing/status line when verbose output is enabled."""
        if verbose:
            print(f"[voxelizer] {msg}")

    aabb_np = np.asarray(bundle.aabb, dtype=np.float64)
    if aabb_np.shape != (2, 3):
        raise ValueError(f"aabb must have shape (2, 3), got {aabb_np.shape}")

    lo, hi = aabb_np[0], aabb_np[1]
    centers_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    centers_np = _structured_voxel_centers(lo, hi, nx, ny, nz, dtype=centers_dtype)
    centers_out = None
    orientation1 = None
    orientation2 = None
    sparse_orientation = None

    t0 = time.perf_counter()
    bundle_np = _bundle_as_dtype(bundle, backend_cfg.np_dtype)
    centers_np = centers_np.astype(backend_cfg.np_dtype, copy=False)
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    actual_workers = _effective_numpy_workers(
        centers_np.shape[0], chunk_voxels, backend_cfg.workers
    )
    classified = _classify_voxels_bundle_numpy(
        centers_np,
        bundle_np,
        chunk=chunk_voxels,
        workers=backend_cfg.workers,
        aabb_pruning=aabb_pruning,
        progress=progress,
        include_orientations=include_orientations,
    )
    if include_orientations:
        yarn_id, orientation1_flat, orientation2_flat = classified
        if orientation_storage == "dense":
            orientation1 = orientation1_flat.reshape(nz, ny, nx, 3)
            orientation2 = orientation2_flat.reshape(nz, ny, nx, 3)
        else:
            sparse_orientation = _sparse_orientation_from_dense(
                yarn_id,
                orientation1_flat,
                orientation2_flat,
                (nz, ny, nx),
            )
    else:
        yarn_id = classified
    t_classify = time.perf_counter() - t0
    log(
        f"classified {centers_np.shape[0]:,} bundle voxels with numpy/"
        f"{actual_workers} workers in {t_classify:.3f}s"
    )
    if include_centers:
        centers_out = centers_np

    data = VoxelGridData(
        yarn_id=yarn_id,
        aabb=aabb_np,
        resolution=(nx, ny, nz),
        backend=backend_cfg.backend,
        device=backend_cfg.device,
        workers=actual_workers,
        dtype=dtype,
        timings=dict(extract=0.0, unpack=0.0, pack=t_pack, classify=t_classify),
        centers=centers_out,
        orientation1=orientation1,
        orientation2=orientation2,
        sparse_orientation=sparse_orientation,
        aabb_pruning=aabb_pruning,
        storage="numpy",
    )
    return _coerce_voxel_grid_output(data, output, device=device)


def voxelize_textile_data(textile: CTextile,
                          nx: int = 64, ny: int = 64, nz: int = 64,
                          backend: str = "numpy",
                          device: Optional[str] = None,
                          dtype: str = "float32",
                          chunk_voxels: int = DEFAULT_NUMPY_CHUNK_VOXELS,
                          workers: Optional[int] = None,
                          verbose: bool = True,
                          include_centers: bool = False,
                          include_orientations: bool = False,
                          orientation_storage: str = "dense",
                          output: str = "backend",
                          aabb_pruning: bool = True,
                          progress: Any = False) -> VoxelGridData:
    """Voxelize a built CTextile and return direct numpy/torch data.

    This path skips Abaqus ``.inp`` generation. It is intended for solvers that
    can consume structured arrays or tensors directly and should avoid the file
    write plus torch-to-CPU copy used by ``voxelize_textile``.

    Parameters
    ----------
    textile : CTextile
        A fully built textile (all section/refine work done by TexGen).
    nx, ny, nz : int
        Voxel resolution along each axis of the domain AABB.
    backend : {"numpy", "auto", "torch"}
        Classification backend. ``numpy`` is the default OpenMP-free CPU path.
        ``auto`` may pick torch when an accelerator is available. ``torch`` can
        leave ``yarn_id`` on the selected device when ``output="backend"``.
    device : {"cuda", "mps", "cpu", None}
        Torch device for classification or forced torch output.
    dtype : {"float32", "float64"}
        Floating-point precision used for classification geometry.
    chunk_voxels : int
        Voxels processed per batch.
    workers : int or None
        Number of numpy worker threads. Ignored by torch classification.
    verbose : bool
        Print per-phase timing.
    include_centers : bool
        Include voxel centers in the returned data object. Disabled by default
        to keep solver handoff memory-light.
    include_orientations : bool
        Include ``orientation1`` and ``orientation2`` grids with shape
        ``(nz, ny, nx, 3)``. ``orientation1`` is the yarn tangent and
        ``orientation2`` is the yarn up vector at the nearest yarn node.
        Matrix voxels are filled with zero vectors in dense storage.
    orientation_storage : {"dense", "sparse"}
        ``dense`` returns full direction grids. ``sparse`` stores directions
        only for yarn voxels in ``data.sparse_orientation``; matrix voxels have
        no direction entries.
    output : {"backend", "numpy", "torch"}
        Storage backend for returned arrays. ``backend`` preserves the
        classification result storage; ``numpy`` forces CPU numpy arrays;
        ``torch`` returns torch tensors.
    aabb_pruning : bool
        Skip yarn/translation candidates whose conservative bounding boxes do
        not overlap the current voxel chunk.
    progress : bool or callable
        Show tqdm progress bars over classifier chunks when true. ``tqdm`` is
        imported lazily and is not required unless this is enabled.

    Returns
    -------
    VoxelGridData
        Structured voxel ids and metadata. ``data.grid`` is a zero-copy
        ``(nz, ny, nx)`` view. ``data.yarn_id`` remains flat in TexGen element
        order for direct finite-element assembly.
    """
    backend = backend.lower()
    output = output.lower()
    orientation_storage = _validate_orientation_storage(orientation_storage)
    if output not in {"backend", "numpy", "torch"}:
        raise ValueError('output must be one of "backend", "numpy", or "torch"')

    _validate_voxelizer_args(
        nx, ny, nz, backend, dtype, chunk_voxels,
        adaptive_levels=0, max_adaptive_cells=nx * ny * nz
    )
    backend_cfg = _resolve_backend(backend, device, dtype, workers, adaptive=False)

    def log(msg):
        """Print one timing/status line when verbose output is enabled."""
        if verbose:
            print(f"[voxelizer] {msg}")

    t0 = time.perf_counter()
    bundle = extract_snapshot_bundle(textile)
    aabb = bundle.aabb
    t_extract = time.perf_counter() - t0
    log(
        f"extracted {bundle.num_yarns} yarns, AABB={aabb.tolist()}, "
        f"backend={backend_cfg.backend}, workers={backend_cfg.workers}, {t_extract:.3f}s"
    )

    if bundle.num_yarns == 0:
        raise RuntimeError("No yarns extracted - textile may be empty or unbuilt")

    if backend_cfg.backend == "numpy":
        data = voxelize_snapshot_bundle_data(
            bundle,
            nx=nx,
            ny=ny,
            nz=nz,
            backend=backend_cfg.backend,
            device=backend_cfg.device,
            dtype=dtype,
            chunk_voxels=chunk_voxels,
            workers=workers,
            verbose=verbose,
            include_centers=include_centers,
            include_orientations=include_orientations,
            orientation_storage=orientation_storage,
            output=output,
            aabb_pruning=aabb_pruning,
            progress=progress,
        )
        data.timings["extract"] = t_extract
        return data

    snapshots = bundle.to_snapshots()

    lo, hi = aabb[0], aabb[1]
    centers_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    centers_np = _structured_voxel_centers(lo, hi, nx, ny, nz, dtype=centers_dtype)
    centers_out = None
    orientation1 = None
    orientation2 = None
    sparse_orientation = None

    t0 = time.perf_counter()
    if backend_cfg.backend == "torch":
        torch_mod = backend_cfg.torch_module
        packed = _pack_yarns(
            snapshots, device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        centers = torch_mod.from_numpy(centers_np).to(
            device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        classified = _classify_voxels_torch(
            centers, packed, chunk=chunk_voxels, aabb_pruning=aabb_pruning,
            progress=progress,
            include_orientations=include_orientations,
            orientation_storage=orientation_storage,
        )
        if not include_orientations:
            yarn_id = classified
        elif orientation_storage == "dense":
            yarn_id, orientation1_flat, orientation2_flat = classified
            orientation1 = orientation1_flat.reshape(nz, ny, nx, 3)
            orientation2 = orientation2_flat.reshape(nz, ny, nx, 3)
        else:
            yarn_id, sparse_payload = classified
            sparse_orientation = SparseOrientationField(
                voxel_indices=sparse_payload[0],
                yarn_ids=sparse_payload[1],
                orientation1=sparse_payload[2],
                orientation2=sparse_payload[3],
                grid_shape=(nz, ny, nx),
            )
        _sync_torch_backend(torch_mod, backend_cfg.device)
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers.shape[0]:,} voxels with torch/"
            f"{backend_cfg.device} in {t_classify:.3f}s"
        )
        if include_centers:
            centers_out = centers
        actual_workers = 1
    else:
        snapshots_np = _snapshots_as_dtype(snapshots, backend_cfg.np_dtype)
        centers_np = centers_np.astype(backend_cfg.np_dtype, copy=False)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        actual_workers = _effective_numpy_workers(
            centers_np.shape[0], chunk_voxels, backend_cfg.workers
        )
        classified = _classify_voxels_numpy(
            centers_np, snapshots_np, chunk=chunk_voxels, workers=backend_cfg.workers,
            aabb_pruning=aabb_pruning, progress=progress,
            include_orientations=include_orientations,
        )
        if include_orientations:
            yarn_id, orientation1_flat, orientation2_flat = classified
            if orientation_storage == "dense":
                orientation1 = orientation1_flat.reshape(nz, ny, nx, 3)
                orientation2 = orientation2_flat.reshape(nz, ny, nx, 3)
            else:
                sparse_orientation = _sparse_orientation_from_dense(
                    yarn_id,
                    orientation1_flat,
                    orientation2_flat,
                    (nz, ny, nx),
                )
        else:
            yarn_id = classified
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers_np.shape[0]:,} voxels with numpy/"
            f"{actual_workers} workers in {t_classify:.3f}s"
        )
        if include_centers:
            centers_out = centers_np

    data = VoxelGridData(
        yarn_id=yarn_id,
        aabb=aabb if backend_cfg.backend != "torch" else torch_mod.as_tensor(
            aabb, device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        ),
        resolution=(nx, ny, nz),
        backend=backend_cfg.backend,
        device=backend_cfg.device,
        workers=actual_workers,
        dtype=dtype,
        timings=dict(extract=t_extract, pack=t_pack, classify=t_classify),
        centers=centers_out,
        orientation1=orientation1,
        orientation2=orientation2,
        sparse_orientation=sparse_orientation,
        aabb_pruning=aabb_pruning,
        storage="torch" if backend_cfg.backend == "torch" else "numpy",
    )
    return _coerce_voxel_grid_output(data, output, device=device)


def voxelize_textile(textile: CTextile,
                     nx: int = 64, ny: int = 64, nz: int = 64,
                     out_inp: str = "out.inp",
                     backend: str = "numpy",
                     device: Optional[str] = None,
                     dtype: str = "float32",
                     chunk_voxels: int = DEFAULT_NUMPY_CHUNK_VOXELS,
                     workers: Optional[int] = None,
                     verbose: bool = True,
                     adaptive: bool = False,
                     adaptive_levels: int = 1,
                     max_adaptive_cells: int = 2_000_000,
                     aabb_pruning: bool = True,
                     progress: Any = False) -> dict:
    """Voxelize a built CTextile and write an Abaqus .inp.

    Parameters
    ----------
    textile : CTextile
        A fully built textile (all section/refine work done by TexGen).
    nx, ny, nz : int
        Voxel resolution along each axis of the domain AABB.
    out_inp : str
        Output Abaqus input deck path.
    backend : {"numpy", "auto", "torch"}
        ``numpy`` uses portable CPU vectorization and is the default OpenMP-free
        path. ``torch`` uses CUDA/MPS/CPU tensors. ``auto`` picks torch only
        when an accelerator is available or when ``device`` is explicitly
        provided; otherwise it uses numpy.
    device : {"cuda", "mps", "cpu", None}
        Torch device. Ignored by the numpy backend.
    dtype : {"float32", "float64"}
        Numerical precision. float32 is usually enough for voxelization.
    chunk_voxels : int
        Voxels processed per batch (controls memory).
    workers : int or None
        Number of numpy worker threads. None uses a conservative auto value
        capped at 4. Ignored by the torch backend.
    verbose : bool
        Print per-phase timing.
    adaptive : bool
        Use lightweight numpy linear-octree refinement instead of a structured
        rectangular grid. This mode writes non-uniform hex cells and does not
        generate p4est-style hanging-node constraints.
    adaptive_levels : int
        Maximum number of center/corner disagreement refinement passes.
    max_adaptive_cells : int
        Safety cap on generated adaptive leaf cells.
    aabb_pruning : bool
        Skip yarn/translation candidates whose conservative bounding boxes do
        not overlap the current voxel chunk. Enabled by default.
    progress : bool or callable
        Show tqdm progress bars over classifier and writer chunks when true.
        ``tqdm`` is imported lazily and is not required unless this is enabled.

    Returns
    -------
    dict with ``yarn_id`` (np.ndarray of shape (nx*ny*nz,), row-major ix+iy*nx+iz*nx*ny order),
    ``aabb`` (2x3), backend/device, and timing info.
    """
    backend = backend.lower()
    _validate_voxelizer_args(
        nx, ny, nz, backend, dtype, chunk_voxels,
        adaptive_levels, max_adaptive_cells
    )
    backend_cfg = _resolve_backend(backend, device, dtype, workers, adaptive)

    def log(msg):
        """Print one timing/status line when verbose output is enabled."""
        if verbose:
            print(f"[voxelizer] {msg}")

    t0 = time.perf_counter()
    bundle = extract_snapshot_bundle(textile)
    snapshots, aabb = bundle.to_snapshots(), bundle.aabb
    t_extract = time.perf_counter() - t0
    log(
        f"extracted {len(snapshots)} yarns, AABB={aabb.tolist()}, "
        f"backend={backend_cfg.backend}, workers={backend_cfg.workers}, {t_extract:.3f}s"
    )

    if len(snapshots) == 0:
        raise RuntimeError("No yarns extracted - textile may be empty or unbuilt")

    lo, hi = aabb[0], aabb[1]

    if adaptive:
        t0 = time.perf_counter()
        snapshots_np = _snapshots_as_dtype(snapshots, backend_cfg.np_dtype)
        lows, sizes, levels = _structured_cell_lows_sizes(lo, hi, nx, ny, nz, backend_cfg.np_dtype)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        lows, sizes, levels = _refine_adaptive_cells(
            lows, sizes, levels, snapshots_np, adaptive_levels, chunk_voxels,
            backend_cfg.workers, max_adaptive_cells, aabb_pruning=aabb_pruning,
            progress=progress
        )
        t_refine = time.perf_counter() - t0
        log(
            f"adaptive mesh has {lows.shape[0]:,} cells after {adaptive_levels} "
            f"level(s), max level={int(levels.max()) if len(levels) else 0}, {t_refine:.3f}s"
        )

        t0 = time.perf_counter()
        actual_workers = _effective_numpy_workers(
            lows.shape[0], chunk_voxels, backend_cfg.workers
        )
        yarn_id = _classify_adaptive_cells_numpy(
            lows, sizes, snapshots_np, chunk_voxels, backend_cfg.workers,
            aabb_pruning=aabb_pruning, progress=progress
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {lows.shape[0]:,} adaptive cells with numpy/"
            f"{actual_workers} workers in {t_classify:.3f}s"
        )

        t0 = time.perf_counter()
        out_path = Path(out_inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells = AdaptiveVoxelCells(lows=lows, sizes=sizes, levels=levels, yarn_id=yarn_id)
        mesh_counts = _write_adaptive_inp(
            out_path, cells, textile_name=_textile_name(textile), progress=progress
        )
        t_write = time.perf_counter() - t0
        log(
            f"wrote {out_path} ({mesh_counts['elements']:,} elements, "
            f"{mesh_counts['nodes']:,} nodes, {t_write:.3f}s)"
        )

        return dict(
            yarn_id=yarn_id,
            aabb=aabb,
            backend=backend_cfg.backend,
            device=backend_cfg.device,
            workers=actual_workers,
            adaptive=True,
            aabb_pruning=aabb_pruning,
            levels=levels,
            num_cells=int(lows.shape[0]),
            mesh=mesh_counts,
            timings=dict(
                extract=t_extract, pack=t_pack, refine=t_refine,
                classify=t_classify, write=t_write
            ),
        )

    centers_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    centers_np = _structured_voxel_centers(lo, hi, nx, ny, nz, dtype=centers_dtype)

    t0 = time.perf_counter()
    if backend_cfg.backend == "torch":
        torch_mod = backend_cfg.torch_module
        packed = _pack_yarns(
            snapshots, device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        centers = torch_mod.from_numpy(centers_np).to(
            device=backend_cfg.device, dtype=backend_cfg.torch_dtype
        )
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        yarn_id_tensor = _classify_voxels_torch(
            centers, packed, chunk=chunk_voxels, aabb_pruning=aabb_pruning,
            progress=progress
        )
        _sync_torch_backend(torch_mod, backend_cfg.device)
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers.shape[0]:,} voxels with torch/"
            f"{backend_cfg.device} in {t_classify:.3f}s"
        )
        yarn_id = yarn_id_tensor.detach().cpu().numpy()
        actual_workers = 1
    else:
        snapshots_np = _snapshots_as_dtype(snapshots, backend_cfg.np_dtype)
        centers_np = centers_np.astype(backend_cfg.np_dtype, copy=False)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        actual_workers = _effective_numpy_workers(
            centers_np.shape[0], chunk_voxels, backend_cfg.workers
        )
        yarn_id = _classify_voxels_numpy(
            centers_np, snapshots_np, chunk=chunk_voxels, workers=backend_cfg.workers,
            aabb_pruning=aabb_pruning, progress=progress
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers_np.shape[0]:,} voxels with numpy/"
            f"{actual_workers} workers in {t_classify:.3f}s"
        )

    t0 = time.perf_counter()
    out_path = Path(out_inp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_inp(out_path, lo, hi, nx, ny, nz, yarn_id,
               textile_name=_textile_name(textile), progress=progress)
    t_write = time.perf_counter() - t0
    log(f"wrote {out_path} ({t_write:.3f}s)")

    return dict(
        yarn_id=yarn_id,
        aabb=aabb,
        backend=backend_cfg.backend,
        device=backend_cfg.device,
        workers=actual_workers,
        adaptive=False, aabb_pruning=aabb_pruning,
        timings=dict(extract=t_extract, pack=t_pack, classify=t_classify, write=t_write),
    )


__all__ = [
    "voxelize_textile",
    "voxelize_textile_data",
    "voxelize_snapshots_data",
    "voxelize_snapshot_bundle_data",
    "extract_snapshots",
    "extract_snapshot_bundle",
    "fastdata_provider_status",
    "YarnSnapshot",
    "SnapshotBundle",
    "AdaptiveVoxelCells",
    "VoxelGridData",
    "VoxelizationCache",
]
