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
    from pytexgen.gpu_voxelizer import voxelize_textile

    T = CShearedTextileWeave2D(3,3,5.0,2.0,0.2618,True,True)
    T.SetYarnWidths(2.0); T.SetYarnHeights(0.8); T.AssignDefaultDomain()

    voxelize_textile(T, nx=64, ny=64, nz=64, out_inp="out.inp", backend="numpy")
    voxelize_textile(T, nx=16, ny=16, nz=8, out_inp="adaptive.inp",
                     backend="numpy", adaptive=True, adaptive_levels=2)
    voxelize_textile(T, nx=64, ny=64, nz=64, out_inp="out_torch.inp", backend="torch")
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from .Core import CTextile, CYarn  # type: ignore
except ImportError:
    from TexGen.Core import CTextile, CYarn  # type: ignore

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


@dataclass
class AdaptiveVoxelCells:
    """Leaf cells for lightweight linear-octree voxel output."""
    lows: np.ndarray       # (E, 3) lower corner per cell
    sizes: np.ndarray      # (E, 3) cell dimensions
    levels: np.ndarray     # (E,) refinement level relative to the base grid
    yarn_id: np.ndarray    # (E,) owning yarn index (-1 = matrix)


@dataclass
class BackendSelection:
    """Resolved numerical backend settings."""
    backend: str
    device: str
    workers: int
    np_dtype: Optional[type] = None
    torch_dtype: Optional[object] = None
    torch_module: Optional[object] = None


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


def _classify_voxels_torch(centers: torch.Tensor,
                           packed: dict,
                           chunk: int = 65536,
                           aabb_pruning: bool = True) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor
        ``int32`` yarn index for each voxel center, shape ``(V,)``. ``-1`` is
        matrix/background.
    """
    torch_mod = _require_torch()
    device = centers.device
    V = centers.shape[0]
    yarn_id = torch_mod.full((V,), -1, device=device, dtype=torch_mod.int32)

    P, T, U, S = packed["P"], packed["T"], packed["U"], packed["S"]
    M_len = packed["M"]
    Sec, N_len = packed["Sec"], packed["N"]
    Tr, K_len = packed["Tr"], packed["K"]
    BoundsLo = packed.get("BoundsLo")
    BoundsHi = packed.get("BoundsHi")
    num_yarns = P.shape[0]

    # Process voxels in chunks to cap VRAM.
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        pts = centers[v0:v1]                           # (C, 3)
        C = pts.shape[0]
        chunk_lo = pts.amin(dim=0)
        chunk_hi = pts.amax(dim=0)
        best_dist = torch_mod.full((C,), float("inf"), device=device)
        best_yarn = torch_mod.full((C,), -1, device=device, dtype=torch_mod.int32)

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

                # Find closest slave node per point without materialising a
                # (C, M, 3) tensor.
                d2 = (
                    active_pts.square().sum(dim=1, keepdim=True)
                    + Pt.square().sum(dim=1).unsqueeze(0)
                    - 2.0 * (active_pts @ Pt.transpose(0, 1))
                ).clamp_min_(0.0)                       # (C, M)
                nn = d2.argmin(dim=1)                   # (C,)

                # Project point into local frame of nearest slave node.
                rel = active_pts - Pt[nn]               # (C, 3)
                tan = Ty[nn]
                up  = Uy[nn]
                sid = Sy[nn]
                u_coord = (rel * sid).sum(-1)           # (C,)
                v_coord = (rel * up ).sum(-1)
                t_coord = (rel * tan).sum(-1)

                # Point-in-polygon in (u, v) plane.
                uv = torch_mod.stack([u_coord, v_coord], dim=-1)  # (C, 2)
                inside = _point_in_polygon_batch(uv, poly, n)

                # Rough "depth" proxy for overlap resolution: use d2 (Euclidean
                # to closest slave node). Real surface distance would require
                # signed distance to polygon edge; this proxy is sufficient for
                # consistent overlap assignment and matches TexGen's behaviour
                # where the yarn with the closest section wins.
                dist = torch_mod.sqrt(d2.gather(1, nn[:, None]).squeeze(-1))
                # Also penalise large longitudinal offset (out-of-section slab).
                dist = dist + t_coord.abs() * 0.1

                if active_idx is None:
                    upd = inside & (dist < best_dist)
                    best_dist = torch_mod.where(upd, dist, best_dist)
                    best_yarn = torch_mod.where(upd, torch_mod.full_like(best_yarn, y_idx), best_yarn)
                else:
                    upd = inside & (dist < best_dist[active_idx])
                    if bool(upd.any().item()):
                        target = active_idx[upd]
                        best_dist[target] = dist[upd]
                        best_yarn[target] = y_idx

        yarn_id[v0:v1] = best_yarn

    return yarn_id


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


def _classify_voxel_chunk_numpy(pts: np.ndarray,
                                snapshots: List[YarnSnapshot],
                                bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                                aabb_pruning: bool = True) -> np.ndarray:
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
            else:
                update = inside & (dist < best_dist[active_idx])
                target = active_idx[update]
                best_dist[target] = dist[update]
                best_yarn[target] = y_idx

    return best_yarn


def _default_numpy_workers() -> int:
    """Return the conservative default number of numpy worker threads.

    Returns
    -------
    int
        Between 1 and 4, capped to avoid oversubscription with BLAS, torch, or
        host applications embedding TexGen.
    """
    return max(1, min(os.cpu_count() or 1, 4))


def _classify_voxels_numpy(centers: np.ndarray,
                           snapshots: List[YarnSnapshot],
                           chunk: int = 65536,
                           workers: Optional[int] = None,
                           aabb_pruning: bool = True) -> np.ndarray:
    """Classify all voxel centers with the numpy backend.

    Parameters
    ----------
    centers : numpy.ndarray
        Voxel center coordinates, shape ``(V, 3)``.
    snapshots : list of YarnSnapshot
        Yarn snapshots to test.
    chunk : int, default=65536
        Number of voxel centers processed per task.
    workers : int or None, default=None
        Number of Python worker threads. ``None`` uses
        :func:`_default_numpy_workers`.
    aabb_pruning : bool, default=True
        Skip yarn/translation candidates whose conservative bounding boxes
        cannot contain the current voxel chunk.

    Returns
    -------
    numpy.ndarray
        ``int32`` yarn index for each center, shape ``(V,)``. ``-1`` is
        matrix/background.
    """
    V = centers.shape[0]
    yarn_id = np.full(V, -1, dtype=np.int32)
    ranges = [(v0, min(v0 + chunk, V)) for v0 in range(0, V, chunk)]
    bounds_list = [_snapshot_translation_bounds(s) for s in snapshots] if aabb_pruning else None

    worker_count = _default_numpy_workers() if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be >= 1")
    worker_count = min(worker_count, len(ranges))

    def classify_range(range_bounds):
        """Classify one ``(start, stop)`` center slice for executor.map."""
        v0, v1 = range_bounds
        return v0, v1, _classify_voxel_chunk_numpy(
            centers[v0:v1], snapshots, bounds=bounds_list, aabb_pruning=aabb_pruning
        )

    if worker_count == 1:
        for range_bounds in ranges:
            v0, v1, ids = classify_range(range_bounds)
            yarn_id[v0:v1] = ids
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for v0, v1, ids in executor.map(classify_range, ranges):
                yarn_id[v0:v1] = ids

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
                           aabb_pruning: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    lows, sizes, levels : tuple of numpy.ndarray
        Refined leaf-cell arrays.
    """
    sample_count = _ADAPTIVE_SAMPLE_OFFSETS.shape[0]
    cell_chunk = max(1, chunk_voxels // sample_count)

    for _ in range(adaptive_levels):
        refine_parts = []
        for c0 in range(0, lows.shape[0], cell_chunk):
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
                                   aabb_pruning: bool = True) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray
        ``int32`` yarn index per cell, shape ``(N,)``.
    """
    centers = lows + sizes * np.asarray(0.5, dtype=sizes.dtype)
    return _classify_voxels_numpy(
        centers, snapshots, chunk=chunk_voxels, workers=workers,
        aabb_pruning=aabb_pruning
    )


# ---------------------------------------------------------------------------
# Abaqus .inp writer (hex elements, per-element yarn index).
# ---------------------------------------------------------------------------


def _write_inp(path: Path, lo, hi, nx, ny, nz, yarn_id: np.ndarray,
               textile_name: str = "TexGenPython"):
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
        for iz in range(nnz):
            for iy in range(nny):
                for ix in range(nnx):
                    x = lo[0] + ix * dx
                    y = lo[1] + iy * dy
                    z = lo[2] + iz * dz
                    emit(f"{nid(ix,iy,iz)}, {x:.6g}, {y:.6g}, {z:.6g}\n")

        flush_lines(f, lines)
        f.write("*Element, type=C3D8R\n")
        eid = 0
        for iz in range(nz):
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
        for yidx in np.unique(yarn_id):
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
                        textile_name: str = "TexGenAdaptivePython") -> dict:
    """Write adaptive non-uniform hex cells as an Abaqus input deck.

    Parameters
    ----------
    path : pathlib.Path
        Output file path. Parent directories must already exist.
    cells : AdaptiveVoxelCells
        Leaf-cell geometry and yarn ownership arrays.
    textile_name : str, default="TexGenAdaptivePython"
        Name written to the Abaqus heading.

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

    for low, size in zip(cells.lows, cells.sizes):
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
        for node_id, coord in enumerate(node_coords, start=1):
            emit(f"{node_id}, {coord[0]:.6g}, {coord[1]:.6g}, {coord[2]:.6g}\n")

        flush_lines(f, lines)
        f.write("*Element, type=C3D8R\n")
        for elem_id, (low, size) in enumerate(zip(cells.lows, cells.sizes), start=1):
            conn = []
            for offset in node_offsets:
                conn.append(node_ids[node_key(low + size * offset)])
            emit(f"{elem_id}, " + ", ".join(str(node_id) for node_id in conn) + "\n")

        flush_lines(f, lines)
        for yidx in np.unique(cells.yarn_id):
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


def voxelize_textile(textile: CTextile,
                     nx: int = 64, ny: int = 64, nz: int = 64,
                     out_inp: str = "out.inp",
                     backend: str = "auto",
                     device: Optional[str] = None,
                     dtype: str = "float32",
                     chunk_voxels: int = 65536,
                     workers: Optional[int] = None,
                     verbose: bool = True,
                     adaptive: bool = False,
                     adaptive_levels: int = 1,
                     max_adaptive_cells: int = 2_000_000,
                     aabb_pruning: bool = True) -> dict:
    """Voxelize a built CTextile and write an Abaqus .inp.

    Parameters
    ----------
    textile : CTextile
        A fully built textile (all section/refine work done by TexGen).
    nx, ny, nz : int
        Voxel resolution along each axis of the domain AABB.
    out_inp : str
        Output Abaqus input deck path.
    backend : {"auto", "numpy", "torch"}
        ``numpy`` uses portable CPU vectorization. ``torch`` uses CUDA/MPS/CPU
        tensors. ``auto`` picks torch only when an accelerator is available or
        when ``device`` is explicitly provided; otherwise it uses numpy.
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
    snapshots, aabb = extract_snapshots(textile)
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
            backend_cfg.workers, max_adaptive_cells, aabb_pruning=aabb_pruning
        )
        t_refine = time.perf_counter() - t0
        log(
            f"adaptive mesh has {lows.shape[0]:,} cells after {adaptive_levels} "
            f"level(s), max level={int(levels.max()) if len(levels) else 0}, {t_refine:.3f}s"
        )

        t0 = time.perf_counter()
        yarn_id = _classify_adaptive_cells_numpy(
            lows, sizes, snapshots_np, chunk_voxels, backend_cfg.workers,
            aabb_pruning=aabb_pruning
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {lows.shape[0]:,} adaptive cells with numpy/"
            f"{backend_cfg.workers} workers in {t_classify:.3f}s"
        )

        t0 = time.perf_counter()
        out_path = Path(out_inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells = AdaptiveVoxelCells(lows=lows, sizes=sizes, levels=levels, yarn_id=yarn_id)
        mesh_counts = _write_adaptive_inp(
            out_path, cells, textile_name=_textile_name(textile)
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
            workers=backend_cfg.workers,
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
            centers, packed, chunk=chunk_voxels, aabb_pruning=aabb_pruning
        )
        _sync_torch_backend(torch_mod, backend_cfg.device)
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers.shape[0]:,} voxels with torch/"
            f"{backend_cfg.device} in {t_classify:.3f}s"
        )
        yarn_id = yarn_id_tensor.detach().cpu().numpy()
    else:
        snapshots_np = _snapshots_as_dtype(snapshots, backend_cfg.np_dtype)
        centers_np = centers_np.astype(backend_cfg.np_dtype, copy=False)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        yarn_id = _classify_voxels_numpy(
            centers_np, snapshots_np, chunk=chunk_voxels, workers=backend_cfg.workers,
            aabb_pruning=aabb_pruning
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers_np.shape[0]:,} voxels with numpy/"
            f"{backend_cfg.workers} workers in {t_classify:.3f}s"
        )

    t0 = time.perf_counter()
    out_path = Path(out_inp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_inp(out_path, lo, hi, nx, ny, nz, yarn_id,
               textile_name=_textile_name(textile))
    t_write = time.perf_counter() - t0
    log(f"wrote {out_path} ({t_write:.3f}s)")

    return dict(
        yarn_id=yarn_id,
        aabb=aabb,
        backend=backend_cfg.backend,
        device=backend_cfg.device,
        workers=backend_cfg.workers,
        adaptive=False, aabb_pruning=aabb_pruning,
        timings=dict(extract=t_extract, pack=t_pack, classify=t_classify, write=t_write),
    )


__all__ = ["voxelize_textile", "extract_snapshots", "YarnSnapshot", "AdaptiveVoxelCells"]
