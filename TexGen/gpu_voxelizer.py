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


def _xyz(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _xy(v) -> np.ndarray:
    return np.array([v.x, v.y], dtype=np.float64)


def _extract_yarn(yarn: CYarn, translations_xyz) -> Optional[YarnSnapshot]:
    """Pull slave-node frame + 2D section polygon out of a fully-built CYarn."""
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
    """Snapshot all yarns + domain AABB from a built textile."""
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
    """Pack yarn arrays into padded tensors for torch kernels."""
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

    return dict(P=P, T=T, U=U, S=S, M=M_len, Sec=Sec, N=N_len, Tr=Tr, K=K_len)


def _point_in_polygon_batch(points_uv: torch.Tensor,
                            polygon: torch.Tensor,
                            poly_len: int) -> torch.Tensor:
    """Ray-casting point-in-polygon test.

    points_uv:  (..., 2)  query points in local (u, v)
    polygon:    (N_max, 2)
    poly_len:   int, actual polygon length (N_max >= poly_len)
    returns:    (...,)  bool
    """
    torch_mod = _require_torch()
    poly = polygon[:poly_len]                          # (N, 2)
    p_next = torch_mod.roll(poly, shifts=-1, dims=0)   # (N, 2)
    # Broadcast: points (..., 1, 2)  vs edges (N, 2)
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
                           chunk: int = 65536) -> torch.Tensor:
    """For every voxel center, return owning yarn index (-1 = matrix/none).

    centers: (V, 3)
    returns: (V,) int32 yarn index
    """
    torch_mod = _require_torch()
    device = centers.device
    V = centers.shape[0]
    yarn_id = torch_mod.full((V,), -1, device=device, dtype=torch_mod.int32)

    P, T, U, S = packed["P"], packed["T"], packed["U"], packed["S"]
    M_len = packed["M"]
    Sec, N_len = packed["Sec"], packed["N"]
    Tr, K_len = packed["Tr"], packed["K"]
    num_yarns = P.shape[0]

    # Process voxels in chunks to cap VRAM.
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        pts = centers[v0:v1]                           # (C, 3)
        C = pts.shape[0]
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

                # Find closest slave node per point: (C, M) distance matrix.
                # Break into sub-chunks if memory tight.
                diff = pts[:, None, :] - Pt[None, :, :] # (C, M, 3)
                d2 = (diff * diff).sum(-1)              # (C, M)
                nn = d2.argmin(dim=1)                   # (C,)

                # Project point into local frame of nearest slave node.
                rel = pts - Pt[nn]                      # (C, 3)
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

                upd = inside & (dist < best_dist)
                best_dist = torch_mod.where(upd, dist, best_dist)
                best_yarn = torch_mod.where(upd, torch_mod.full_like(best_yarn, y_idx), best_yarn)

        yarn_id[v0:v1] = best_yarn

    return yarn_id


def _snapshots_as_dtype(snapshots: List[YarnSnapshot], dtype) -> List[YarnSnapshot]:
    """Return snapshots with arrays cast once for the selected numerical backend."""
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


def _point_in_polygon_batch_numpy(points_uv: np.ndarray,
                                  polygon: np.ndarray,
                                  poly_len: int) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test for numpy arrays."""
    poly = polygon[:poly_len]
    p_next = np.roll(poly, shift=-1, axis=0)

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
                                snapshots: List[YarnSnapshot]) -> np.ndarray:
    """Classify one contiguous voxel-center chunk."""
    C = pts.shape[0]
    best_dist = np.full(C, np.inf, dtype=pts.dtype)
    best_yarn = np.full(C, -1, dtype=np.int32)

    for y_idx, snap in enumerate(snapshots):
        Py = snap.positions
        Ty = snap.tangents
        Uy = snap.ups
        Sy = snap.sides
        poly = snap.section
        n = snap.section.shape[0]

        for offset in snap.translations:
            Pt = Py + offset

            diff = pts[:, None, :] - Pt[None, :, :]
            d2 = np.einsum("cmd,cmd->cm", diff, diff)
            nn = np.argmin(d2, axis=1)

            rel = pts - Pt[nn]
            tan = Ty[nn]
            up = Uy[nn]
            sid = Sy[nn]
            u_coord = np.einsum("cd,cd->c", rel, sid)
            v_coord = np.einsum("cd,cd->c", rel, up)
            t_coord = np.einsum("cd,cd->c", rel, tan)

            uv = np.stack([u_coord, v_coord], axis=-1)
            inside = _point_in_polygon_batch_numpy(uv, poly, n)

            nearest_d2 = d2[np.arange(C), nn]
            dist = np.sqrt(nearest_d2) + np.abs(t_coord) * 0.1
            update = inside & (dist < best_dist)
            best_dist[update] = dist[update]
            best_yarn[update] = y_idx

    return best_yarn


def _default_numpy_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 4))


def _classify_voxels_numpy(centers: np.ndarray,
                           snapshots: List[YarnSnapshot],
                           chunk: int = 65536,
                           workers: Optional[int] = None) -> np.ndarray:
    """For every voxel center, return owning yarn index (-1 = matrix/none)."""
    V = centers.shape[0]
    yarn_id = np.full(V, -1, dtype=np.int32)
    ranges = [(v0, min(v0 + chunk, V)) for v0 in range(0, V, chunk)]

    worker_count = _default_numpy_workers() if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be >= 1")
    worker_count = min(worker_count, len(ranges))

    def classify_range(bounds):
        v0, v1 = bounds
        return v0, v1, _classify_voxel_chunk_numpy(centers[v0:v1], snapshots)

    if worker_count == 1:
        for bounds in ranges:
            v0, v1, ids = classify_range(bounds)
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
    """Create base-grid cell lower corners, sizes, and level ids."""
    cell_size = ((hi - lo) / np.array([nx, ny, nz], dtype=np.float64)).astype(dtype)
    xs = lo[0] + np.arange(nx, dtype=np.float64) * cell_size[0]
    ys = lo[1] + np.arange(ny, dtype=np.float64) * cell_size[1]
    zs = lo[2] + np.arange(nz, dtype=np.float64) * cell_size[2]
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    lows = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(dtype, copy=False)
    sizes = np.broadcast_to(cell_size, lows.shape).copy()
    levels = np.zeros(lows.shape[0], dtype=np.int16)
    return lows, sizes, levels


def _subdivide_cells(lows: np.ndarray,
                     sizes: np.ndarray,
                     levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each parent hex cell into eight children."""
    offsets = _CHILD_OFFSETS.astype(lows.dtype, copy=False)
    half_sizes = sizes * np.asarray(0.5, dtype=sizes.dtype)
    child_sizes = np.repeat(half_sizes, 8, axis=0)
    child_lows = np.repeat(lows, 8, axis=0) + child_sizes * np.tile(offsets, (lows.shape[0], 1))
    child_levels = np.repeat(levels + 1, 8)
    return child_lows, child_sizes, child_levels


def _cell_sample_points(lows: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return center plus corner samples for each cell."""
    offsets = _ADAPTIVE_SAMPLE_OFFSETS.astype(lows.dtype, copy=False)
    sample_count = offsets.shape[0]
    return (
        np.repeat(lows, sample_count, axis=0)
        + np.repeat(sizes, sample_count, axis=0) * np.tile(offsets, (lows.shape[0], 1))
    )


def _refine_adaptive_cells(lows: np.ndarray,
                           sizes: np.ndarray,
                           levels: np.ndarray,
                           snapshots: List[YarnSnapshot],
                           adaptive_levels: int,
                           chunk_voxels: int,
                           workers: Optional[int],
                           max_adaptive_cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine cells whose center/corners do not agree on yarn ownership."""
    sample_count = _ADAPTIVE_SAMPLE_OFFSETS.shape[0]
    cell_chunk = max(1, chunk_voxels // sample_count)

    for _ in range(adaptive_levels):
        refine_parts = []
        for c0 in range(0, lows.shape[0], cell_chunk):
            c1 = min(c0 + cell_chunk, lows.shape[0])
            samples = _cell_sample_points(lows[c0:c1], sizes[c0:c1])
            sample_ids = _classify_voxels_numpy(
                samples, snapshots, chunk=chunk_voxels, workers=workers
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
                                   workers: Optional[int]) -> np.ndarray:
    """Classify adaptive leaf cells by their centers."""
    centers = lows + sizes * np.asarray(0.5, dtype=sizes.dtype)
    return _classify_voxels_numpy(centers, snapshots, chunk=chunk_voxels, workers=workers)


# ---------------------------------------------------------------------------
# Abaqus .inp writer (hex elements, per-element yarn index).
# ---------------------------------------------------------------------------


def _write_inp(path: Path, lo, hi, nx, ny, nz, yarn_id: np.ndarray,
               textile_name: str = "TexGenPython"):
    """Write an Abaqus .inp with structured hex mesh + per-element ELSETs."""
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz

    nnx, nny, nnz = nx + 1, ny + 1, nz + 1

    def nid(ix, iy, iz):
        return 1 + ix + iy * nnx + iz * nnx * nny

    with open(path, "w") as f:
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
                    f.write(f"{nid(ix,iy,iz)}, {x:.6g}, {y:.6g}, {z:.6g}\n")

        f.write("*Element, type=C3D8R\n")
        eid = 0
        elems_per_yarn = {}
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
                    f.write(f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")
                    yidx = int(yarn_id[eid - 1])
                    elems_per_yarn.setdefault(yidx, []).append(eid)

        # ELSETs per yarn (including -1 = matrix).
        for yidx, ids in elems_per_yarn.items():
            name = "Matrix" if yidx < 0 else f"Yarn{yidx}"
            f.write(f"*Elset, elset={name}\n")
            # Abaqus: 16 ids per line.
            for i in range(0, len(ids), 16):
                f.write(", ".join(str(e) for e in ids[i:i+16]) + ",\n")

        f.write("*End Part\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=TexGenInstance, part=TexGenPart\n*End Instance\n")
        f.write("*End Assembly\n")


def _write_adaptive_inp(path: Path,
                        cells: AdaptiveVoxelCells,
                        textile_name: str = "TexGenAdaptivePython") -> dict:
    """Write non-uniform adaptive hex cells as an Abaqus input deck."""
    node_offsets = _HEX_NODE_OFFSETS.astype(cells.lows.dtype, copy=False)
    node_ids = {}
    node_coords = []
    elem_nodes = []

    for low, size in zip(cells.lows, cells.sizes):
        conn = []
        for offset in node_offsets:
            coord = low + size * offset
            key = tuple(np.round(coord.astype(np.float64), 12))
            node_id = node_ids.get(key)
            if node_id is None:
                node_id = len(node_coords) + 1
                node_ids[key] = node_id
                node_coords.append(coord.astype(np.float64, copy=False))
            conn.append(node_id)
        elem_nodes.append(conn)

    with open(path, "w") as f:
        f.write("*Heading\n")
        f.write(f"TexGen Python adaptive voxel mesh: {textile_name}\n")
        f.write("** Lightweight linear-octree mesh generated by numpy backend.\n")
        f.write("** Hanging-node constraints and p4est-style 2:1 balancing are not generated.\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("**\n*Part, name=TexGenPart\n*Node\n")
        for node_id, coord in enumerate(node_coords, start=1):
            f.write(f"{node_id}, {coord[0]:.6g}, {coord[1]:.6g}, {coord[2]:.6g}\n")

        f.write("*Element, type=C3D8R\n")
        elems_per_yarn = {}
        for elem_id, conn in enumerate(elem_nodes, start=1):
            f.write(f"{elem_id}, " + ", ".join(str(node_id) for node_id in conn) + "\n")
            yidx = int(cells.yarn_id[elem_id - 1])
            elems_per_yarn.setdefault(yidx, []).append(elem_id)

        for yidx, ids in elems_per_yarn.items():
            name = "Matrix" if yidx < 0 else f"Yarn{yidx}"
            f.write(f"*Elset, elset={name}\n")
            for i in range(0, len(ids), 16):
                f.write(", ".join(str(e) for e in ids[i:i+16]) + ",\n")

        f.write("*End Part\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=TexGenInstance, part=TexGenPart\n*End Instance\n")
        f.write("*End Assembly\n")

    return dict(nodes=len(node_coords), elements=len(elem_nodes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
                     max_adaptive_cells: int = 2_000_000) -> dict:
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

    Returns
    -------
    dict with ``yarn_id`` (np.ndarray of shape (nx*ny*nz,), row-major ix+iy*nx+iz*nx*ny order),
    ``aabb`` (2x3), backend/device, and timing info.
    """
    backend = backend.lower()
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
        backend_device = device
        workers_used = 1
        torch_dtype = {"float32": torch_mod.float32, "float64": torch_mod.float64}[dtype]
    else:
        backend_device = "cpu"
        workers_used = _default_numpy_workers() if workers is None else workers
        if workers_used < 1:
            raise ValueError("workers must be >= 1")
        np_dtype = {"float32": np.float32, "float64": np.float64}[dtype]

    def log(msg):
        if verbose:
            print(f"[voxelizer] {msg}")

    t0 = time.perf_counter()
    snapshots, aabb = extract_snapshots(textile)
    t_extract = time.perf_counter() - t0
    log(
        f"extracted {len(snapshots)} yarns, AABB={aabb.tolist()}, "
        f"backend={backend}, workers={workers_used}, {t_extract:.3f}s"
    )

    if len(snapshots) == 0:
        raise RuntimeError("No yarns extracted - textile may be empty or unbuilt")

    lo, hi = aabb[0], aabb[1]

    if adaptive:
        t0 = time.perf_counter()
        snapshots_np = _snapshots_as_dtype(snapshots, np_dtype)
        lows, sizes, levels = _structured_cell_lows_sizes(lo, hi, nx, ny, nz, np_dtype)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        lows, sizes, levels = _refine_adaptive_cells(
            lows, sizes, levels, snapshots_np, adaptive_levels, chunk_voxels,
            workers_used, max_adaptive_cells
        )
        t_refine = time.perf_counter() - t0
        log(
            f"adaptive mesh has {lows.shape[0]:,} cells after {adaptive_levels} "
            f"level(s), max level={int(levels.max()) if len(levels) else 0}, {t_refine:.3f}s"
        )

        t0 = time.perf_counter()
        yarn_id = _classify_adaptive_cells_numpy(
            lows, sizes, snapshots_np, chunk_voxels, workers_used
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {lows.shape[0]:,} adaptive cells with numpy/"
            f"{workers_used} workers in {t_classify:.3f}s"
        )

        t0 = time.perf_counter()
        out_path = Path(out_inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells = AdaptiveVoxelCells(lows=lows, sizes=sizes, levels=levels, yarn_id=yarn_id)
        mesh_counts = _write_adaptive_inp(
            out_path, cells, textile_name=getattr(textile, "GetName", lambda: "Textile")()
        )
        t_write = time.perf_counter() - t0
        log(
            f"wrote {out_path} ({mesh_counts['elements']:,} elements, "
            f"{mesh_counts['nodes']:,} nodes, {t_write:.3f}s)"
        )

        return dict(
            yarn_id=yarn_id,
            aabb=aabb,
            backend=backend,
            device=backend_device,
            workers=workers_used,
            adaptive=True,
            levels=levels,
            num_cells=int(lows.shape[0]),
            mesh=mesh_counts,
            timings=dict(
                extract=t_extract, pack=t_pack, refine=t_refine,
                classify=t_classify, write=t_write
            ),
        )

    # Build voxel centers (row-major: ix varies fastest).
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz
    xs = lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    ys = lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    zs = lo[2] + (np.arange(nz, dtype=np.float64) + 0.5) * dz
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")  # outer-to-inner: z,y,x
    centers_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    t0 = time.perf_counter()
    if backend == "torch":
        packed = _pack_yarns(snapshots, device=device, dtype=torch_dtype)
        centers = torch_mod.from_numpy(centers_np).to(device=device, dtype=torch_dtype)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        yarn_id_tensor = _classify_voxels_torch(centers, packed, chunk=chunk_voxels)
        if device == "cuda":
            torch_mod.cuda.synchronize()
        t_classify = time.perf_counter() - t0
        log(f"classified {centers.shape[0]:,} voxels with torch/{device} in {t_classify:.3f}s")
        yarn_id = yarn_id_tensor.detach().cpu().numpy()
    else:
        snapshots_np = _snapshots_as_dtype(snapshots, np_dtype)
        centers_np = centers_np.astype(np_dtype, copy=False)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        yarn_id = _classify_voxels_numpy(
            centers_np, snapshots_np, chunk=chunk_voxels, workers=workers_used
        )
        t_classify = time.perf_counter() - t0
        log(
            f"classified {centers_np.shape[0]:,} voxels with numpy/"
            f"{workers_used} workers in {t_classify:.3f}s"
        )

    t0 = time.perf_counter()
    out_path = Path(out_inp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_inp(out_path, lo, hi, nx, ny, nz, yarn_id,
               textile_name=getattr(textile, "GetName", lambda: "Textile")())
    t_write = time.perf_counter() - t0
    log(f"wrote {out_path} ({t_write:.3f}s)")

    return dict(
        yarn_id=yarn_id, aabb=aabb, backend=backend, device=backend_device, workers=workers_used,
        adaptive=False,
        timings=dict(extract=t_extract, pack=t_pack, classify=t_classify, write=t_write),
    )


__all__ = ["voxelize_textile", "extract_snapshots", "YarnSnapshot", "AdaptiveVoxelCells"]
