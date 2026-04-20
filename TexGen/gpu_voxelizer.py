"""
GPU-accelerated voxelization bypass for TexGen.

Drop-in replacement for ``CRectangularVoxelMesh.SaveVoxelMesh(...)`` that:
  1. Takes a fully-built ``CTextile`` from TexGen (all refine / interference /
     section-mesh work already done by TexGen's C++ core).
  2. Snapshots each yarn's slave-node frame + section polygon into plain arrays.
  3. Classifies every voxel center by "point-in-swept-polygon" test on the GPU.
  4. Writes an Abaqus ``.inp`` file compatible with TexGen's own output format
     (hex elements + per-element yarn index).

Design goals:
  * TexGen source code is NOT modified - this is a pure Python bypass.
  * Works with any ``CTextile`` subclass (2D/3D/sheared/orthogonal/...).
  * Torch backend: runs on CUDA, Metal (MPS), or CPU. Swap in a Triton kernel
    later by replacing ``_classify_voxels_torch``.

Usage:
    from TexGen.Core import *
    from TexGen.gpu_voxelizer import voxelize_textile

    T = CShearedTextileWeave2D(3,3,5.0,2.0,0.2618,True,True)
    T.SetYarnWidths(2.0); T.SetYarnHeights(0.8); T.AssignDefaultDomain()

    voxelize_textile(T, nx=64, ny=64, nz=64, out_inp="out.inp")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "gpu_voxelizer requires PyTorch. Install with:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
    ) from e

from TexGen.Core import CTextile, CYarn  # type: ignore

# BUILD_TYPE bitmask constants from CYarn. SWIG exposes them as CYarn.SURFACE etc.
# Fallback to raw values if the enum binding differs.
try:
    _LINE    = CYarn.LINE
    _SURFACE = CYarn.SURFACE
    _VOLUME  = CYarn.VOLUME
except AttributeError:
    _LINE, _SURFACE, _VOLUME = 1 << 0, 1 << 1, 1 << 2


# ---------------------------------------------------------------------------
# Geometry snapshot: extract plain numpy arrays from TexGen's C++ objects.
# ---------------------------------------------------------------------------


@dataclass
class YarnSnapshot:
    """Per-yarn GPU-friendly geometry snapshot."""
    positions: np.ndarray      # (M, 3) slave node world positions
    tangents:  np.ndarray      # (M, 3) unit tangent along yarn length
    ups:       np.ndarray      # (M, 3) unit up (perpendicular to tangent)
    sides:     np.ndarray      # (M, 3) unit side = tangent x up (right-handed frame)
    section:   np.ndarray      # (N, 2) 2D polygon (u=side, v=up) at each slave node
    translations: np.ndarray   # (K, 3) periodic-image translations (includes origin)


def _xyz(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _xy(v) -> np.ndarray:
    return np.array([v.x, v.y], dtype=np.float64)


def _extract_yarn(yarn: CYarn, translations_xyz) -> Optional[YarnSnapshot]:
    """Pull slave-node frame + 2D section polygon out of a fully-built CYarn."""
    # Force all geometry to be built - idempotent if already built.
    yarn.BuildYarnIfNeeded(_LINE | _SURFACE | _VOLUME)

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
# GPU classification: for every voxel center, find owning yarn.
# ---------------------------------------------------------------------------


def _pack_yarns(snapshots: List[YarnSnapshot], device, dtype=torch.float32):
    """Pack yarn arrays into padded tensors for batched GPU kernels."""
    num_yarns = len(snapshots)
    M_max = max(s.positions.shape[0] for s in snapshots)
    N_max = max(s.section.shape[0] for s in snapshots)
    K_max = max(s.translations.shape[0] for s in snapshots)

    P = torch.zeros((num_yarns, M_max, 3), device=device, dtype=dtype)
    T = torch.zeros_like(P)
    U = torch.zeros_like(P)
    S = torch.zeros_like(P)
    M_len = torch.zeros(num_yarns, device=device, dtype=torch.int32)
    Sec = torch.zeros((num_yarns, N_max, 2), device=device, dtype=dtype)
    N_len = torch.zeros(num_yarns, device=device, dtype=torch.int32)
    Tr = torch.zeros((num_yarns, K_max, 3), device=device, dtype=dtype)
    K_len = torch.zeros(num_yarns, device=device, dtype=torch.int32)

    for i, s in enumerate(snapshots):
        m = s.positions.shape[0]
        P[i, :m] = torch.from_numpy(s.positions).to(device=device, dtype=dtype)
        T[i, :m] = torch.from_numpy(s.tangents).to(device=device, dtype=dtype)
        U[i, :m] = torch.from_numpy(s.ups).to(device=device, dtype=dtype)
        S[i, :m] = torch.from_numpy(s.sides).to(device=device, dtype=dtype)
        M_len[i] = m
        n = s.section.shape[0]
        Sec[i, :n] = torch.from_numpy(s.section).to(device=device, dtype=dtype)
        N_len[i] = n
        k = s.translations.shape[0]
        Tr[i, :k] = torch.from_numpy(s.translations).to(device=device, dtype=dtype)
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
    poly = polygon[:poly_len]                          # (N, 2)
    p_next = torch.roll(poly, shifts=-1, dims=0)       # (N, 2)
    # Broadcast: points (..., 1, 2)  vs edges (N, 2)
    u = points_uv[..., None, 0]                        # (..., 1)
    v = points_uv[..., None, 1]
    x1 = poly[:, 0]; y1 = poly[:, 1]
    x2 = p_next[:, 0]; y2 = p_next[:, 1]
    # Crossing test: edge straddles v and ray going +u crosses it.
    cond1 = (y1 > v) != (y2 > v)
    # x-intercept of edge at height v
    denom = (y2 - y1)
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
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
    device = centers.device
    V = centers.shape[0]
    yarn_id = torch.full((V,), -1, device=device, dtype=torch.int32)

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
        best_dist = torch.full((C,), float("inf"), device=device)
        best_yarn = torch.full((C,), -1, device=device, dtype=torch.int32)

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
                uv = torch.stack([u_coord, v_coord], dim=-1)  # (C, 2)
                inside = _point_in_polygon_batch(uv, poly, n)

                # Rough "depth" proxy for overlap resolution: use d2 (Euclidean
                # to closest slave node). Real surface distance would require
                # signed distance to polygon edge; this proxy is sufficient for
                # consistent overlap assignment and matches TexGen's behaviour
                # where the yarn with the closest section wins.
                dist = torch.sqrt(d2.gather(1, nn[:, None]).squeeze(-1))
                # Also penalise large longitudinal offset (out-of-section slab).
                dist = dist + t_coord.abs() * 0.1

                upd = inside & (dist < best_dist)
                best_dist = torch.where(upd, dist, best_dist)
                best_yarn = torch.where(upd, torch.full_like(best_yarn, y_idx), best_yarn)

        yarn_id[v0:v1] = best_yarn

    return yarn_id


# ---------------------------------------------------------------------------
# Abaqus .inp writer (hex elements, per-element yarn index).
# ---------------------------------------------------------------------------


def _write_inp(path: Path, lo, hi, nx, ny, nz, yarn_id: np.ndarray,
               textile_name: str = "TexGenGPU"):
    """Write an Abaqus .inp with structured hex mesh + per-element ELSETs."""
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz

    nnx, nny, nnz = nx + 1, ny + 1, nz + 1

    def nid(ix, iy, iz):
        return 1 + ix + iy * nnx + iz * nnx * nny

    with open(path, "w") as f:
        f.write("*Heading\n")
        f.write(f"TexGen GPU voxel mesh: {textile_name}\n")
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def voxelize_textile(textile: CTextile,
                     nx: int = 64, ny: int = 64, nz: int = 64,
                     out_inp: str = "out.inp",
                     device: Optional[str] = None,
                     dtype: str = "float32",
                     chunk_voxels: int = 65536,
                     verbose: bool = True) -> dict:
    """GPU-voxelize a built CTextile and write an Abaqus .inp.

    Parameters
    ----------
    textile : CTextile
        A fully built textile (all section/refine work done by TexGen).
    nx, ny, nz : int
        Voxel resolution along each axis of the domain AABB.
    out_inp : str
        Output Abaqus input deck path.
    device : {"cuda", "mps", "cpu", None}
        None picks the best available.
    dtype : {"float32", "float64"}
        Precision on GPU. float32 is plenty for voxelization.
    chunk_voxels : int
        Voxels processed per batch (controls VRAM).
    verbose : bool
        Print per-phase timing.

    Returns
    -------
    dict with ``yarn_id`` (np.ndarray of shape (nx*ny*nz,), row-major ix+iy*nx+iz*nx*ny order),
    ``aabb`` (2x3), and timing info.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]

    def log(msg):
        if verbose:
            print(f"[gpu_voxelizer] {msg}")

    t0 = time.perf_counter()
    snapshots, aabb = extract_snapshots(textile)
    t_extract = time.perf_counter() - t0
    log(f"extracted {len(snapshots)} yarns, AABB={aabb.tolist()}, {t_extract:.3f}s")

    if len(snapshots) == 0:
        raise RuntimeError("No yarns extracted - textile may be empty or unbuilt")

    t0 = time.perf_counter()
    packed = _pack_yarns(snapshots, device=device, dtype=torch_dtype)
    t_pack = time.perf_counter() - t0

    # Build voxel centers (row-major: ix varies fastest).
    lo, hi = aabb[0], aabb[1]
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz
    xs = lo[0] + (np.arange(nx) + 0.5) * dx
    ys = lo[1] + (np.arange(ny) + 0.5) * dy
    zs = lo[2] + (np.arange(nz) + 0.5) * dz
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")  # outer-to-inner: z,y,x
    centers_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    centers = torch.from_numpy(centers_np).to(device=device, dtype=torch_dtype)

    t0 = time.perf_counter()
    yarn_id_gpu = _classify_voxels_torch(centers, packed, chunk=chunk_voxels)
    if device == "cuda":
        torch.cuda.synchronize()
    t_classify = time.perf_counter() - t0
    log(f"classified {centers.shape[0]:,} voxels on {device} in {t_classify:.3f}s")

    yarn_id = yarn_id_gpu.detach().cpu().numpy()

    t0 = time.perf_counter()
    out_path = Path(out_inp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_inp(out_path, lo, hi, nx, ny, nz, yarn_id,
               textile_name=getattr(textile, "GetName", lambda: "Textile")())
    t_write = time.perf_counter() - t0
    log(f"wrote {out_path} ({t_write:.3f}s)")

    return dict(
        yarn_id=yarn_id, aabb=aabb, device=device,
        timings=dict(extract=t_extract, pack=t_pack, classify=t_classify, write=t_write),
    )


__all__ = ["voxelize_textile", "extract_snapshots", "YarnSnapshot"]
