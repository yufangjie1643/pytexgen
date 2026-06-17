"""Export helpers for modern voxel data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_inp_from_voxel_data(
    data,
    path,
    *,
    textile_name: str = "ModernTextile",
    progress=False,
) -> Path:
    """Write structured ``VoxelGridData`` to an Abaqus ``.inp`` file.

    Element node order follows TexGen 3.13.1 ``CVoxelMesh::OutputHexElements``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    numpy_data = data.to_numpy() if hasattr(data, "to_numpy") else data
    nx, ny, nz = numpy_data.resolution
    lo = np.asarray(numpy_data.aabb[0], dtype=np.float64)
    hi = np.asarray(numpy_data.aabb[1], dtype=np.float64)
    yarn_id = np.asarray(numpy_data.yarn_id)
    _write_structured_legacy_inp(out, lo, hi, nx, ny, nz, yarn_id, textile_name, progress)
    return out


def _write_structured_legacy_inp(
    path: Path,
    lo: np.ndarray,
    hi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    yarn_id: np.ndarray,
    textile_name: str,
    progress=False,
) -> None:
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    dz = (hi[2] - lo[2]) / nz
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1

    def nid(ix: int, iy: int, iz: int) -> int:
        return 1 + ix + iy * nnx + iz * nnx * nny

    def progress_iter(iterable, **_kwargs):
        if callable(progress):
            return progress(iterable, **_kwargs)
        return iterable

    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("*Heading\n")
        out.write(f"TexGen modern voxel mesh: {textile_name}\n")
        out.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        out.write("**\n*Part, name=TexGenPart\n*Node\n")

        for iz in progress_iter(range(nnz), total=nnz, desc="write nodes", unit="z-slice"):
            for iy in range(nny):
                for ix in range(nnx):
                    x = lo[0] + ix * dx
                    y = lo[1] + iy * dy
                    z = lo[2] + iz * dz
                    out.write(f"{nid(ix, iy, iz)}, {x:.6g}, {y:.6g}, {z:.6g}\n")

        out.write("*Element, type=C3D8R\n")
        eid = 0
        for iz in progress_iter(range(nz), total=nz, desc="write elements", unit="z-slice"):
            for iy in range(ny):
                for ix in range(nx):
                    eid += 1
                    nodes = (
                        nid(ix + 1, iy, iz),
                        nid(ix + 1, iy + 1, iz),
                        nid(ix, iy + 1, iz),
                        nid(ix, iy, iz),
                        nid(ix + 1, iy, iz + 1),
                        nid(ix + 1, iy + 1, iz + 1),
                        nid(ix, iy + 1, iz + 1),
                        nid(ix, iy, iz + 1),
                    )
                    out.write(f"{eid}, " + ", ".join(str(node) for node in nodes) + "\n")

        unique_yarns = np.unique(yarn_id)
        for yidx in progress_iter(
            unique_yarns,
            total=len(unique_yarns),
            desc="write element sets",
            unit="set",
        ):
            ids = np.nonzero(yarn_id == yidx)[0] + 1
            name = "Matrix" if yidx < 0 else f"Yarn{int(yidx)}"
            out.write(f"*Elset, elset={name}\n")
            for start in range(0, len(ids), 16):
                out.write(", ".join(str(int(eid)) for eid in ids[start:start + 16]) + ",\n")

        out.write("*End Part\n*Assembly, name=Assembly\n")
        out.write("*Instance, name=TexGenInstance, part=TexGenPart\n*End Instance\n")
        out.write("*End Assembly\n")
