"""Build a 2x2 2D weave, convert yarn volume cells to tetrahedra, and render it.

Outputs:
    build/tetgen_2d_weave_tetra/weave_2x2_tet.inp
    build/tetgen_2d_weave_tetra/weave_2x2_tet.png

Run after installing pytexgen:
    uv run python script/tetgen_2d_weave_tetra.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pytexgen import CMesh, CTextileWeave2D


def build_plain_weave(resolution: int) -> CTextileWeave2D:
    textile = CTextileWeave2D(2, 2, 1.0, 0.2, True)
    textile.SwapPosition(0, 1)
    textile.SwapPosition(1, 0)
    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(resolution)
    textile.AssignDefaultDomain()
    return textile


def signed_tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def tetrahedralize_textile(textile: CTextileWeave2D) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mesh = CMesh()
    textile.AddVolumeToMesh(mesh)
    source_wedges = len(list(mesh.GetIndices(CMesh.WEDGE))) // CMesh.GetNumNodes(CMesh.WEDGE)
    source_hexes = len(list(mesh.GetIndices(CMesh.HEX))) // CMesh.GetNumNodes(CMesh.HEX)

    mesh.ConvertToTetMesh()

    nodes = np.array(
        [[mesh.GetNode(i).x, mesh.GetNode(i).y, mesh.GetNode(i).z] for i in range(mesh.GetNumNodes())],
        dtype=np.float64,
    )
    tet_indices = np.array(list(mesh.GetIndices(CMesh.TET)), dtype=np.int64).reshape(-1, 4)

    volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(nodes[tet_indices[:, 1]] - nodes[tet_indices[:, 0]], nodes[tet_indices[:, 2]] - nodes[tet_indices[:, 0]]),
            nodes[tet_indices[:, 3]] - nodes[tet_indices[:, 0]],
        )
        / 6.0
    )
    duplicate_node_tets = int(sum(len(set(tet.tolist())) != 4 for tet in tet_indices))
    zero_volume_tets = int(np.count_nonzero(volumes <= 1e-14))

    stats = {
        "source_wedges": float(source_wedges),
        "source_hexes": float(source_hexes),
        "nodes": float(nodes.shape[0]),
        "tet_elements": float(tet_indices.shape[0]),
        "min_volume": float(volumes.min()) if volumes.size else math.nan,
        "max_volume": float(volumes.max()) if volumes.size else math.nan,
        "sum_abs_volume": float(volumes.sum()),
        "duplicate_node_tets": float(duplicate_node_tets),
        "zero_volume_tets": float(zero_volume_tets),
    }
    return nodes, tet_indices, stats


def write_abaqus_tet_inp(path: Path, nodes: np.ndarray, tet_indices: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("*Heading\n")
        fh.write("** 2x2 TexGen plain weave converted to C3D4 tetrahedra\n")
        fh.write("*Node\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            fh.write(f"{i}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        fh.write("*Element, type=C3D4, elset=YARNS\n")
        for eid, tet in enumerate(tet_indices, start=1):
            n1, n2, n3, n4 = (int(v) + 1 for v in tet)
            fh.write(f"{eid}, {n1}, {n2}, {n3}, {n4}\n")
        fh.write("*Elset, elset=YARNS, generate\n")
        fh.write(f"1, {len(tet_indices)}, 1\n")


def external_tet_faces(tet_indices: np.ndarray) -> list[tuple[int, int, int]]:
    local_faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in tet_indices:
        for face in local_faces:
            oriented = tuple(int(tet[i]) for i in face)
            key = tuple(sorted(oriented))
            face_map.setdefault(key, []).append(oriented)
    return [faces[0] for faces in face_map.values() if len(faces) == 1]


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def render_tet_surface(path: Path, nodes: np.ndarray, tet_indices: np.ndarray, elev: float, azim: float) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the PNG preview. "
            "Run build.ps1/build.bat/build.sh or install it with `uv pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = external_tet_faces(tet_indices)
    polygons = [nodes[list(face)] for face in faces]

    fig = plt.figure(figsize=(10, 7), dpi=220, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    collection = Poly3DCollection(
        polygons,
        facecolors="#7aa6c2",
        edgecolors="#202020",
        linewidths=0.08,
        alpha=0.92,
    )
    ax.add_collection3d(collection)
    set_axes_equal(ax, nodes)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 2x2 2D weave tetra mesh and PNG preview.")
    parser.add_argument("--output-dir", default="build/tetgen_2d_weave_tetra")
    parser.add_argument("--prefix", default="weave_2x2_tet")
    parser.add_argument("--resolution", type=int, default=8, help="TexGen yarn section resolution.")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    textile = build_plain_weave(args.resolution)
    nodes, tet_indices, stats = tetrahedralize_textile(textile)

    inp_path = out_dir / f"{args.prefix}.inp"
    png_path = out_dir / f"{args.prefix}.png"
    write_abaqus_tet_inp(inp_path, nodes, tet_indices)
    render_tet_surface(png_path, nodes, tet_indices, elev=args.elev, azim=args.azim)

    print("2x2 weave tetra mesh generated")
    print(f"  source wedges: {int(stats['source_wedges'])}")
    print(f"  source hexes: {int(stats['source_hexes'])}")
    print(f"  nodes: {int(stats['nodes'])}")
    print(f"  tet elements: {int(stats['tet_elements'])}")
    print(f"  min tet volume: {stats['min_volume']:.12g}")
    print(f"  zero volume tets <= 1e-14: {int(stats['zero_volume_tets'])}")
    print(f"  wrote: {inp_path}")
    print(f"  wrote: {png_path}")


if __name__ == "__main__":
    main()
