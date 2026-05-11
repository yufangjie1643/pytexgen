"""Small numpy/scipy FEM solve for the 2x2 weave C3D4 tetra mesh.

The default path generates the mesh in memory using ``tetgen_2d_weave_tetra.py``,
assembles a homogeneous isotropic linear-elastic stiffness matrix, applies a
small prescribed axial strain on boundary nodes, and solves with scipy sparse.

Run:
    uv run python script/tet_fem_solve.py
    uv run python script/tet_fem_solve.py --inp build/tetgen_2d_weave_tetra/weave_2x2_tet.inp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from tetgen_2d_weave_tetra import (
    build_plain_weave,
    external_tet_faces,
    render_tet_surface,
    set_axes_equal,
    tetrahedralize_textile,
    write_abaqus_tet_inp,
)


def read_abaqus_c3d4(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nodes_by_label: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, int, int, int]] = []
    section = None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        lower = line.lower()
        if lower.startswith("*node"):
            section = "node"
            continue
        if lower.startswith("*element"):
            section = "element" if "c3d4" in lower else None
            continue
        if lower.startswith("*"):
            section = None
            continue

        parts = [part.strip() for part in line.split(",") if part.strip()]
        if section == "node" and len(parts) >= 4:
            nodes_by_label[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif section == "element" and len(parts) >= 5:
            elements.append(tuple(int(value) for value in parts[1:5]))

    labels = sorted(nodes_by_label)
    label_to_index = {label: i for i, label in enumerate(labels)}
    nodes = np.array([nodes_by_label[label] for label in labels], dtype=np.float64)
    tets = np.array([[label_to_index[label] for label in element] for element in elements], dtype=np.int64)
    if nodes.size == 0 or tets.size == 0:
        raise RuntimeError(f"No C3D4 tetra mesh found in {path}")
    return nodes, tets


def elasticity_matrix(young: float, poisson: float) -> np.ndarray:
    if not (-1.0 < poisson < 0.5):
        raise ValueError("Poisson ratio must be in (-1, 0.5)")
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d = np.zeros((6, 6), dtype=np.float64)
    d[:3, :3] = lam
    d[0, 0] += 2.0 * mu
    d[1, 1] += 2.0 * mu
    d[2, 2] += 2.0 * mu
    d[3, 3] = mu
    d[4, 4] = mu
    d[5, 5] = mu
    return d


def tet_b_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.column_stack((np.ones(4), coords))
    det = float(np.linalg.det(x))
    volume = abs(det) / 6.0
    if volume <= 1e-14:
        raise ValueError(f"Degenerate tetrahedron volume={volume:.3e}")

    inv_x = np.linalg.inv(x)
    gradients = inv_x[1:, :].T
    b = np.zeros((6, 12), dtype=np.float64)

    for i, (dndx, dndy, dndz) in enumerate(gradients):
        j = 3 * i
        b[0, j] = dndx
        b[1, j + 1] = dndy
        b[2, j + 2] = dndz
        b[3, j + 1] = dndz
        b[3, j + 2] = dndy
        b[4, j] = dndz
        b[4, j + 2] = dndx
        b[5, j] = dndy
        b[5, j + 1] = dndx
    return b, volume


def assemble_stiffness(nodes: np.ndarray, tets: np.ndarray, young: float, poisson: float):
    d = elasticity_matrix(young, poisson)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    volumes = np.empty(len(tets), dtype=np.float64)

    for e, tet in enumerate(tets):
        b, volume = tet_b_matrix(nodes[tet])
        ke = volume * (b.T @ d @ b)
        dofs = np.repeat(tet, 3) * 3 + np.tile(np.arange(3), 4)
        ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(ii.ravel().tolist())
        cols.extend(jj.ravel().tolist())
        values.extend(ke.ravel().tolist())
        volumes[e] = volume

    ndof = nodes.shape[0] * 3
    stiffness = coo_matrix((values, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return stiffness, volumes


def boundary_dofs(nodes: np.ndarray, strain: float) -> tuple[np.ndarray, np.ndarray]:
    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    tol = float(span.max() * 1e-8)

    on_boundary = np.any((nodes <= mins + tol) | (nodes >= maxs - tol), axis=1)
    prescribed_nodes = np.flatnonzero(on_boundary)
    prescribed_dofs = np.repeat(prescribed_nodes, 3) * 3 + np.tile(np.arange(3), len(prescribed_nodes))

    values = np.zeros_like(prescribed_dofs, dtype=np.float64)
    for i, node in enumerate(prescribed_nodes):
        base = 3 * i
        values[base] = strain * (nodes[node, 0] - mins[0])
        values[base + 1] = 0.0
        values[base + 2] = 0.0
    return prescribed_dofs, values


def solve_dirichlet(stiffness, prescribed_dofs: np.ndarray, prescribed_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ndof = stiffness.shape[0]
    force = np.zeros(ndof, dtype=np.float64)
    displacement = np.zeros(ndof, dtype=np.float64)
    displacement[prescribed_dofs] = prescribed_values

    fixed = np.zeros(ndof, dtype=bool)
    fixed[prescribed_dofs] = True
    free = np.flatnonzero(~fixed)

    rhs = force[free] - stiffness[free][:, prescribed_dofs] @ prescribed_values
    displacement[free] = spsolve(stiffness[free][:, free], rhs)
    reactions = stiffness @ displacement - force
    return displacement, reactions


def render_deformed(path: Path, nodes: np.ndarray, tets: np.ndarray, displacement: np.ndarray, scale: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    disp_nodes = displacement.reshape(-1, 3)
    deformed = nodes + scale * disp_nodes
    faces = external_tet_faces(tets)
    polygons = [deformed[list(face)] for face in faces]
    magnitudes = np.array([np.linalg.norm(disp_nodes[list(face)], axis=1).mean() for face in faces])

    fig = plt.figure(figsize=(10, 7), dpi=220, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    collection = Poly3DCollection(
        polygons,
        edgecolors="#202020",
        linewidths=0.06,
        alpha=0.95,
    )
    collection.set_array(magnitudes)
    collection.set_cmap("viridis")
    ax.add_collection3d(collection)
    set_axes_equal(ax, deformed)
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.colorbar(collection, ax=ax, shrink=0.62, pad=0.02, label="mean |u|")
    fig.tight_layout(pad=0)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a small C3D4 tetra FEM problem with numpy/scipy.")
    parser.add_argument("--inp", type=Path, default=None, help="Existing C3D4 Abaqus .inp. Omit to generate a 2x2 weave mesh.")
    parser.add_argument("--output-dir", type=Path, default=Path("build/tet_fem_solve"))
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--young", type=float, default=70e9)
    parser.add_argument("--poisson", type=float, default=0.25)
    parser.add_argument("--strain", type=float, default=1e-3)
    parser.add_argument("--deform-scale", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.inp:
        nodes, tets = read_abaqus_c3d4(args.inp)
        mesh_path = args.inp
    else:
        textile = build_plain_weave(args.resolution)
        nodes, tets, mesh_stats = tetrahedralize_textile(textile)
        mesh_path = args.output_dir / "weave_2x2_tet.inp"
        write_abaqus_tet_inp(mesh_path, nodes, tets)
        render_tet_surface(args.output_dir / "weave_2x2_tet.png", nodes, tets, elev=24.0, azim=-58.0)

    stiffness, volumes = assemble_stiffness(nodes, tets, young=args.young, poisson=args.poisson)
    prescribed_dofs, prescribed_values = boundary_dofs(nodes, args.strain)
    displacement, reactions = solve_dirichlet(stiffness, prescribed_dofs, prescribed_values)

    disp_nodes = displacement.reshape(-1, 3)
    x = nodes[:, 0]
    xmax_nodes = np.flatnonzero(np.isclose(x, x.max(), atol=(np.ptp(nodes, axis=0).max() * 1e-8)))
    reaction_x = float(reactions[xmax_nodes * 3].sum())
    strain_energy = float(0.5 * displacement @ (stiffness @ displacement))

    npz_path = args.output_dir / "fem_result.npz"
    png_path = args.output_dir / "fem_deformed.png"
    np.savez_compressed(
        npz_path,
        nodes=nodes,
        tets=tets,
        displacement=disp_nodes,
        reactions=reactions.reshape(-1, 3),
        volumes=volumes,
        young=args.young,
        poisson=args.poisson,
        strain=args.strain,
    )
    render_deformed(png_path, nodes, tets, displacement, scale=args.deform_scale)

    summary = {
        "mesh": str(mesh_path),
        "nodes": int(nodes.shape[0]),
        "tet_elements": int(tets.shape[0]),
        "dofs": int(stiffness.shape[0]),
        "free_dofs": int(stiffness.shape[0] - len(prescribed_dofs)),
        "prescribed_dofs": int(len(prescribed_dofs)),
        "min_volume": float(volumes.min()),
        "zero_volume_tets_le_1e-14": int(np.count_nonzero(volumes <= 1e-14)),
        "max_displacement": float(np.linalg.norm(disp_nodes, axis=1).max()),
        "right_face_reaction_x": reaction_x,
        "strain_energy": strain_energy,
        "result_npz": str(npz_path),
        "deformed_png": str(png_path),
    }
    (args.output_dir / "fem_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("FEM solve completed")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
