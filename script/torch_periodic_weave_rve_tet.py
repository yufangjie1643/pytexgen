"""Generate a pure Python/torch periodic 2x2 plain-weave tetra RVE.

This script does not call TexGen's C++ mesher or TetGen. It builds a periodic
background tetrahedral lattice, cuts it with a torch implicit yarn model, and
exports Abaqus C3D4 mesh data.

Run:
    .venv/Scripts/python.exe script/torch_periodic_weave_rve_tet.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from TexGen.torch_periodic_mesher import (
        build_plain_weave_rve,
        mesh_quality_summary,
        periodic_node_pairs,
        write_abaqus_inp,
        write_pbc_pairs_csv,
    )
except ImportError:
    from pytexgen.torch_periodic_mesher import (
        build_plain_weave_rve,
        mesh_quality_summary,
        periodic_node_pairs,
        write_abaqus_inp,
        write_pbc_pairs_csv,
    )


def external_faces(elements: np.ndarray) -> list[tuple[int, int, int]]:
    local_faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for tet in elements:
        for face in local_faces:
            oriented = tuple(int(tet[i]) for i in face)
            face_map[tuple(sorted(oriented))].append(oriented)
    return [faces[0] for faces in face_map.values() if len(faces) == 1]


def render_material_preview(mesh, path: Path) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for PNG preview rendering") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    nodes = mesh.nodes.detach().cpu().numpy()
    elements = mesh.elements.detach().cpu().numpy()
    material_ids = mesh.material_ids.detach().cpu().numpy()
    colors = {
        0: "#d8d8d8",
        1: "#cc4c37",
        2: "#2f7fb8",
        3: "#4a9b57",
        4: "#b77925",
    }
    alphas = {0: 0.12, 1: 0.82, 2: 0.82, 3: 0.82, 4: 0.82}

    fig = plt.figure(figsize=(10, 7), dpi=240, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    for mat in sorted(set(int(v) for v in material_ids.tolist())):
        mat_elements = elements[material_ids == mat]
        faces = external_faces(mat_elements)
        polygons = [[nodes[index] for index in face] for face in faces]
        if not polygons:
            continue
        ax.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=colors.get(mat, "#7aa6c2"),
                edgecolors="#202020" if mat else "#8a8a8a",
                linewidths=0.025 if mat else 0.02,
                alpha=alphas.get(mat, 0.75),
            )
        )

    edge_set = set()
    for tet in elements:
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            edge_set.add(tuple(sorted((int(tet[a]), int(tet[b])))))
    edge_segments = [[nodes[i], nodes[j]] for i, j in sorted(edge_set)]
    ax.add_collection3d(Line3DCollection(edge_segments, colors="#222222", linewidths=0.01, alpha=0.10))

    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except AttributeError:
        pass
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a torch periodic 2x2 plain-weave tetra RVE.")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("build/torch_periodic_weave_rve"))
    parser.add_argument("--prefix", default="torch_weave_2x2_rve")
    parser.add_argument("--gap-size", type=float, default=0.0)
    parser.add_argument("--density-mode", choices=("uniform", "topology"), default="uniform")
    parser.add_argument("--topology-interface-levels", type=int, default=1)
    parser.add_argument("--topology-crossing-levels", type=int, default=1)
    parser.add_argument("--topology-gap-levels", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_plain_weave_rve(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
        device=args.device,
        gap_size=args.gap_size,
        density_mode=args.density_mode,
        topology_interface_levels=args.topology_interface_levels,
        topology_crossing_levels=args.topology_crossing_levels,
        topology_gap_levels=args.topology_gap_levels,
    )
    pairs = periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))
    summary = mesh_quality_summary(mesh)
    summary["gap_size"] = args.gap_size
    summary["effective_undulation"] = mesh.metadata.get("effective_undulation")
    summary["density_mode"] = mesh.metadata.get("density_mode")
    summary["coordinate_counts"] = mesh.metadata.get("coordinate_counts")
    summary["pbc_pair_counts"] = {axis: len(axis_pairs) for axis, axis_pairs in pairs.items()}

    inp_path = args.output_dir / f"{args.prefix}.inp"
    csv_path = args.output_dir / f"{args.prefix}_pbc_pairs.csv"
    json_path = args.output_dir / f"{args.prefix}_quality.json"
    png_path = args.output_dir / f"{args.prefix}.png"

    write_abaqus_inp(mesh, inp_path)
    write_pbc_pairs_csv(pairs, csv_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_render:
        render_material_preview(mesh, png_path)

    print("torch periodic 2x2 RVE generated")
    print(f"  nodes: {summary['nodes']}")
    print(f"  elements: {summary['elements']}")
    print(f"  zero-volume tets <= 1e-14: {summary['zero_volume_tets_le_1e-14']}")
    print(f"  total volume: {summary['total_abs_volume']:.12g}")
    print(f"  bbox volume: {summary['bbox_volume']:.12g}")
    print(f"  gap size: {summary['gap_size']:.12g}")
    print(f"  effective undulation: {summary['effective_undulation']:.12g}")
    print(f"  density mode: {summary['density_mode']}")
    print(f"  coordinate counts: {summary['coordinate_counts']}")
    print(f"  different yarn shared faces: {summary['different_yarn_shared_faces']}")
    print(f"  pbc pairs: {summary['pbc_pair_counts']}")
    print(f"  wrote: {inp_path}")
    print(f"  wrote: {csv_path}")
    print(f"  wrote: {json_path}")
    if not args.no_render:
        print(f"  wrote: {png_path}")


if __name__ == "__main__":
    main()
