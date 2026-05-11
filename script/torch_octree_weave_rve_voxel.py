"""Generate a pure Python/torch octree-derived voxel RVE for a 2x2 plain weave.

The mesh is a conforming rectilinear C3D8/C3D8R voxel mesh. Octree refinement is
used to discover interface-focused coordinate planes, then the coordinates are
globalized so periodic surface nodes can be coupled directly by PBC equations.
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
        build_plain_weave_octree_voxel_rve,
        hex_mesh_quality_summary,
        material_interface_face_summary,
        periodic_node_pairs,
        write_abaqus_voxel_inp,
        write_pbc_pairs_csv,
    )
except ImportError:
    from pytexgen.torch_periodic_mesher import (
        build_plain_weave_octree_voxel_rve,
        hex_mesh_quality_summary,
        material_interface_face_summary,
        periodic_node_pairs,
        write_abaqus_voxel_inp,
        write_pbc_pairs_csv,
    )


def external_hex_faces(elements: np.ndarray) -> list[tuple[int, int, int, int]]:
    local_faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    face_map: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for hex_element in elements:
        for face in local_faces:
            oriented = tuple(int(hex_element[index]) for index in face)
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
        0: "#d4d8d3",
        1: "#c93f32",
        2: "#2878b8",
        3: "#3f8f55",
        4: "#b67820",
    }
    alphas = {0: 0.10, 1: 0.86, 2: 0.86, 3: 0.86, 4: 0.86}

    fig = plt.figure(figsize=(10, 7), dpi=220, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    edge_set = set()
    for material in sorted(set(int(value) for value in material_ids.tolist())):
        material_elements = elements[material_ids == material]
        faces = external_hex_faces(material_elements)
        polygons = [[nodes[index] for index in face] for face in faces]
        if not polygons:
            continue
        ax.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=colors.get(material, "#7aa6c2"),
                edgecolors="#1f1f1f" if material else "#8a8a8a",
                linewidths=0.015 if material else 0.01,
                alpha=alphas.get(material, 0.75),
            )
        )
        if material > 0:
            for face in faces:
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                    edge_set.add(tuple(sorted((int(face[a]), int(face[b])))))

    if edge_set:
        edge_segments = [[nodes[i], nodes[j]] for i, j in sorted(edge_set)]
        ax.add_collection3d(Line3DCollection(edge_segments, colors="#222222", linewidths=0.01, alpha=0.12))

    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except AttributeError:
        pass
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a torch octree-derived 2x2 plain-weave voxel RVE.")
    parser.add_argument("--base-nx", type=int, default=8)
    parser.add_argument("--base-ny", type=int, default=8)
    parser.add_argument("--base-nz", type=int, default=3)
    parser.add_argument("--max-refinement", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("build/torch_periodic_weave_rve"))
    parser.add_argument("--prefix", default="torch_weave_2x2_rve_octree_voxel")
    parser.add_argument("--gap-size", type=float, default=0.01)
    parser.add_argument("--interface-refinement-passes", type=int, default=1)
    parser.add_argument("--element-type", default="C3D8R", choices=("C3D8", "C3D8R"))
    parser.add_argument("--allow-yarn-contact", action="store_false", dest="separate_contacts")
    parser.add_argument("--no-render", action="store_true")
    parser.set_defaults(separate_contacts=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_plain_weave_octree_voxel_rve(
        base_nx=args.base_nx,
        base_ny=args.base_ny,
        base_nz=args.base_nz,
        max_refinement=args.max_refinement,
        domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
        device=args.device,
        gap_size=args.gap_size,
        separate_contacts=args.separate_contacts,
        interface_refinement_passes=args.interface_refinement_passes,
    )
    pairs = periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))
    summary = hex_mesh_quality_summary(mesh)
    interface_summary = material_interface_face_summary(mesh)
    summary["gap_size"] = args.gap_size
    summary["element_type"] = args.element_type
    summary["material_interface_faces"] = interface_summary
    summary["metadata"] = mesh.metadata
    summary["pbc_pair_counts"] = {axis: len(axis_pairs) for axis, axis_pairs in pairs.items()}

    inp_path = args.output_dir / f"{args.prefix}.inp"
    csv_path = args.output_dir / f"{args.prefix}_pbc_pairs.csv"
    json_path = args.output_dir / f"{args.prefix}_quality.json"
    png_path = args.output_dir / f"{args.prefix}.png"

    write_abaqus_voxel_inp(mesh, inp_path, element_type=args.element_type)
    write_pbc_pairs_csv(pairs, csv_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_render:
        render_material_preview(mesh, png_path)

    print("torch octree voxel 2x2 RVE generated")
    print(f"  nodes: {summary['nodes']}")
    print(f"  elements: {summary['elements']}")
    print(f"  zero-volume hexes <= 1e-14: {summary['zero_volume_hexes_le_1e-14']}")
    print(f"  total volume: {summary['total_abs_volume']:.12g}")
    print(f"  bbox volume: {summary['bbox_volume']:.12g}")
    print(f"  gap size: {summary['gap_size']:.12g}")
    print(f"  coordinate counts: {mesh.metadata.get('coordinate_counts')}")
    print(f"  octree leaf count: {mesh.metadata.get('octree_leaf_count')}")
    print(f"  interface refinement added points: {mesh.metadata.get('interface_refinement_added_points')}")
    print(f"  material interface faces: {interface_summary}")
    print(f"  separated contact cells: {mesh.metadata.get('separated_contact_cells')}")
    print(f"  different yarn shared faces: {summary['different_yarn_shared_faces']}")
    print(f"  pbc pairs: {summary['pbc_pair_counts']}")
    print(f"  wrote: {inp_path}")
    print(f"  wrote: {csv_path}")
    print(f"  wrote: {json_path}")
    if not args.no_render:
        print(f"  wrote: {png_path}")


if __name__ == "__main__":
    main()
