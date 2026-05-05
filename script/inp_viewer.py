# -*- coding: utf-8 -*-
"""
inp_viewer.py — 从 Abaqus INP 文件中提取并可视化 Yarn 单元（MAT1）。

用法:
    uv run python script/inp_viewer.py
    uv run python script/inp_viewer.py path/to/mesh.inp
    uv run python script/inp_viewer.py path/to/mesh.inp --backend matplotlib --output view.png --background white

不传 INP 时会自动查找常见输出目录中最新的 .inp 文件。
"""

import sys
import argparse
import numpy as np
import re
from pathlib import Path

try:
    import pyvista as pv
except ImportError:
    pv = None


# ──────────────────────────────────────────────
# INP 解析
# ──────────────────────────────────────────────

def parse_inp(inp_path: str):
    """
    解析 Abaqus INP 文件。

    返回:
        nodes    : dict {label(int) -> np.array([x, y, z])}
        elements : dict {label(int) -> list[int]}   8节点六面体
        elsets   : dict {name(str)  -> list[int]}   单元标签列表
    """
    nodes    = {}
    elements = {}
    elsets   = {}

    section        = None   # 'node' | 'element' | 'elset'
    elset_name     = None
    elset_generate = False
    pending_conn   = None   # (label, partial_conn) 跨行单元

    with open(inp_path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('**'):
                continue

            # ── 关键字行 ──────────────────────────────────
            if line.startswith('*'):
                if pending_conn is not None:
                    _store_element(pending_conn, elements)
                    pending_conn = None

                parts = [p.strip() for p in line.split(',')]
                kw = parts[0].lstrip('*').upper()

                if kw == 'NODE':
                    section = 'node'
                elif kw == 'ELEMENT':
                    section = 'element'
                elif kw == 'ELSET':
                    section = 'elset'
                    elset_name     = None
                    elset_generate = False
                    opts = {}
                    flags = []
                    for p in parts[1:]:
                        if '=' in p:
                            k, v = p.split('=', 1)
                            opts[k.strip().upper()] = v.strip()
                        else:
                            flags.append(p.strip().upper())
                    elset_name     = opts.get('ELSET')
                    elset_generate = 'GENERATE' in flags
                    if elset_name:
                        elsets.setdefault(elset_name, [])
                else:
                    section = None
                continue

            # ── 数据行 ────────────────────────────────────
            vals = [v.strip() for v in line.split(',') if v.strip()]

            if section == 'node':
                label = int(vals[0])
                nodes[label] = np.array([float(vals[1]), float(vals[2]), float(vals[3])])

            elif section == 'element':
                int_vals = [int(v) for v in vals]
                if pending_conn is None:
                    label = int_vals[0]
                    conn  = int_vals[1:]
                    if len(conn) >= 8:
                        elements[label] = conn[:8]
                    else:
                        pending_conn = (label, conn)
                else:
                    label, conn = pending_conn
                    conn.extend(int_vals)
                    if len(conn) >= 8:
                        elements[label] = conn[:8]
                        pending_conn = None
                    else:
                        pending_conn = (label, conn)

            elif section == 'elset' and elset_name:
                int_vals = [int(v) for v in vals]
                if elset_generate:
                    start, end, step = int_vals[0], int_vals[1], int_vals[2]
                    elsets[elset_name].extend(range(start, end + 1, step))
                else:
                    elsets[elset_name].extend(int_vals)

    if pending_conn is not None:
        _store_element(pending_conn, elements)

    return nodes, elements, elsets


def _store_element(pending_conn, elements):
    label, conn = pending_conn
    if len(conn) >= 8:
        elements[label] = conn[:8]


# ──────────────────────────────────────────────
# 构建 PyVista UnstructuredGrid
# ──────────────────────────────────────────────

def build_mesh(nodes: dict, elements: dict, elem_labels: list):
    """
    将选定单元（C3D8 六面体）构建为 PyVista UnstructuredGrid。
    """
    if pv is None:
        raise RuntimeError("PyVista 未安装；请安装 pyvista 或使用 --backend matplotlib")

    # 收集用到的节点，建立连续 0-based 索引
    used = set()
    for lbl in elem_labels:
        if lbl in elements:
            used.update(elements[lbl])

    sorted_nodes = sorted(used)
    node_index   = {lbl: i for i, lbl in enumerate(sorted_nodes)}
    points       = np.array([nodes[lbl] for lbl in sorted_nodes], dtype=np.float64)

    cells      = []
    cell_types = []
    valid      = []

    for lbl in elem_labels:
        if lbl not in elements:
            continue
        conn = elements[lbl]
        if len(conn) < 8:
            continue
        cells.append(8)
        cells.extend(node_index[n] for n in conn[:8])
        cell_types.append(pv.CellType.HEXAHEDRON)
        valid.append(lbl)

    if not valid:
        raise RuntimeError("没有找到可用的六面体单元，请确认单元集名称。")

    mesh = pv.UnstructuredGrid(
        np.array(cells, dtype=np.int_),
        np.array(cell_types, dtype=np.uint8),
        points,
    )
    return mesh, valid


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def _find_yarn_labels(elsets: dict, yarn_elset: str):
    """
    根据 yarn_elset 参数解析要显示的单元标签列表。

    - 若 yarn_elset 为 'auto'：合并所有名称含 'yarn' 的单元集。
    - 若精确匹配：直接使用。
    - 否则：模糊匹配（大小写不敏感子串）。
    """
    if yarn_elset.lower() == 'auto':
        matched = sorted(k for k in elsets if 'yarn' in k.lower())
        if not matched:
            raise ValueError(f"未找到任何含 'yarn' 的单元集，可用: {list(elsets.keys())}")
        labels = []
        for k in matched:
            labels.extend(elsets[k])
        print(f"  自动合并 {len(matched)} 个 Yarn 单元集: {matched}")
        return list(set(labels)), '+'.join(matched)

    if yarn_elset in elsets:
        return elsets[yarn_elset], yarn_elset

    candidates = [k for k in elsets if yarn_elset.upper() in k.upper()]
    if not candidates:
        raise ValueError(f"找不到单元集 '{yarn_elset}'，可用集合: {list(elsets.keys())}")
    print(f"  模糊匹配到 {len(candidates)} 个单元集: {candidates}")
    labels = []
    for k in candidates:
        labels.extend(elsets[k])
    return list(set(labels)), '+'.join(candidates)


def _yarn_sort_key(name: str):
    match = re.search(r'(\d+)$', name)
    if match:
        return (name[:match.start()].lower(), int(match.group(1)))
    return (name.lower(), -1)


def find_default_inp():
    repo_dir = Path(__file__).resolve().parent.parent
    search_roots = [
        repo_dir / 'Saved_SiC_SiC_Shallow_Cross_Straight' / 'RVE',
        repo_dir / 'Saved_Shallow_Cross_Textiles',
        repo_dir / 'build' / 'sic_rve_test',
        repo_dir / 'build' / 'shallow_cross_view',
        repo_dir / 'build',
    ]
    candidates = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob('*.inp'):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    if not candidates:
        raise ValueError("未传入 INP，且默认目录中没有找到 .inp 文件。请先生成 RVE 或显式传入 INP 路径。")
    return max(candidates, key=lambda item: item.stat().st_mtime)


# 各 yarn 的配色（循环使用）
YARN_PALETTE = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
    '#9C755F', '#BAB0AC',
]


def visualize_yarn(inp_path: str, yarn_elset: str = 'auto',
                   color: str = None, opacity: float = 1.0,
                   output: str = None, background: str = '#1a1a2e',
                   show_axes: bool = True, show_title: bool = True,
                   backend: str = 'pyvista', elev: float = 24,
                   azim: float = -58):
    print(f"解析: {inp_path}")
    nodes, elements, elsets = parse_inp(inp_path)
    print(f"  节点: {len(nodes)}    单元: {len(elements)}")
    print(f"  单元集: {list(elsets.keys())}")

    # 收集要渲染的 yarn 列表：auto 模式按单元集分色，否则合并为一组
    if yarn_elset.lower() == 'auto':
        yarn_names = sorted((k for k in elsets if 'yarn' in k.lower()), key=_yarn_sort_key)
        if not yarn_names:
            raise ValueError(f"未找到任何含 'yarn' 的单元集，可用: {list(elsets.keys())}")
        print(f"  自动识别 {len(yarn_names)} 个 Yarn 单元集: {yarn_names}")
        yarn_groups = [(name, elsets[name]) for name in yarn_names]
    else:
        labels, display_name = _find_yarn_labels(elsets, yarn_elset)
        yarn_groups = [(display_name, labels)]

    if backend == 'matplotlib':
        _visualize_yarn_matplotlib(
            inp_path,
            nodes,
            elements,
            yarn_groups,
            color=color,
            opacity=opacity,
            output=output,
            background=background,
            show_axes=show_axes,
            show_title=show_title,
            elev=elev,
            azim=azim,
        )
        return

    if backend == 'plotly':
        _visualize_yarn_plotly(
            inp_path,
            nodes,
            elements,
            yarn_groups,
            color=color,
            opacity=opacity,
            output=output,
            background=background,
        )
        return

    if pv is None:
        raise RuntimeError("PyVista 未安装；请安装 pyvista 或使用 --backend matplotlib")

    plotter = pv.Plotter(lighting='three lights', off_screen=bool(output))
    plotter.set_background(background)

    for i, (name, labels) in enumerate(yarn_groups):
        mesh, valid = build_mesh(nodes, elements, labels)
        if not valid:
            continue

        print(f"  [{i+1}/{len(yarn_groups)}] {name}: {len(valid)} 单元")
        surface = mesh.extract_surface()

        fill_color = color if color else YARN_PALETTE[i % len(YARN_PALETTE)]
        plotter.add_mesh(
            surface,
            color=fill_color,
            opacity=opacity,
            show_edges=True,
            edge_color='#000000',
            line_width=0.5,
            smooth_shading=False,   # 关闭法线插值，保持面片的平面感
        )

    if show_axes:
        plotter.add_axes(line_width=3, xlabel='X', ylabel='Y', zlabel='Z')
    if show_title:
        title_color = 'black' if background.lower() in ('white', '#fff', '#ffffff') else 'white'
        plotter.add_title(Path(inp_path).name, font_size=11, color=title_color)

    print("渲染中...")
    if output:
        plotter.show(screenshot=output, auto_close=True)
        print(f"图片已保存: {output}")
    else:
        plotter.show()


_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def _external_hex_faces(elements: dict, elem_labels: list):
    """Return exterior faces for selected 8-node hex elements."""
    face_map = {}
    for lbl in elem_labels:
        conn = elements.get(lbl)
        if not conn or len(conn) < 8:
            continue
        for face in _HEX_FACES:
            labels = tuple(conn[i] for i in face)
            key = tuple(sorted(labels))
            if key in face_map:
                face_map.pop(key)
            else:
                face_map[key] = labels
    return list(face_map.values())


def _set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _visualize_yarn_matplotlib(
    inp_path,
    nodes,
    elements,
    yarn_groups,
    color=None,
    opacity=1.0,
    output=None,
    background='white',
    show_axes=True,
    show_title=True,
    elev=24,
    azim=-58,
):
    if output is None:
        raise ValueError("matplotlib 后端需要 --output PNG 路径")

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    bg = background
    fig = plt.figure(figsize=(11, 8), dpi=220, facecolor=bg)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg)

    all_points = []
    for i, (name, labels) in enumerate(yarn_groups):
        valid = [lbl for lbl in labels if lbl in elements]
        if not valid:
            continue
        faces = _external_hex_faces(elements, valid)
        polys = [[nodes[node_label] for node_label in face] for face in faces]
        if not polys:
            continue

        print(f"  [{i+1}/{len(yarn_groups)}] {name}: {len(valid)} 单元, {len(polys)} 外表面")
        fill_color = color if color else YARN_PALETTE[i % len(YARN_PALETTE)]
        collection = Poly3DCollection(
            polys,
            facecolors=fill_color,
            edgecolors='#202020',
            linewidths=0.03,
            alpha=opacity,
        )
        ax.add_collection3d(collection)
        all_points.extend(point for poly in polys for point in poly)

    if not all_points:
        raise RuntimeError("没有找到可渲染的 yarn 外表面。")

    _set_axes_equal(ax, np.asarray(all_points, dtype=np.float64))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')

    if show_title:
        ax.set_title(Path(inp_path).name, color='black' if bg.lower() in ('white', '#fff', '#ffffff') else 'white')
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"图片已保存: {output}")


def _visualize_yarn_plotly(
    inp_path,
    nodes,
    elements,
    yarn_groups,
    color=None,
    opacity=1.0,
    output=None,
    background='white',
):
    if output is None:
        raise ValueError("plotly 后端需要 --output HTML 路径")

    import plotly.graph_objects as go

    fig = go.Figure()
    for i_group, (name, labels) in enumerate(yarn_groups):
        valid = [lbl for lbl in labels if lbl in elements]
        if not valid:
            continue
        faces = _external_hex_faces(elements, valid)
        if not faces:
            continue

        node_index = {}
        xs, ys, zs = [], [], []

        def index_for(node_label):
            if node_label not in node_index:
                point = nodes[node_label]
                node_index[node_label] = len(xs)
                xs.append(float(point[0]))
                ys.append(float(point[1]))
                zs.append(float(point[2]))
            return node_index[node_label]

        tris_i, tris_j, tris_k = [], [], []
        for face in faces:
            a, b, c, d = [index_for(node_label) for node_label in face]
            tris_i.extend([a, a])
            tris_j.extend([b, c])
            tris_k.extend([c, d])

        fill_color = color if color else YARN_PALETTE[i_group % len(YARN_PALETTE)]
        print(f"  [{i_group+1}/{len(yarn_groups)}] {name}: {len(valid)} 单元, {len(faces)} 外表面")
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=tris_i,
                j=tris_j,
                k=tris_k,
                name=name,
                color=fill_color,
                opacity=opacity,
                flatshading=True,
                hoverinfo='name',
                showscale=False,
            )
        )

    fig.update_layout(
        title=Path(inp_path).name,
        paper_bgcolor=background,
        plot_bgcolor=background,
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False, backgroundcolor=background),
            yaxis=dict(visible=False, backgroundcolor=background),
            zaxis=dict(visible=False, backgroundcolor=background),
        ),
        margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"交互 HTML 已保存: {output}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化 INP 文件中的 Yarn 单元')
    parser.add_argument('inp', nargs='?', default=None, help='Abaqus .inp 文件路径；不传则自动找最新 .inp')
    parser.add_argument('--elset',       default='auto',    help="Yarn 单元集名称（默认: auto，自动合并所有含 'yarn' 的集合）")
    parser.add_argument('--color',    default=None,  help='统一颜色（默认: 按 yarn 自动分色）')
    parser.add_argument('--opacity',  type=float, default=1.0, help='透明度 0~1（默认: 1.0）')
    parser.add_argument('--output',   default=None, help='输出 PNG 路径；设置后使用离屏渲染')
    parser.add_argument('--background', default='#1a1a2e', help='背景颜色（默认: #1a1a2e）')
    parser.add_argument('--no-axes', action='store_true', help='不显示坐标轴')
    parser.add_argument('--no-title', action='store_true', help='不显示标题')
    parser.add_argument('--backend', choices=['pyvista', 'matplotlib', 'plotly'], default='pyvista',
                        help='渲染后端（默认: pyvista；无 OpenGL 环境可用 matplotlib，交互 HTML 用 plotly）')
    parser.add_argument('--elev', type=float, default=24, help='matplotlib 视角俯仰角（默认: 24）')
    parser.add_argument('--azim', type=float, default=-58, help='matplotlib 视角方位角（默认: -58）')
    args = parser.parse_args()

    inp_path = Path(args.inp) if args.inp else find_default_inp()
    if args.inp is None:
        print(f"未指定 INP，使用最新文件: {inp_path}")

    visualize_yarn(
        inp_path=str(inp_path),
        yarn_elset=args.elset,
        color=args.color,
        opacity=args.opacity,
        output=args.output,
        background=args.background,
        show_axes=not args.no_axes,
        show_title=not args.no_title,
        backend=args.backend,
        elev=args.elev,
        azim=args.azim,
    )
