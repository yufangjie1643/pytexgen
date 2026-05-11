"""Pure Python/torch periodic tetrahedral RVE meshing utilities.

This module intentionally does not call TexGen's C++ meshing or TetGen paths.
It builds a periodic background tetrahedral lattice, cuts it with torch-defined
implicit yarn geometry, and exports a simple Abaqus C3D4 input deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import csv
import math

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "torch_periodic_mesher requires torch. Install with `pip install \"pytexgen[gpu]\"`."
    ) from exc


Axis = str
DomainLike = Sequence[Sequence[float]]


@dataclass
class PeriodicTetMesh:
    """Tetrahedral mesh with material labels and RVE domain metadata."""

    nodes: "torch.Tensor"
    elements: "torch.Tensor"
    domain: "torch.Tensor"
    material_ids: "torch.Tensor"
    material_names: Dict[int, str] = field(default_factory=lambda: {0: "Matrix"})
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PeriodicHexMesh:
    """Axis-aligned hexahedral voxel mesh with material labels and RVE metadata."""

    nodes: "torch.Tensor"
    elements: "torch.Tensor"
    domain: "torch.Tensor"
    material_ids: "torch.Tensor"
    material_names: Dict[int, str] = field(default_factory=lambda: {0: "Matrix"})
    metadata: Dict[str, object] = field(default_factory=dict)


def _as_domain_tensor(domain: DomainLike, device: str | torch.device = "cpu") -> "torch.Tensor":
    arr = torch.as_tensor(domain, dtype=torch.float64, device=device)
    if tuple(arr.shape) != (2, 3):
        raise ValueError("domain must have shape (2, 3)")
    if bool(torch.any(arr[1] <= arr[0]).item()):
        raise ValueError("domain upper bounds must be greater than lower bounds")
    return arr


def _new_mesh(
    nodes: "torch.Tensor",
    elements: "torch.Tensor",
    domain: "torch.Tensor",
    material_ids: Optional["torch.Tensor"] = None,
    material_names: Optional[Mapping[int, str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> PeriodicTetMesh:
    if material_ids is None:
        material_ids = torch.zeros(elements.shape[0], dtype=torch.int64, device=elements.device)
    return PeriodicTetMesh(
        nodes=nodes,
        elements=elements.to(dtype=torch.int64),
        domain=domain,
        material_ids=material_ids.to(dtype=torch.int64, device=elements.device),
        material_names=dict(material_names or {0: "Matrix"}),
        metadata=dict(metadata or {}),
    )


def _new_hex_mesh(
    nodes: "torch.Tensor",
    elements: "torch.Tensor",
    domain: "torch.Tensor",
    material_ids: Optional["torch.Tensor"] = None,
    material_names: Optional[Mapping[int, str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> PeriodicHexMesh:
    if material_ids is None:
        material_ids = torch.zeros(elements.shape[0], dtype=torch.int64, device=elements.device)
    return PeriodicHexMesh(
        nodes=nodes,
        elements=elements.to(dtype=torch.int64),
        domain=domain,
        material_ids=material_ids.to(dtype=torch.int64, device=elements.device),
        material_names=dict(material_names or {0: "Matrix"}),
        metadata=dict(metadata or {}),
    )


def tet_signed_volumes(nodes: "torch.Tensor", elements: "torch.Tensor") -> "torch.Tensor":
    """Return signed volumes for linear tetrahedra."""

    pts = nodes[elements.to(dtype=torch.long)]
    return torch.sum(torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0], dim=1) * (pts[:, 3] - pts[:, 0]), dim=1) / 6.0


def tet_volumes(nodes: "torch.Tensor", elements: "torch.Tensor") -> "torch.Tensor":
    """Return absolute volumes for linear tetrahedra."""

    return tet_signed_volumes(nodes, elements).abs()


def hex_volumes(nodes: "torch.Tensor", elements: "torch.Tensor") -> "torch.Tensor":
    """Return volumes for axis-aligned rectilinear C3D8 voxel elements."""

    pts = nodes[elements.to(dtype=torch.long)]
    mins = pts.min(dim=1).values
    maxs = pts.max(dim=1).values
    return torch.prod(maxs - mins, dim=1).abs()


def _orient_tets_positive(nodes: "torch.Tensor", elements: "torch.Tensor") -> "torch.Tensor":
    oriented = elements.clone()
    signed = tet_signed_volumes(nodes, oriented)
    flip = torch.nonzero(signed < 0.0, as_tuple=False).flatten()
    if flip.numel() > 0:
        tmp = oriented[flip, 0].clone()
        oriented[flip, 0] = oriented[flip, 1]
        oriented[flip, 1] = tmp
    return oriented


def _coordinate_tensor(values: Sequence[float], axis_name: str, device: str | torch.device) -> "torch.Tensor":
    arr = torch.as_tensor(values, dtype=torch.float64, device=device)
    if arr.ndim != 1 or arr.numel() < 2:
        raise ValueError(f"{axis_name}_coords must be a one-dimensional sequence with at least two values")
    if bool(torch.any(arr[1:] <= arr[:-1]).item()):
        raise ValueError(f"{axis_name}_coords must be strictly increasing")
    return arr


def generate_periodic_tet_lattice_from_coordinates(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
    device: str | torch.device = "cpu",
) -> PeriodicTetMesh:
    """Generate a periodic Cartesian tetrahedral lattice from explicit coordinates."""

    xs = _coordinate_tensor(x_coords, "x", device)
    ys = _coordinate_tensor(y_coords, "y", device)
    zs = _coordinate_tensor(z_coords, "z", device)
    nx = int(xs.numel() - 1)
    ny = int(ys.numel() - 1)
    nz = int(zs.numel() - 1)
    dom = torch.stack(
        (
            torch.stack((xs[0], ys[0], zs[0])),
            torch.stack((xs[-1], ys[-1], zs[-1])),
        )
    )
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    nodes = grid.reshape(-1, 3)

    def idx(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets: List[Tuple[int, int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v000 = idx(i, j, k)
                v100 = idx(i + 1, j, k)
                v010 = idx(i, j + 1, k)
                v110 = idx(i + 1, j + 1, k)
                v001 = idx(i, j, k + 1)
                v101 = idx(i + 1, j, k + 1)
                v011 = idx(i, j + 1, k + 1)
                v111 = idx(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    ]
                )
    elements = torch.as_tensor(tets, dtype=torch.int64, device=device)
    elements = _orient_tets_positive(nodes, elements)
    return _new_mesh(
        nodes,
        elements,
        dom,
        metadata={
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "source": "periodic_lattice",
            "coordinate_counts": {"x": int(xs.numel()), "y": int(ys.numel()), "z": int(zs.numel())},
        },
    )


def generate_periodic_tet_lattice(
    nx: int,
    ny: int,
    nz: int,
    domain: DomainLike,
    device: str | torch.device = "cpu",
) -> PeriodicTetMesh:
    """Generate a periodic Cartesian tetrahedral lattice for an RVE box."""

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, and nz must be positive")
    dom = _as_domain_tensor(domain, device=device)
    xs = torch.linspace(dom[0, 0], dom[1, 0], nx + 1, dtype=torch.float64, device=device)
    ys = torch.linspace(dom[0, 1], dom[1, 1], ny + 1, dtype=torch.float64, device=device)
    zs = torch.linspace(dom[0, 2], dom[1, 2], nz + 1, dtype=torch.float64, device=device)
    mesh = generate_periodic_tet_lattice_from_coordinates(xs, ys, zs, device=device)
    mesh.metadata.update({"nx": nx, "ny": ny, "nz": nz})
    return mesh


def generate_periodic_hex_lattice_from_coordinates(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
    device: str | torch.device = "cpu",
) -> PeriodicHexMesh:
    """Generate a conforming rectilinear C3D8 voxel mesh from explicit coordinates."""

    xs = _coordinate_tensor(x_coords, "x", device)
    ys = _coordinate_tensor(y_coords, "y", device)
    zs = _coordinate_tensor(z_coords, "z", device)
    nx = int(xs.numel() - 1)
    ny = int(ys.numel() - 1)
    nz = int(zs.numel() - 1)
    dom = torch.stack(
        (
            torch.stack((xs[0], ys[0], zs[0])),
            torch.stack((xs[-1], ys[-1], zs[-1])),
        )
    )
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    nodes = grid.reshape(-1, 3)

    def idx(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                hexes.append(
                    (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    )
                )
    elements = torch.as_tensor(hexes, dtype=torch.int64, device=device)
    return _new_hex_mesh(
        nodes,
        elements,
        dom,
        metadata={
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "source": "periodic_hex_lattice",
            "coordinate_counts": {"x": int(xs.numel()), "y": int(ys.numel()), "z": int(zs.numel())},
        },
    )


def _axis_index(axis: Axis) -> int:
    mapping = {"x": 0, "y": 1, "z": 2}
    if axis not in mapping:
        raise ValueError(f"unsupported axis {axis!r}")
    return mapping[axis]


def _coord_key(values: Iterable[float], tol: float) -> Tuple[int, ...]:
    return tuple(int(round(float(v) / tol)) for v in values)


def periodic_node_pairs(
    nodes: "torch.Tensor",
    domain: "torch.Tensor | DomainLike",
    axes: Sequence[Axis] = ("x", "y"),
    tol: float = 1e-8,
) -> Dict[Axis, List[Tuple[int, int]]]:
    """Pair periodic boundary nodes on selected RVE faces.

    Returned node indices are zero-based. Writers convert them to Abaqus labels.
    """

    dom = domain if isinstance(domain, torch.Tensor) else _as_domain_tensor(domain)
    pts = nodes.detach().cpu().numpy()
    dom_np = dom.detach().cpu().numpy()
    pairs: Dict[Axis, List[Tuple[int, int]]] = {}
    for axis in axes:
        ax = _axis_index(axis)
        other = [i for i in range(3) if i != ax]
        lo, hi = dom_np[0, ax], dom_np[1, ax]
        minus: Dict[Tuple[int, ...], int] = {}
        plus: Dict[Tuple[int, ...], int] = {}
        for idx, point in enumerate(pts):
            key = _coord_key((point[i] for i in other), tol)
            if abs(point[ax] - lo) <= tol:
                minus[key] = idx
            if abs(point[ax] - hi) <= tol:
                plus[key] = idx
        common = sorted(set(minus).intersection(plus))
        pairs[axis] = [(minus[key], plus[key]) for key in common]
    return pairs


def _periodic_delta(coord: "torch.Tensor", center: float, length: float) -> "torch.Tensor":
    return torch.remainder(coord - center + 0.5 * length, length) - 0.5 * length


def _unique_sorted(values: Iterable[float], lo: float, hi: float, tol: float = 1e-12) -> np.ndarray:
    filtered = sorted(float(value) for value in values if lo - tol <= float(value) <= hi + tol)
    out: List[float] = []
    for value in filtered:
        clipped = min(hi, max(lo, value))
        if not out or abs(clipped - out[-1]) > tol:
            out.append(clipped)
    if out[0] != lo:
        out.insert(0, lo)
    if out[-1] != hi:
        out.append(hi)
    return np.asarray(out, dtype=np.float64)


def _add_periodic_axis_point(points: List[float], value: float, lo: float, hi: float, tol: float = 1e-12) -> None:
    length = hi - lo
    wrapped = ((float(value) - lo) % length) + lo
    if abs(wrapped - lo) <= tol or abs(wrapped - hi) <= tol:
        return
    points.append(wrapped)


def _topology_xy_coordinates(
    count: int,
    lo: float,
    hi: float,
    centers: Sequence[float],
    half_width: float,
    interface_levels: int,
    crossing_levels: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("topology coordinate counts must be positive")
    if interface_levels < 0 or crossing_levels < 0:
        raise ValueError("topology refinement levels must be non-negative")
    length = hi - lo
    points = np.linspace(lo, hi, count + 1, dtype=np.float64).tolist()
    base_step = length / count
    interface_delta = min(0.25 * half_width, 0.35 * base_step)
    for center in centers:
        _add_periodic_axis_point(points, center, lo, hi)
        for level in range(1, crossing_levels + 1):
            delta = half_width * level / (crossing_levels + 1)
            _add_periodic_axis_point(points, center - delta, lo, hi)
            _add_periodic_axis_point(points, center + delta, lo, hi)
        for side in (-1.0, 1.0):
            boundary = center + side * half_width
            _add_periodic_axis_point(points, boundary, lo, hi)
            for level in range(1, interface_levels + 1):
                delta = interface_delta * level
                _add_periodic_axis_point(points, boundary - delta, lo, hi)
                _add_periodic_axis_point(points, boundary + delta, lo, hi)
    return _unique_sorted(points, lo, hi)


def _topology_z_coordinates(
    count: int,
    lo: float,
    hi: float,
    yarn_height: float,
    undulation: float,
    gap_size: float,
    gap_levels: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("topology coordinate counts must be positive")
    if gap_levels < 0:
        raise ValueError("topology gap levels must be non-negative")
    z_mid = 0.5 * (lo + hi)
    half_height = 0.5 * yarn_height
    effective_undulation = _gap_adjusted_undulation(yarn_height, undulation, gap_size)
    points = np.linspace(lo, hi, count + 1, dtype=np.float64).tolist()
    anchors = [
        z_mid,
        z_mid - half_height,
        z_mid + half_height,
        z_mid - effective_undulation,
        z_mid + effective_undulation,
        z_mid - (effective_undulation + half_height),
        z_mid + (effective_undulation + half_height),
        z_mid - abs(effective_undulation - half_height),
        z_mid + abs(effective_undulation - half_height),
    ]
    if gap_size > 0.0 and gap_levels > 0:
        anchors.extend((z_mid - 0.5 * gap_size, z_mid + 0.5 * gap_size))
    points.extend(anchors)
    return _unique_sorted(points, lo, hi)


def plain_weave_topology_coordinates(
    nx: int,
    ny: int,
    nz: int,
    domain: DomainLike = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
    interface_levels: int = 1,
    crossing_levels: int = 1,
    gap_levels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nonuniform coordinate lines clustered around plain-weave topology."""

    dom = _as_domain_tensor(domain).detach().cpu().numpy()
    lo = dom[0]
    hi = dom[1]
    lx = float(hi[0] - lo[0])
    ly = float(hi[1] - lo[1])
    half_width = 0.5 * yarn_width
    x_centers = (float(lo[0] + 0.25 * lx), float(lo[0] + 0.75 * lx))
    y_centers = (float(lo[1] + 0.25 * ly), float(lo[1] + 0.75 * ly))
    xs = _topology_xy_coordinates(nx, float(lo[0]), float(hi[0]), x_centers, half_width, interface_levels, crossing_levels)
    ys = _topology_xy_coordinates(ny, float(lo[1]), float(hi[1]), y_centers, half_width, interface_levels, crossing_levels)
    zs = _topology_z_coordinates(nz, float(lo[2]), float(hi[2]), yarn_height, undulation, gap_size, gap_levels)
    return xs, ys, zs


def _gap_adjusted_undulation(yarn_height: float, undulation: float, gap_size: float) -> float:
    if gap_size < 0.0:
        raise ValueError("gap_size must be non-negative")
    return max(float(undulation), 0.5 * (float(yarn_height) + float(gap_size)))


def _apply_crossing_gap_clip(
    phi: "torch.Tensor",
    z: "torch.Tensor",
    z_mid: float,
    half_height: float,
    gap_size: float,
    active: "torch.Tensor",
    top_yarn: bool,
) -> "torch.Tensor":
    if gap_size <= 0.0:
        return phi
    if top_yarn:
        side_phi = (z_mid + 0.5 * gap_size - z) / half_height
    else:
        side_phi = (z - (z_mid - 0.5 * gap_size)) / half_height
    return torch.where(active, torch.maximum(phi, side_phi), phi)


def plain_weave_yarn_levelsets(
    points: "torch.Tensor",
    domain: "torch.Tensor | DomainLike",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
) -> "torch.Tensor":
    """Evaluate one implicit level-set column per yarn."""

    dom = domain if isinstance(domain, torch.Tensor) else _as_domain_tensor(domain, points.device)
    lo = dom[0]
    hi = dom[1]
    lx = float((hi[0] - lo[0]).item())
    ly = float((hi[1] - lo[1]).item())
    z_mid = float(((lo[2] + hi[2]) * 0.5).item())
    a = yarn_width * 0.5
    b = yarn_height * 0.5
    effective_undulation = _gap_adjusted_undulation(yarn_height, undulation, gap_size)

    x = points[:, 0] - lo[0]
    y = points[:, 1] - lo[1]
    z = points[:, 2]
    x_wave = torch.cos(2.0 * math.pi * (x / lx - 0.25))
    y_wave = torch.cos(2.0 * math.pi * (y / ly - 0.25))
    yarn_phis: List["torch.Tensor"] = []
    x_centers = (0.25 * lx, 0.75 * lx)
    y_centers = (0.25 * ly, 0.75 * ly)

    # Two x-directed yarns. Centers sit at quarter and three-quarter pitch.
    for row, y_center in enumerate(y_centers):
        transverse = _periodic_delta(y, y_center, ly)
        phase_sign = 1.0 if row == 0 else -1.0
        zc = z_mid + phase_sign * effective_undulation * x_wave
        phi = torch.sqrt((transverse / a) ** 2 + ((z - zc) / b) ** 2) - 1.0
        for col, x_center in enumerate(x_centers):
            crossing_active = torch.abs(_periodic_delta(x, x_center, lx)) <= a
            phi = _apply_crossing_gap_clip(
                phi,
                z,
                z_mid,
                b,
                gap_size,
                crossing_active,
                top_yarn=(row == col),
            )
        yarn_phis.append(phi)

    # Two y-directed yarns. They are opposite to the x-yarn phase at crossings.
    for col, x_center in enumerate(x_centers):
        transverse = _periodic_delta(x, x_center, lx)
        phase_sign = -1.0 if col == 0 else 1.0
        zc = z_mid + phase_sign * effective_undulation * y_wave
        phi = torch.sqrt((transverse / a) ** 2 + ((z - zc) / b) ** 2) - 1.0
        for row, y_center in enumerate(y_centers):
            crossing_active = torch.abs(_periodic_delta(y, y_center, ly)) <= a
            phi = _apply_crossing_gap_clip(
                phi,
                z,
                z_mid,
                b,
                gap_size,
                crossing_active,
                top_yarn=(row != col),
            )
        yarn_phis.append(phi)

    return torch.stack(yarn_phis, dim=1)


def plain_weave_levelset(
    points: "torch.Tensor",
    domain: "torch.Tensor | DomainLike",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Evaluate a torch implicit 2x2 plain-weave yarn model.

    The returned ``phi`` is negative inside the nearest yarn. ``yarn_ids`` are
    zero-based labels for the nearest yarn, regardless of inside/outside status.
    """

    stacked = plain_weave_yarn_levelsets(
        points,
        domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
    )
    return torch.min(stacked, dim=1)


def classify_plain_weave_points(
    points: "torch.Tensor",
    domain: "torch.Tensor | DomainLike",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
) -> "torch.Tensor":
    """Classify points as Matrix=0 or YarnN=1..4 using the plain-weave level set."""

    phi, yarn_ids = plain_weave_levelset(
        points,
        domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
    )
    return torch.where(phi <= 0.0, yarn_ids.to(dtype=torch.int64) + 1, torch.zeros_like(yarn_ids, dtype=torch.int64))


def _different_yarn_contact_cells(elements: "torch.Tensor", material_ids: "torch.Tensor") -> List[int]:
    if elements.shape[1] == 4:
        local_faces: Sequence[Sequence[int]] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    elif elements.shape[1] == 8:
        local_faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    else:
        raise ValueError("unsupported element topology")
    faces: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    contact_cells: set[int] = set()
    elems_np = elements.detach().cpu().numpy()
    mats_np = material_ids.detach().cpu().numpy()
    for element_index, element in enumerate(elems_np):
        material = int(mats_np[element_index])
        for face in local_faces:
            key = tuple(sorted(int(element[index]) for index in face))
            if key in faces:
                other_index, other_material = faces.pop(key)
                if material > 0 and other_material > 0 and material != other_material:
                    contact_cells.add(element_index)
                    contact_cells.add(other_index)
            else:
                faces[key] = (element_index, material)
    return sorted(contact_cells)


def separate_yarn_contact_cells(elements: "torch.Tensor", material_ids: "torch.Tensor") -> Tuple["torch.Tensor", int]:
    """Relabel direct different-yarn contacts as matrix cells for voxel gap robustness."""

    contact_cells = _different_yarn_contact_cells(elements, material_ids)
    if not contact_cells:
        return material_ids.clone(), 0
    separated = material_ids.clone()
    separated[torch.as_tensor(contact_cells, dtype=torch.long, device=material_ids.device)] = 0
    return separated, len(contact_cells)


def _classify_hex_centers(
    mesh: PeriodicHexMesh,
    yarn_width: float,
    yarn_height: float,
    undulation: float,
    gap_size: float,
    separate_contacts: bool,
) -> Tuple["torch.Tensor", int]:
    pts = mesh.nodes[mesh.elements.to(dtype=torch.long)]
    centers = pts.mean(dim=1)
    material_ids = classify_plain_weave_points(
        centers,
        mesh.domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
    )
    if separate_contacts:
        return separate_yarn_contact_cells(mesh.elements, material_ids)
    return material_ids, 0


def _add_interval_midpoint(points: set[float], coords: np.ndarray, index: int, min_size: float) -> None:
    if index < 0 or index >= len(coords) - 1:
        return
    if float(coords[index + 1] - coords[index]) <= min_size:
        return
    points.add(float(0.5 * (coords[index] + coords[index + 1])))


def refine_material_interface_coordinates(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
    domain: DomainLike,
    device: str | torch.device,
    yarn_width: float,
    yarn_height: float,
    undulation: float,
    gap_size: float,
    passes: int,
    separate_contacts: bool = True,
    min_cell_size: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Split rectilinear coordinates around material-transition faces."""

    if passes < 0:
        raise ValueError("interface_refinement_passes must be non-negative")
    xs = np.asarray(x_coords, dtype=np.float64)
    ys = np.asarray(y_coords, dtype=np.float64)
    zs = np.asarray(z_coords, dtype=np.float64)
    pass_metadata: List[Dict[str, int]] = []
    total_added = 0
    total_interface_faces = 0

    for pass_index in range(passes):
        mesh = generate_periodic_hex_lattice_from_coordinates(xs, ys, zs, device=device)
        material_ids, _separated = _classify_hex_centers(
            mesh,
            yarn_width=yarn_width,
            yarn_height=yarn_height,
            undulation=undulation,
            gap_size=gap_size,
            separate_contacts=separate_contacts,
        )
        nx = len(xs) - 1
        ny = len(ys) - 1
        nz = len(zs) - 1
        materials = material_ids.detach().cpu().numpy().reshape((nx, ny, nz))
        x_points = set(float(value) for value in xs)
        y_points = set(float(value) for value in ys)
        z_points = set(float(value) for value in zs)
        interface_faces = 0

        def mark_x_face(i: int, j: int, k: int) -> None:
            _add_interval_midpoint(x_points, xs, i, min_cell_size)
            _add_interval_midpoint(x_points, xs, i + 1, min_cell_size)
            _add_interval_midpoint(y_points, ys, j, min_cell_size)
            _add_interval_midpoint(z_points, zs, k, min_cell_size)

        def mark_y_face(i: int, j: int, k: int) -> None:
            _add_interval_midpoint(y_points, ys, j, min_cell_size)
            _add_interval_midpoint(y_points, ys, j + 1, min_cell_size)
            _add_interval_midpoint(x_points, xs, i, min_cell_size)
            _add_interval_midpoint(z_points, zs, k, min_cell_size)

        def mark_z_face(i: int, j: int, k: int) -> None:
            _add_interval_midpoint(z_points, zs, k, min_cell_size)
            _add_interval_midpoint(z_points, zs, k + 1, min_cell_size)
            _add_interval_midpoint(x_points, xs, i, min_cell_size)
            _add_interval_midpoint(y_points, ys, j, min_cell_size)

        diff_x = materials[:-1, :, :] != materials[1:, :, :]
        for i, j, k in np.argwhere(diff_x):
            interface_faces += 1
            mark_x_face(int(i), int(j), int(k))

        diff_y = materials[:, :-1, :] != materials[:, 1:, :]
        for i, j, k in np.argwhere(diff_y):
            interface_faces += 1
            mark_y_face(int(i), int(j), int(k))

        diff_z = materials[:, :, :-1] != materials[:, :, 1:]
        for i, j, k in np.argwhere(diff_z):
            interface_faces += 1
            mark_z_face(int(i), int(j), int(k))

        new_xs = _unique_sorted(x_points, float(xs[0]), float(xs[-1]))
        new_ys = _unique_sorted(y_points, float(ys[0]), float(ys[-1]))
        new_zs = _unique_sorted(z_points, float(zs[0]), float(zs[-1]))
        added = (len(new_xs) - len(xs)) + (len(new_ys) - len(ys)) + (len(new_zs) - len(zs))
        pass_metadata.append(
            {
                "pass": pass_index + 1,
                "interface_faces": int(interface_faces),
                "added_points": int(added),
            }
        )
        total_interface_faces = int(interface_faces)
        total_added += int(added)
        xs, ys, zs = new_xs, new_ys, new_zs
        if added == 0:
            break

    return (
        xs,
        ys,
        zs,
        {
            "interface_refinement_added_points": int(total_added),
            "interface_refinement_pass_details": pass_metadata,
            "interface_refinement_last_face_count": int(total_interface_faces),
        },
    )


def _intersection_label(
    a: int,
    b: int,
    phi_a: float,
    phi_b: float,
    new_nodes: List[np.ndarray],
    edge_nodes: Dict[Tuple[int, int], int],
) -> int:
    key = (a, b) if a < b else (b, a)
    if key in edge_nodes:
        return edge_nodes[key]
    denom = phi_a - phi_b
    t = 0.5 if abs(denom) < 1e-14 else phi_a / denom
    t = min(1.0, max(0.0, t))
    point = new_nodes[a] + t * (new_nodes[b] - new_nodes[a])
    label = len(new_nodes)
    new_nodes.append(point)
    edge_nodes[key] = label
    return label


def _canonical_plane(normal: np.ndarray, offset: float) -> Tuple[np.ndarray, float]:
    idx = int(np.argmax(np.abs(normal)))
    if normal[idx] < 0.0:
        normal = -normal
        offset = -offset
    return normal, offset


def _find_hull_faces(coords: np.ndarray, eps: float = 1e-10) -> List[Tuple[int, int, int]]:
    """Return triangular hull faces for a small convex point set."""

    num = coords.shape[0]
    planes: List[Tuple[np.ndarray, float, set[int]]] = []
    for i, j, k in combinations(range(num), 3):
        normal = np.cross(coords[j] - coords[i], coords[k] - coords[i])
        norm = float(np.linalg.norm(normal))
        if norm <= eps:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, coords[i]))
        signed = coords @ normal + offset
        if not (np.all(signed <= eps) or np.all(signed >= -eps)):
            continue
        normal, offset = _canonical_plane(normal, offset)
        verts = set(np.nonzero(np.abs(coords @ normal + offset) <= 1e-8)[0].tolist())
        matched = False
        for plane_index, (old_normal, old_offset, old_verts) in enumerate(planes):
            if abs(float(np.dot(old_normal, normal)) - 1.0) <= 1e-8 and abs(old_offset - offset) <= 1e-8:
                planes[plane_index] = (old_normal, old_offset, old_verts.union(verts))
                matched = True
                break
        if not matched:
            planes.append((normal, offset, verts))

    faces: List[Tuple[int, int, int]] = []
    seen: set[Tuple[int, int, int]] = set()
    for normal, _offset, verts in planes:
        ordered_input = sorted(verts)
        if len(ordered_input) < 3:
            continue
        center = coords[ordered_input].mean(axis=0)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, normal))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        angles = [
            math.atan2(float(np.dot(coords[index] - center, v)), float(np.dot(coords[index] - center, u)))
            for index in ordered_input
        ]
        ordered = [index for _, index in sorted(zip(angles, ordered_input))]
        for m in range(1, len(ordered) - 1):
            tri = (ordered[0], ordered[m], ordered[m + 1])
            key = tuple(sorted(tri))
            if key not in seen:
                faces.append(tri)
                seen.add(key)
    return faces


def _tetrahedralize_convex_polyhedron(
    vertex_labels: Sequence[int],
    new_nodes: List[np.ndarray],
) -> List[Tuple[int, int, int, int]]:
    labels = list(dict.fromkeys(vertex_labels))
    if len(labels) < 4:
        return []
    coords = np.asarray([new_nodes[label] for label in labels], dtype=np.float64)
    centroid_label = len(new_nodes)
    new_nodes.append(coords.mean(axis=0))
    faces = _find_hull_faces(coords)
    tets: List[Tuple[int, int, int, int]] = []
    for face in faces:
        tet = (centroid_label, labels[face[0]], labels[face[1]], labels[face[2]])
        volume = _signed_volume_np(*(new_nodes[index] for index in tet))
        if abs(volume) <= 1e-16:
            continue
        if volume < 0.0:
            tet = (tet[1], tet[0], tet[2], tet[3])
        tets.append(tet)
    return tets


def _signed_volume_np(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def _mode_int(values: Sequence[int], default: int) -> int:
    counts: Dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return int(default)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def cut_tet_mesh_by_levelset(
    mesh: PeriodicTetMesh,
    phi: "torch.Tensor",
    yarn_ids: "torch.Tensor | np.ndarray",
    eps: float = 1e-12,
    eligible_material_ids: Optional[Iterable[int]] = None,
    inside_material_id: Optional[int] = None,
    min_output_volume: float = 1e-14,
) -> PeriodicTetMesh:
    """Cut a tetrahedral lattice by a binary yarn/matrix level set."""

    eligible = None if eligible_material_ids is None else {int(mat) for mat in eligible_material_ids}
    phi_np = np.asarray(phi.detach().cpu() if isinstance(phi, torch.Tensor) else phi, dtype=np.float64)
    yarn_np = np.asarray(yarn_ids.detach().cpu() if isinstance(yarn_ids, torch.Tensor) else yarn_ids, dtype=np.int64)
    nodes_np = mesh.nodes.detach().cpu().numpy().astype(np.float64)
    elements_np = mesh.elements.detach().cpu().numpy().astype(np.int64)
    input_materials = mesh.material_ids.detach().cpu().numpy().astype(np.int64)
    new_nodes: List[np.ndarray] = [node.copy() for node in nodes_np]
    new_elements: List[Tuple[int, int, int, int]] = []
    materials: List[int] = []
    edge_nodes: Dict[Tuple[int, int], int] = {}
    skipped_degenerate = 0

    tet_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_intersections: Dict[Tuple[int, int], int] = {}

    def add_tet(labels: Sequence[int], material_id: int) -> None:
        nonlocal skipped_degenerate
        if len(set(labels)) != 4:
            skipped_degenerate += 1
            return
        tet = tuple(int(label) for label in labels)
        volume = _signed_volume_np(*(new_nodes[index] for index in tet))
        if abs(volume) <= min_output_volume:
            skipped_degenerate += 1
            return
        if volume < 0.0:
            tet = (tet[1], tet[0], tet[2], tet[3])
        new_elements.append(tet)
        materials.append(int(material_id))

    def crossing_label(local_a: int, local_b: int, tet_labels: np.ndarray, values: np.ndarray) -> int:
        key = tuple(sorted((local_a, local_b)))
        if key in edge_intersections:
            return edge_intersections[key]
        label = _intersection_label(
            int(tet_labels[local_a]),
            int(tet_labels[local_b]),
            float(values[local_a]),
            float(values[local_b]),
            new_nodes,
            edge_nodes,
        )
        edge_intersections[key] = label
        return label

    def resolved_inside_material(tet: np.ndarray, inside_idx: Sequence[int]) -> int:
        if inside_material_id is not None:
            return int(inside_material_id)
        return _mode_int(
            [int(yarn_np[tet[i]]) for i in inside_idx],
            default=int(yarn_np[tet[0]]),
        ) + 1

    for element_index, tet in enumerate(elements_np):
        edge_intersections.clear()
        outside_material = int(input_materials[element_index])
        if eligible is not None and outside_material not in eligible:
            add_tet(tuple(int(v) for v in tet), outside_material)
            continue
        values = phi_np[tet]
        inside = values <= eps
        n_inside = int(inside.sum())
        if n_inside == 0:
            add_tet(tuple(int(v) for v in tet), outside_material)
            continue
        if n_inside == 4:
            labels = tuple(int(v) for v in tet)
            add_tet(labels, resolved_inside_material(tet, range(4)))
            continue

        inside_idx = [i for i in range(4) if inside[i]]
        outside_idx = [i for i in range(4) if not inside[i]]
        yarn_material = resolved_inside_material(tet, inside_idx)

        if n_inside == 1:
            i0 = inside_idx[0]
            o0, o1, o2 = outside_idx
            p0 = crossing_label(i0, o0, tet, values)
            p1 = crossing_label(i0, o1, tet, values)
            p2 = crossing_label(i0, o2, tet, values)
            add_tet((int(tet[i0]), p0, p1, p2), yarn_material)
            add_tet((int(tet[o0]), int(tet[o1]), int(tet[o2]), p2), outside_material)
            add_tet((int(tet[o0]), int(tet[o1]), p1, p2), outside_material)
            add_tet((int(tet[o0]), p0, p1, p2), outside_material)
            continue

        if n_inside == 3:
            o0 = outside_idx[0]
            i0, i1, i2 = inside_idx
            p0 = crossing_label(i0, o0, tet, values)
            p1 = crossing_label(i1, o0, tet, values)
            p2 = crossing_label(i2, o0, tet, values)
            add_tet((int(tet[o0]), p0, p1, p2), outside_material)
            add_tet((int(tet[i0]), int(tet[i1]), int(tet[i2]), p2), yarn_material)
            add_tet((int(tet[i0]), int(tet[i1]), p1, p2), yarn_material)
            add_tet((int(tet[i0]), p0, p1, p2), yarn_material)
            continue

        if n_inside == 2:
            i0, i1 = inside_idx
            o0, o1 = outside_idx
            p00 = crossing_label(i0, o0, tet, values)
            p01 = crossing_label(i0, o1, tet, values)
            p10 = crossing_label(i1, o0, tet, values)
            p11 = crossing_label(i1, o1, tet, values)
            add_tet((int(tet[i0]), p00, p01, p11), yarn_material)
            add_tet((int(tet[i0]), p00, p10, p11), yarn_material)
            add_tet((int(tet[i0]), int(tet[i1]), p10, p11), yarn_material)
            add_tet((int(tet[o0]), p00, p10, p11), outside_material)
            add_tet((int(tet[o0]), p00, p01, p11), outside_material)
            add_tet((int(tet[o0]), int(tet[o1]), p01, p11), outside_material)
            continue

    out_nodes = torch.as_tensor(np.asarray(new_nodes), dtype=torch.float64, device=mesh.nodes.device)
    out_elements = torch.as_tensor(np.asarray(new_elements), dtype=torch.int64, device=mesh.nodes.device)
    out_elements = _orient_tets_positive(out_nodes, out_elements)
    material_ids = torch.as_tensor(materials, dtype=torch.int64, device=mesh.nodes.device)
    names = {0: "Matrix"}
    for mat in sorted(set(materials)):
        if mat > 0:
            names[mat] = f"Yarn{mat - 1}"
    metadata = dict(mesh.metadata)
    metadata.update(
        {
            "source": "torch_levelset_cut",
            "input_elements": int(mesh.elements.shape[0]),
            "cut_elements": int(out_elements.shape[0]),
            "skipped_degenerate_tets": int(metadata.get("skipped_degenerate_tets", 0)) + skipped_degenerate,
        }
    )
    return _new_mesh(out_nodes, out_elements, mesh.domain, material_ids, names, metadata)


def build_plain_weave_rve(
    nx: int,
    ny: int,
    nz: int,
    domain: DomainLike = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
    device: str | torch.device = "cpu",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
    density_mode: str = "uniform",
    topology_interface_levels: int = 1,
    topology_crossing_levels: int = 1,
    topology_gap_levels: int = 1,
) -> PeriodicTetMesh:
    """Build a 2x2 plain-weave yarn + matrix RVE with a torch level set."""

    mode = density_mode.lower()
    if mode == "uniform":
        lattice = generate_periodic_tet_lattice(nx, ny, nz, domain, device=device)
    elif mode == "topology":
        xs, ys, zs = plain_weave_topology_coordinates(
            nx=nx,
            ny=ny,
            nz=nz,
            domain=domain,
            yarn_width=yarn_width,
            yarn_height=yarn_height,
            undulation=undulation,
            gap_size=gap_size,
            interface_levels=topology_interface_levels,
            crossing_levels=topology_crossing_levels,
            gap_levels=topology_gap_levels,
        )
        lattice = generate_periodic_tet_lattice_from_coordinates(xs, ys, zs, device=device)
    else:
        raise ValueError("density_mode must be 'uniform' or 'topology'")
    effective_undulation = _gap_adjusted_undulation(yarn_height, undulation, gap_size)
    cut = lattice
    num_yarns = 4
    for yarn_index in range(num_yarns):
        yarn_phi = plain_weave_yarn_levelsets(
            cut.nodes,
            cut.domain,
            yarn_width=yarn_width,
            yarn_height=yarn_height,
            undulation=undulation,
            gap_size=gap_size,
        )[:, yarn_index]
        yarn_ids = torch.full(
            (cut.nodes.shape[0],),
            yarn_index,
            dtype=torch.int64,
            device=cut.nodes.device,
        )
        cut = cut_tet_mesh_by_levelset(
            cut,
            yarn_phi,
            yarn_ids,
            eligible_material_ids=(0,),
            inside_material_id=yarn_index + 1,
        )
    cut.metadata.update(
        {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "yarn_width": yarn_width,
            "yarn_height": yarn_height,
            "undulation": undulation,
            "effective_undulation": effective_undulation,
            "gap_size": gap_size,
            "density_mode": mode,
            "coordinate_counts": dict(lattice.metadata.get("coordinate_counts", {})),
            "topology_interface_levels": topology_interface_levels if mode == "topology" else 0,
            "topology_crossing_levels": topology_crossing_levels if mode == "topology" else 0,
            "topology_gap_levels": topology_gap_levels if mode == "topology" else 0,
        }
    )
    return cut


def _cell_corners_np(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )


def _cell_sample_grid_np(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    xs = (lo[0], 0.5 * (lo[0] + hi[0]), hi[0])
    ys = (lo[1], 0.5 * (lo[1] + hi[1]), hi[1])
    zs = (lo[2], 0.5 * (lo[2] + hi[2]), hi[2])
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def _cell_needs_refinement(
    lo: np.ndarray,
    hi: np.ndarray,
    domain: "torch.Tensor",
    device: str | torch.device,
    yarn_width: float,
    yarn_height: float,
    undulation: float,
    gap_size: float,
    interface_band: float,
) -> bool:
    samples = torch.as_tensor(_cell_sample_grid_np(lo, hi), dtype=torch.float64, device=device)
    phis = plain_weave_yarn_levelsets(
        samples,
        domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
    )
    classifications = classify_plain_weave_points(
        samples,
        domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
    )
    if torch.unique(classifications).numel() > 1:
        return True
    if bool((phis.abs().min(dim=1).values <= interface_band).any().item()):
        return True
    return False


def plain_weave_octree_voxel_coordinates(
    base_nx: int,
    base_ny: int,
    base_nz: int,
    max_refinement: int,
    domain: DomainLike = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
    device: str | torch.device = "cpu",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
    interface_band: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Create octree-derived rectilinear coordinates clustered near yarn interfaces."""

    if base_nx <= 0 or base_ny <= 0 or base_nz <= 0:
        raise ValueError("base_nx, base_ny, and base_nz must be positive")
    if max_refinement < 0:
        raise ValueError("max_refinement must be non-negative")
    dom = _as_domain_tensor(domain, device=device)
    dom_np = dom.detach().cpu().numpy()
    lo = dom_np[0]
    hi = dom_np[1]
    if interface_band is None:
        base_spacing = min((hi[0] - lo[0]) / base_nx, (hi[1] - lo[1]) / base_ny, (hi[2] - lo[2]) / base_nz)
        interface_band = 0.75 * base_spacing / (2**max_refinement)

    xs0 = np.linspace(lo[0], hi[0], base_nx + 1, dtype=np.float64)
    ys0 = np.linspace(lo[1], hi[1], base_ny + 1, dtype=np.float64)
    zs0 = np.linspace(lo[2], hi[2], base_nz + 1, dtype=np.float64)
    leaves: List[Tuple[np.ndarray, np.ndarray, int]] = []
    stack: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for i in range(base_nx):
        for j in range(base_ny):
            for k in range(base_nz):
                stack.append(
                    (
                        np.asarray([xs0[i], ys0[j], zs0[k]], dtype=np.float64),
                        np.asarray([xs0[i + 1], ys0[j + 1], zs0[k + 1]], dtype=np.float64),
                        0,
                    )
                )

    while stack:
        cell_lo, cell_hi, level = stack.pop()
        if level >= max_refinement or not _cell_needs_refinement(
            cell_lo,
            cell_hi,
            dom,
            device,
            yarn_width,
            yarn_height,
            undulation,
            gap_size,
            float(interface_band),
        ):
            leaves.append((cell_lo, cell_hi, level))
            continue
        mid = 0.5 * (cell_lo + cell_hi)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    child_lo = np.asarray(
                        [
                            cell_lo[0] if dx == 0 else mid[0],
                            cell_lo[1] if dy == 0 else mid[1],
                            cell_lo[2] if dz == 0 else mid[2],
                        ],
                        dtype=np.float64,
                    )
                    child_hi = np.asarray(
                        [
                            mid[0] if dx == 0 else cell_hi[0],
                            mid[1] if dy == 0 else cell_hi[1],
                            mid[2] if dz == 0 else cell_hi[2],
                        ],
                        dtype=np.float64,
                    )
                    stack.append((child_lo, child_hi, level + 1))

    x_points: List[float] = []
    y_points: List[float] = []
    z_points: List[float] = []
    for cell_lo, cell_hi, _level in leaves:
        x_points.extend((float(cell_lo[0]), float(cell_hi[0])))
        y_points.extend((float(cell_lo[1]), float(cell_hi[1])))
        z_points.extend((float(cell_lo[2]), float(cell_hi[2])))

    topology_xs, topology_ys, topology_zs = plain_weave_topology_coordinates(
        nx=base_nx,
        ny=base_ny,
        nz=base_nz,
        domain=domain,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
        interface_levels=1,
        crossing_levels=1,
        gap_levels=1,
    )
    x_points.extend(float(value) for value in topology_xs)
    y_points.extend(float(value) for value in topology_ys)
    z_points.extend(float(value) for value in topology_zs)
    xs = _unique_sorted(x_points, float(lo[0]), float(hi[0]))
    ys = _unique_sorted(y_points, float(lo[1]), float(hi[1]))
    zs = _unique_sorted(z_points, float(lo[2]), float(hi[2]))
    metadata = {
        "octree_leaf_count": len(leaves),
        "octree_max_leaf_level": max((level for _lo, _hi, level in leaves), default=0),
        "octree_interface_band": float(interface_band),
        "topology_anchor_counts": {"x": len(topology_xs), "y": len(topology_ys), "z": len(topology_zs)},
    }
    return xs, ys, zs, metadata


def build_plain_weave_octree_voxel_rve(
    base_nx: int,
    base_ny: int,
    base_nz: int,
    max_refinement: int = 1,
    domain: DomainLike = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
    device: str | torch.device = "cpu",
    yarn_width: float = 0.80,
    yarn_height: float = 0.10,
    undulation: float = 0.040,
    gap_size: float = 0.0,
    interface_band: Optional[float] = None,
    separate_contacts: bool = True,
    interface_refinement_passes: int = 1,
) -> PeriodicHexMesh:
    """Build a torch-classified octree-derived voxel RVE as C3D8 hexahedra."""

    xs, ys, zs, octree_metadata = plain_weave_octree_voxel_coordinates(
        base_nx=base_nx,
        base_ny=base_ny,
        base_nz=base_nz,
        max_refinement=max_refinement,
        domain=domain,
        device=device,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
        interface_band=interface_band,
    )
    xs, ys, zs, interface_metadata = refine_material_interface_coordinates(
        xs,
        ys,
        zs,
        domain=domain,
        device=device,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
        passes=interface_refinement_passes,
        separate_contacts=separate_contacts,
    )
    mesh = generate_periodic_hex_lattice_from_coordinates(xs, ys, zs, device=device)
    material_ids, separated_contact_cells = _classify_hex_centers(
        mesh,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
        undulation=undulation,
        gap_size=gap_size,
        separate_contacts=separate_contacts,
    )
    names = {0: "Matrix"}
    for mat in torch.unique(material_ids).detach().cpu().tolist():
        if int(mat) > 0:
            names[int(mat)] = f"Yarn{int(mat) - 1}"
    mesh.material_ids = material_ids
    mesh.material_names = names
    mesh.metadata.update(
        {
            "source": "torch_octree_voxel",
            "base_nx": base_nx,
            "base_ny": base_ny,
            "base_nz": base_nz,
            "max_refinement": max_refinement,
            "yarn_width": yarn_width,
            "yarn_height": yarn_height,
            "undulation": undulation,
            "effective_undulation": _gap_adjusted_undulation(yarn_height, undulation, gap_size),
            "gap_size": gap_size,
            "separate_contacts": bool(separate_contacts),
            "separated_contact_cells": int(separated_contact_cells),
            "interface_refinement_passes": int(interface_refinement_passes),
            "coordinate_counts": {"x": len(xs), "y": len(ys), "z": len(zs)},
            "rectilinear_cells": int(mesh.elements.shape[0]),
            **octree_metadata,
            **interface_metadata,
        }
    )
    return mesh


def material_elsets(mesh: PeriodicTetMesh) -> Dict[str, List[int]]:
    """Return one-based Abaqus element labels grouped by material name."""

    result: Dict[str, List[int]] = {}
    material_ids = mesh.material_ids.detach().cpu().numpy()
    for mat_id, name in sorted(mesh.material_names.items()):
        labels = np.nonzero(material_ids == mat_id)[0] + 1
        if labels.size:
            result[name] = labels.astype(int).tolist()
    return result


def _write_labels(fh, labels: Sequence[int], width: int = 16) -> None:
    for start in range(0, len(labels), width):
        fh.write(", ".join(str(v) for v in labels[start : start + width]) + "\n")


def write_abaqus_inp(mesh: PeriodicTetMesh, path: str | Path) -> None:
    """Write an Abaqus C3D4 input deck with Matrix/Yarn elsets."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nodes = mesh.nodes.detach().cpu().numpy()
    elements = mesh.elements.detach().cpu().numpy()
    with out.open("w", encoding="utf-8") as fh:
        fh.write("*Heading\n")
        fh.write("** Pure Python/torch periodic plain-weave RVE tetra mesh\n")
        fh.write("*Node\n")
        for idx, (x, y, z) in enumerate(nodes, start=1):
            fh.write(f"{idx}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        fh.write("*Element, Type=C3D4\n")
        for eid, tet in enumerate(elements, start=1):
            n1, n2, n3, n4 = (int(v) + 1 for v in tet)
            fh.write(f"{eid}, {n1}, {n2}, {n3}, {n4}\n")
        fh.write("*ElSet, ElSet=All, Generate\n")
        fh.write(f"1, {elements.shape[0]}, 1\n")
        for name, labels in material_elsets(mesh).items():
            fh.write(f"*ElSet, ElSet={name}\n")
            _write_labels(fh, labels)
        pairs = periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))
        for axis, axis_pairs in pairs.items():
            minus = [a + 1 for a, _b in axis_pairs]
            plus = [b + 1 for _a, b in axis_pairs]
            fh.write(f"*NSet, NSet={axis.upper()}MIN\n")
            _write_labels(fh, minus)
            fh.write(f"*NSet, NSet={axis.upper()}MAX\n")
            _write_labels(fh, plus)


def write_abaqus_voxel_inp(mesh: PeriodicHexMesh, path: str | Path, element_type: str = "C3D8R") -> None:
    """Write an Abaqus hexahedral voxel input deck with Matrix/Yarn elsets."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nodes = mesh.nodes.detach().cpu().numpy()
    elements = mesh.elements.detach().cpu().numpy()
    with out.open("w", encoding="utf-8") as fh:
        fh.write("*Heading\n")
        fh.write("** Pure Python/torch octree-derived plain-weave RVE voxel mesh\n")
        fh.write("*Node\n")
        for idx, (x, y, z) in enumerate(nodes, start=1):
            fh.write(f"{idx}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        fh.write(f"*Element, Type={element_type}\n")
        for eid, hex_element in enumerate(elements, start=1):
            labels = [str(int(v) + 1) for v in hex_element]
            fh.write(f"{eid}, {', '.join(labels)}\n")
        fh.write("*ElSet, ElSet=All, Generate\n")
        fh.write(f"1, {elements.shape[0]}, 1\n")
        for name, labels in material_elsets(mesh).items():
            fh.write(f"*ElSet, ElSet={name}\n")
            _write_labels(fh, labels)
        pairs = periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))
        for axis, axis_pairs in pairs.items():
            minus = [a + 1 for a, _b in axis_pairs]
            plus = [b + 1 for _a, b in axis_pairs]
            fh.write(f"*NSet, NSet={axis.upper()}MIN\n")
            _write_labels(fh, minus)
            fh.write(f"*NSet, NSet={axis.upper()}MAX\n")
            _write_labels(fh, plus)


def write_pbc_pairs_csv(pairs: Mapping[Axis, Sequence[Tuple[int, int]]], path: str | Path) -> None:
    """Write zero-based periodic pairs as one-based node labels for Abaqus use."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["axis", "minus_node", "plus_node"])
        for axis in sorted(pairs):
            for minus, plus in pairs[axis]:
                writer.writerow([axis, minus + 1, plus + 1])


def count_different_yarn_shared_faces(mesh: PeriodicTetMesh) -> int:
    """Count internal faces directly shared by two different yarn materials."""

    faces: Dict[Tuple[int, ...], int] = {}
    count = 0
    if mesh.elements.shape[1] == 4:
        local_faces: Sequence[Sequence[int]] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    elif mesh.elements.shape[1] == 8:
        local_faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    else:
        raise ValueError("unsupported element topology")
    elements = mesh.elements.detach().cpu().numpy()
    material_ids = mesh.material_ids.detach().cpu().numpy()
    for element_index, tet in enumerate(elements):
        material = int(material_ids[element_index])
        for face in local_faces:
            key = tuple(sorted(int(tet[index]) for index in face))
            if key in faces:
                other_material = faces.pop(key)
                if material > 0 and other_material > 0 and material != other_material:
                    count += 1
            else:
                faces[key] = material
    return count


def _axis_aligned_face_area(nodes: np.ndarray, labels: Sequence[int]) -> float:
    pts = nodes[list(labels)]
    extents = np.ptp(pts, axis=0)
    nonzero = [float(value) for value in extents if float(value) > 1e-14]
    if len(nonzero) < 2:
        return 0.0
    return float(nonzero[0] * nonzero[1])


def material_interface_face_summary(mesh: PeriodicHexMesh) -> Dict[str, float | int]:
    """Summarize internal faces shared by different material ids."""

    if mesh.elements.shape[1] != 8:
        raise ValueError("material_interface_face_summary currently supports C3D8 voxel meshes")
    local_faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    elements = mesh.elements.detach().cpu().numpy()
    material_ids = mesh.material_ids.detach().cpu().numpy()
    nodes = mesh.nodes.detach().cpu().numpy()
    faces: Dict[Tuple[int, ...], Tuple[int, Tuple[int, ...]]] = {}
    areas: List[float] = []
    matrix_yarn_faces = 0
    yarn_yarn_faces = 0

    for element_index, element in enumerate(elements):
        material = int(material_ids[element_index])
        for face in local_faces:
            labels = tuple(int(element[index]) for index in face)
            key = tuple(sorted(labels))
            if key in faces:
                other_material, other_labels = faces.pop(key)
                if material != other_material:
                    areas.append(_axis_aligned_face_area(nodes, labels))
                    if material > 0 and other_material > 0:
                        yarn_yarn_faces += 1
                    else:
                        matrix_yarn_faces += 1
            else:
                faces[key] = (material, labels)

    return {
        "faces": int(len(areas)),
        "matrix_yarn_faces": int(matrix_yarn_faces),
        "yarn_yarn_faces": int(yarn_yarn_faces),
        "max_area": float(max(areas)) if areas else 0.0,
        "min_area": float(min(areas)) if areas else 0.0,
        "mean_area": float(sum(areas) / len(areas)) if areas else 0.0,
    }


def mesh_quality_summary(mesh: PeriodicTetMesh) -> Dict[str, object]:
    """Return basic volume and material-count diagnostics."""

    volumes = tet_volumes(mesh.nodes, mesh.elements)
    dom = mesh.domain.detach().cpu().numpy()
    bbox_volume = float(np.prod(dom[1] - dom[0]))
    material_counts = {
        mesh.material_names.get(int(mat), str(int(mat))): int((mesh.material_ids == int(mat)).sum().item())
        for mat in torch.unique(mesh.material_ids).detach().cpu().tolist()
    }
    material_volumes = {
        mesh.material_names.get(int(mat), str(int(mat))): float(volumes[mesh.material_ids == int(mat)].sum().item())
        for mat in torch.unique(mesh.material_ids).detach().cpu().tolist()
    }
    return {
        "nodes": int(mesh.nodes.shape[0]),
        "elements": int(mesh.elements.shape[0]),
        "zero_volume_tets_le_1e-14": int((volumes <= 1e-14).sum().item()),
        "total_abs_volume": float(volumes.sum().item()),
        "bbox_volume": bbox_volume,
        "min_abs_volume": float(volumes.min().item()) if volumes.numel() else 0.0,
        "max_abs_volume": float(volumes.max().item()) if volumes.numel() else 0.0,
        "different_yarn_shared_faces": count_different_yarn_shared_faces(mesh),
        "material_counts": material_counts,
        "material_volumes": material_volumes,
    }


def hex_mesh_quality_summary(mesh: PeriodicHexMesh) -> Dict[str, object]:
    """Return basic volume and material-count diagnostics for voxel hexahedra."""

    volumes = hex_volumes(mesh.nodes, mesh.elements)
    dom = mesh.domain.detach().cpu().numpy()
    bbox_volume = float(np.prod(dom[1] - dom[0]))
    material_counts = {
        mesh.material_names.get(int(mat), str(int(mat))): int((mesh.material_ids == int(mat)).sum().item())
        for mat in torch.unique(mesh.material_ids).detach().cpu().tolist()
    }
    material_volumes = {
        mesh.material_names.get(int(mat), str(int(mat))): float(volumes[mesh.material_ids == int(mat)].sum().item())
        for mat in torch.unique(mesh.material_ids).detach().cpu().tolist()
    }
    return {
        "nodes": int(mesh.nodes.shape[0]),
        "elements": int(mesh.elements.shape[0]),
        "zero_volume_hexes_le_1e-14": int((volumes <= 1e-14).sum().item()),
        "total_abs_volume": float(volumes.sum().item()),
        "bbox_volume": bbox_volume,
        "min_abs_volume": float(volumes.min().item()) if volumes.numel() else 0.0,
        "max_abs_volume": float(volumes.max().item()) if volumes.numel() else 0.0,
        "different_yarn_shared_faces": count_different_yarn_shared_faces(mesh),
        "material_counts": material_counts,
        "material_volumes": material_volumes,
    }


__all__ = [
    "PeriodicHexMesh",
    "PeriodicTetMesh",
    "build_plain_weave_octree_voxel_rve",
    "build_plain_weave_rve",
    "classify_plain_weave_points",
    "cut_tet_mesh_by_levelset",
    "generate_periodic_tet_lattice",
    "generate_periodic_tet_lattice_from_coordinates",
    "generate_periodic_hex_lattice_from_coordinates",
    "count_different_yarn_shared_faces",
    "hex_mesh_quality_summary",
    "hex_volumes",
    "material_interface_face_summary",
    "material_elsets",
    "mesh_quality_summary",
    "periodic_node_pairs",
    "plain_weave_levelset",
    "plain_weave_octree_voxel_coordinates",
    "plain_weave_topology_coordinates",
    "plain_weave_yarn_levelsets",
    "refine_material_interface_coordinates",
    "separate_yarn_contact_cells",
    "tet_signed_volumes",
    "tet_volumes",
    "write_abaqus_inp",
    "write_abaqus_voxel_inp",
    "write_pbc_pairs_csv",
]
