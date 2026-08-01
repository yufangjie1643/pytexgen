"""Portable NumPy reproduction of TexGen's point-in-yarn classifier.

The module deliberately separates the one-time SWIG geometry extraction from
the hot classification loop.  :class:`NumpyExactGeometry` contains only NumPy
arrays and ordinary Python values, so a cached instance can be voxelized
repeatedly without calling back into TexGen's C++ objects.

This is an exact-compatibility path, not the slave-node approximation used by
``classification="tensor"``.  Unsupported TexGen geometry is rejected during
extraction instead of being classified approximately.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


_CONVERGENCE_TOLERANCE = 1.0e-6
_POLYNOMIAL_SAMPLES = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
_POLYNOMIAL_VANDERMONDE = np.stack(
    [
        np.ones(4),
        _POLYNOMIAL_SAMPLES,
        _POLYNOMIAL_SAMPLES**2,
        _POLYNOMIAL_SAMPLES**3,
    ],
    axis=1,
)
_POLYNOMIAL_VANDERMONDE_INV = np.linalg.inv(_POLYNOMIAL_VANDERMONDE)


def _xyz(value: Any) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z], dtype=np.float64)


def _xy(value: Any) -> np.ndarray:
    return np.asarray([value.x, value.y], dtype=np.float64)


def _normalise(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(
        values,
        lengths,
        out=np.zeros_like(values),
        where=lengths != 0.0,
    )


def _rotate_about_axis(values: np.ndarray,
                       axes: np.ndarray,
                       angles: np.ndarray) -> np.ndarray:
    cos_angle = np.cos(angles)[..., None]
    sin_angle = np.sin(angles)[..., None]
    return (
        values * cos_angle
        + np.cross(axes, values) * sin_angle
        + axes
        * np.sum(axes * values, axis=-1, keepdims=True)
        * (1.0 - cos_angle)
    )


@dataclass(frozen=True)
class NumpyExactYarnGeometry:
    """Array-only description of one TexGen yarn."""

    curve_coefficients: np.ndarray  # (segments, 4, 3), ascending powers
    bezier_control_points: Optional[np.ndarray]  # (segments, 4, 3)
    master_ups: np.ndarray          # (segments + 1, 3), unprojected
    master_angles: np.ndarray       # (segments + 1,)
    linear_tangents: Optional[np.ndarray]  # (segments, 2, 3), linear only
    section_points: np.ndarray      # (all segment knots, section_points, 2)
    mesh_nodes: np.ndarray          # (all segment knots, mesh_nodes, 2)
    section_knot_positions: np.ndarray  # flattened local-u knots
    section_knot_offsets: np.ndarray    # (segments + 1,)
    section_lengths: np.ndarray         # physical master-segment lengths
    triangle_indices: np.ndarray    # (triangles, 3)
    quad_indices: np.ndarray        # (quads, 4)
    translations: np.ndarray       # (images, 3), TexGen order
    yarn_aabb: np.ndarray           # (2, 3), untranslated
    segment_aabbs: np.ndarray       # (segments, 2, 3), untranslated
    interpolation_type: str
    section_type: str
    force_in_plane_tangent: bool
    ramped: bool
    polar: bool
    constant_section: bool
    position_interpolated_section: bool

    @property
    def num_segments(self) -> int:
        return int(self.curve_coefficients.shape[0])


@dataclass(frozen=True)
class NumpyExactGeometry:
    """Reusable pure-NumPy snapshot for TexGen-compatible classification."""

    yarns: Tuple[NumpyExactYarnGeometry, ...]
    aabb: np.ndarray

    @property
    def num_yarns(self) -> int:
        return len(self.yarns)


def _mesh_arrays(mesh: Any,
                 mesh_type: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [[node.x, node.y] for node in mesh.GetNodes()], dtype=np.float64
    )
    triangles = np.asarray(
        list(mesh.GetIndices(mesh_type.TRI)), dtype=np.int64
    ).reshape(-1, 3)
    quads = np.asarray(
        list(mesh.GetIndices(mesh_type.QUAD)), dtype=np.int64
    ).reshape(-1, 4)
    return nodes, triangles, quads


def _extract_sections(yarn: Any,
                      num_master_nodes: int,
                      core: Any) -> Tuple[
                          np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, str, bool, bool, bool
                      ]:
    yarn_section = yarn.GetYarnSection()
    section_type = str(yarn_section.GetType())
    num_points = int(yarn.GetNumSectionPoints())
    raw_section_lengths = np.asarray(
        yarn.GetYarnSectionLengths(), dtype=np.float64
    )

    if section_type == "CYarnSectionConstant":
        concrete = yarn_section.GetSectionConstant()
        ramped = False
        polar = False
        constant = True
    elif section_type == "CYarnSectionInterpNode":
        concrete = yarn_section.GetSectionInterpNode()
        count = int(concrete.GetNumNodeSections())
        if count != num_master_nodes:
            raise NotImplementedError(
                "numpy_exact requires one CYarnSectionInterpNode section per "
                f"master node; received {count} sections for {num_master_nodes} nodes"
            )
        ramped = bool(concrete.GetRamped())
        polar = bool(concrete.GetPolar())
        constant = False
    elif section_type == "CYarnSectionInterpPosition":
        concrete = yarn_section.GetSectionInterpPosition()
        section_positions = np.asarray(
            [
                float(concrete.GetSectionPosition(index))
                for index in range(int(concrete.GetNumNodeSections()))
            ],
            dtype=np.float64,
        )
        if section_positions.size == 0:
            raise NotImplementedError(
                "numpy_exact requires at least one position-interpolated section"
            )
        ramped = bool(concrete.GetRamped())
        polar = bool(concrete.GetPolar())
        constant = False
    else:
        raise NotImplementedError(
            "numpy_exact supports constant, node-interpolated, and "
            f"position-interpolated sections, not {section_type}"
        )

    section_lengths = core.DoubleVector(raw_section_lengths)
    info = core.YARN_POSITION_INFORMATION()
    info.SectionLengths = section_lengths
    points = []
    mesh_nodes = []
    knot_positions = []
    knot_offsets = [0]
    triangles = None
    quads = None
    if section_type == "CYarnSectionInterpPosition":
        for index, position in enumerate(section_positions):
            section = concrete.GetSection(index)
            points.append(
                [
                    [point.x, point.y]
                    for point in section.GetPoints(num_points, False)
                ]
            )
            mesh = section.GetMesh(num_points, False)
            nodes, section_triangles, section_quads = _mesh_arrays(
                mesh, core.CMesh
            )
            if triangles is None:
                triangles = section_triangles
                quads = section_quads
            elif (
                nodes.shape != mesh_nodes[0].shape
                or not np.array_equal(section_triangles, triangles)
                or not np.array_equal(section_quads, quads)
            ):
                raise NotImplementedError(
                    "numpy_exact requires compatible position-interpolated "
                    "section-mesh topology"
                )
            mesh_nodes.append(nodes)
        assert triangles is not None and quads is not None
        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(mesh_nodes, dtype=np.float64),
            triangles,
            quads,
            section_positions,
            np.zeros(num_master_nodes, dtype=np.int64),
            section_type,
            ramped,
            polar,
            constant,
        )

    for segment in range(num_master_nodes - 1):
        if section_type == "CYarnSectionInterpNode":
            mids = [
                float(concrete.GetMidNodeSectionPos(segment, index))
                for index in range(
                    int(concrete.GetNumMidNodeSections(segment))
                )
            ]
        else:
            mids = []
        segment_knots = [0.0, *mids, 1.0]
        if any(
            right <= left
            for left, right in zip(segment_knots, segment_knots[1:])
        ):
            raise NotImplementedError(
                "numpy_exact requires strictly increasing mid-node section positions"
            )
        for position in segment_knots:
            info.iSection = segment
            info.dSectionPosition = position
            section = yarn_section.GetSection(info, num_points, False)
            points.append(
                [[point.x, point.y] for point in section]
            )
            mesh = yarn_section.GetSectionMesh(info, num_points, False)
            nodes, section_triangles, section_quads = _mesh_arrays(
                mesh, core.CMesh
            )
            if triangles is None:
                triangles = section_triangles
                quads = section_quads
            elif (
                nodes.shape != mesh_nodes[0].shape
                or not np.array_equal(section_triangles, triangles)
                or not np.array_equal(section_quads, quads)
            ):
                raise NotImplementedError(
                    "numpy_exact requires compatible section-mesh topology at all "
                    "node and mid-node sections"
                )
            mesh_nodes.append(nodes)
            knot_positions.append(position)
        knot_offsets.append(len(knot_positions))

    assert triangles is not None and quads is not None
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(mesh_nodes, dtype=np.float64),
        triangles,
        quads,
        np.asarray(knot_positions, dtype=np.float64),
        np.asarray(knot_offsets, dtype=np.int64),
        section_type,
        ramped,
        polar,
        constant,
    )


def _extract_curve(yarn: Any) -> Tuple[
        np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray,
        Optional[np.ndarray], str, bool
]:
    interpolation = yarn.GetInterpolation()
    interpolation_type = str(interpolation.GetType())
    supported = {
        "CInterpolationBezier",
        "CInterpolationCubic",
        "CInterpolationLinear",
    }
    if interpolation_type not in supported:
        raise NotImplementedError(
            "numpy_exact supports Bezier, cubic, and linear interpolation, "
            f"not {interpolation_type}"
        )

    masters = yarn.GetMasterNodes()
    num_masters = len(masters)
    if num_masters < 2:
        raise ValueError("numpy_exact requires at least two master nodes per yarn")

    # Initialise mutable interpolation caches once, while still in the serial
    # extraction phase. The resulting snapshot never calls SWIG again.
    interpolation.Initialise(masters)
    coefficients = np.empty((num_masters - 1, 4, 3), dtype=np.float64)
    bezier_control_points = (
        np.empty((num_masters - 1, 4, 3), dtype=np.float64)
        if interpolation_type == "CInterpolationBezier"
        else None
    )
    linear_tangents = (
        np.empty((num_masters - 1, 2, 3), dtype=np.float64)
        if interpolation_type == "CInterpolationLinear"
        else None
    )
    for segment in range(num_masters - 1):
        sampled_nodes = [
            interpolation.GetNode(masters, segment, float(u))
            for u in _POLYNOMIAL_SAMPLES
        ]
        sampled = np.asarray([_xyz(node.GetPosition()) for node in sampled_nodes])
        if interpolation_type == "CInterpolationLinear":
            start = _xyz(masters[segment].GetPosition())
            end = _xyz(masters[segment + 1].GetPosition())
            coefficients[segment, 0] = start
            coefficients[segment, 1] = end - start
            coefficients[segment, 2:] = 0.0
        else:
            coefficients[segment] = _POLYNOMIAL_VANDERMONDE_INV @ sampled
        if bezier_control_points is not None:
            start = _xyz(masters[segment].GetPosition())
            end = _xyz(masters[segment + 1].GetPosition())
            tangent1 = _xyz(sampled_nodes[0].GetTangent())
            tangent2 = _xyz(sampled_nodes[-1].GetTangent())
            third_length = float(np.linalg.norm(end - start)) / 3.0
            bezier_control_points[segment] = np.asarray(
                [
                    start,
                    start + tangent1 * third_length,
                    end - tangent2 * third_length,
                    end,
                ]
            )
        if linear_tangents is not None:
            linear_tangents[segment, 0] = _xyz(
                interpolation.GetNode(masters, segment, 0.0).GetTangent()
            )
            linear_tangents[segment, 1] = _xyz(
                interpolation.GetNode(masters, segment, 1.0).GetTangent()
            )

        # Guard against future interpolation implementations claiming one of
        # these names while no longer being polynomial in the local parameter.
        for check_u in (0.2, 0.7):
            reference = _xyz(
                interpolation.GetNode(masters, segment, check_u).GetPosition()
            )
            estimate = _evaluate_position(
                coefficients[segment],
                np.asarray([check_u]),
                interpolation_type=interpolation_type,
                bezier_control_points=(
                    None
                    if bezier_control_points is None
                    else bezier_control_points[segment]
                ),
            )[0]
            scale = max(1.0, float(np.linalg.norm(reference)))
            if not np.allclose(estimate, reference, rtol=0.0, atol=2.0e-12 * scale):
                raise NotImplementedError(
                    f"{interpolation_type} segment {segment} is not reproducible "
                    "as a cubic polynomial"
                )

    master_ups = np.asarray([_xyz(node.GetUp()) for node in masters])
    zero_up = np.linalg.norm(master_ups, axis=1) == 0.0
    master_ups[zero_up] = np.asarray([0.0, 0.0, 1.0])
    master_angles = np.asarray(
        [float(node.GetAngle()) for node in masters], dtype=np.float64
    )
    return (
        coefficients,
        bezier_control_points,
        master_ups,
        master_angles,
        linear_tangents,
        interpolation_type,
        bool(interpolation.GetForceInPlaneTangent()),
    )


def _aabb_array(bounds: Sequence[Any]) -> np.ndarray:
    return np.asarray([_xyz(bounds[0]), _xyz(bounds[1])], dtype=np.float64)


def extract_numpy_exact_geometry(textile: Any) -> NumpyExactGeometry:
    """Extract a reusable array-only exact geometry snapshot from ``CTextile``.

    Extraction is the only phase that accesses TexGen/SWIG objects. Unsupported
    geometry raises :class:`NotImplementedError` rather than silently using an
    approximate classifier.
    """
    try:
        from . import Core as core
    except ImportError:  # Legacy source-tree package name.
        import TexGen.Core as core

    CYarn = core.CYarn
    usage = int(CYarn.LINE) | int(CYarn.SURFACE) | int(CYarn.VOLUME)
    domain = textile.GetDomain()
    if domain is None:
        raise ValueError("numpy_exact requires an assigned TexGen domain")
    domain_points = np.asarray(
        [
            [item.x, item.y, item.z]
            for item in domain.GetMesh().GetNodes()
        ],
        dtype=np.float64,
    )
    if domain_points.size == 0:
        raise ValueError("numpy_exact cannot determine an AABB from an empty domain")
    aabb = np.asarray(
        [domain_points.min(axis=0), domain_points.max(axis=0)], dtype=np.float64
    )
    # The rectangular exact voxelizer asks CDomainPlanes spanning the point
    # AABB for periodic images, even when the textile owns a sheared domain.
    # Using the full voxel AABB is a conservative superset of its centre AABB;
    # extra images cannot contain a centre and therefore do not alter ordering.
    classifier_domain = core.CDomainPlanes(
        core.XYZ(*aabb[0]), core.XYZ(*aabb[1])
    )

    yarns = []
    for yarn_index in range(int(textile.GetNumYarns())):
        yarn = textile.GetYarn(yarn_index)
        yarn.GetSlaveNodes(usage)
        (
            coefficients,
            bezier_control_points,
            master_ups,
            master_angles,
            linear_tangents,
            interpolation_type,
            force_in_plane,
        ) = _extract_curve(yarn)
        (
            section_points,
            mesh_nodes,
            triangle_indices,
            quad_indices,
            section_knot_positions,
            section_knot_offsets,
            section_type,
            ramped,
            polar,
            constant_section,
        ) = _extract_sections(yarn, master_ups.shape[0], core)

        translations = np.asarray(
            [
                [item.x, item.y, item.z]
                for item in classifier_domain.GetTranslations(yarn)
            ],
            dtype=np.float64,
        )
        if translations.size == 0:
            translations = np.zeros((1, 3), dtype=np.float64)
        translations = translations.reshape(-1, 3)
        segment_aabbs = np.asarray(
            [
                _aabb_array(yarn.GetSectionAABB(segment))
                for segment in range(coefficients.shape[0])
            ],
            dtype=np.float64,
        )
        yarns.append(
            NumpyExactYarnGeometry(
                curve_coefficients=coefficients,
                bezier_control_points=bezier_control_points,
                master_ups=master_ups,
                master_angles=master_angles,
                linear_tangents=linear_tangents,
                section_points=section_points,
                mesh_nodes=mesh_nodes,
                section_knot_positions=section_knot_positions,
                section_knot_offsets=section_knot_offsets,
                section_lengths=np.asarray(
                    yarn.GetYarnSectionLengths(), dtype=np.float64
                ),
                triangle_indices=triangle_indices,
                quad_indices=quad_indices,
                translations=translations,
                yarn_aabb=_aabb_array(yarn.GetAABB()),
                segment_aabbs=segment_aabbs,
                interpolation_type=interpolation_type,
                section_type=section_type,
                force_in_plane_tangent=force_in_plane,
                ramped=ramped,
                polar=polar,
                constant_section=constant_section,
                position_interpolated_section=(
                    section_type == "CYarnSectionInterpPosition"
                ),
            )
        )

    return NumpyExactGeometry(yarns=tuple(yarns), aabb=aabb)


def _evaluate_position(
    coefficients: np.ndarray,
    u: np.ndarray,
    *,
    interpolation_type: str = "CInterpolationCubic",
    bezier_control_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    if interpolation_type == "CInterpolationBezier":
        assert bezier_control_points is not None
        mu = u[:, None]
        mum1 = 1.0 - mu
        p1, p2, p3, p4 = bezier_control_points
        return (
            (mum1 * mum1 * mum1) * p1
            + (3.0 * mu * mum1 * mum1) * p2
            + (3.0 * mu * mu * mum1) * p3
            + (mu * mu * mu) * p4
        )
    return (
        (
            coefficients[3][None, :] * u[:, None]
            + coefficients[2][None, :]
        )
        * u[:, None]
        + coefficients[1][None, :]
    ) * u[:, None] + coefficients[0][None, :]


def _evaluate_frame(yarn: NumpyExactYarnGeometry,
                    segment: int,
                    u: np.ndarray) -> Tuple[
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray
                    ]:
    coefficients = yarn.curve_coefficients[segment]
    bezier_controls = (
        None
        if yarn.bezier_control_points is None
        else yarn.bezier_control_points[segment]
    )
    position = _evaluate_position(
        coefficients,
        u,
        interpolation_type=yarn.interpolation_type,
        bezier_control_points=bezier_controls,
    )
    if bezier_controls is not None:
        mu = u[:, None]
        mum1 = 1.0 - mu
        p1, p2, p3, p4 = bezier_controls
        tangent = (
            (-3.0 * mum1 * mum1) * p1
            + (3.0 * ((mum1 * mum1) - 2.0 * mu * mum1)) * p2
            + (3.0 * (2.0 * mu * mum1 - mu * mu)) * p3
            + (3.0 * mu * mu) * p4
        )
    elif yarn.linear_tangents is None:
        tangent = (
            coefficients[1][None, :]
            + 2.0 * coefficients[2][None, :] * u[:, None]
            + 3.0 * coefficients[3][None, :] * (u * u)[:, None]
        )
    else:
        endpoints = yarn.linear_tangents[segment]
        tangent = endpoints[0] + (endpoints[1] - endpoints[0]) * u[:, None]
    if yarn.force_in_plane_tangent:
        tangent[:, 2] = 0.0
    tangent = _normalise(tangent)

    up1 = yarn.master_ups[segment]
    up2 = yarn.master_ups[segment + 1]
    up = up1 + (up2 - up1) * u[:, None]
    up = _normalise(up)
    up = up - tangent * np.sum(tangent * up, axis=1, keepdims=True)
    up = _normalise(up)

    angle1 = yarn.master_angles[segment]
    angle2 = yarn.master_angles[segment + 1]
    angle = angle1 + (angle2 - angle1) * u
    side = _rotate_about_axis(np.cross(tangent, up), up, angle)
    normal = _rotate_about_axis(tangent, up, angle)
    return position, tangent, up, side, normal, angle


def _interpolate_arrays(first: np.ndarray,
                        second: np.ndarray,
                        u: np.ndarray,
                        ramped: bool,
                        polar: bool) -> np.ndarray:
    if first.ndim == 2:
        first = first[None, :, :]
        second = second[None, :, :]
    fraction = 3.0 * u * u - 2.0 * u * u * u if ramped else u
    fraction = fraction[:, None, None]
    if not polar:
        return first + (second - first) * fraction

    length1 = np.linalg.norm(first, axis=2)
    length2 = np.linalg.norm(second, axis=2)
    angle1 = np.arctan2(first[..., 1], first[..., 0])
    angle2 = np.arctan2(second[..., 1], second[..., 0])
    delta = angle2 - angle1
    angle1 = np.where(delta > math.pi, angle1 + 2.0 * math.pi, angle1)
    angle2 = np.where(delta < -math.pi, angle2 + 2.0 * math.pi, angle2)
    f = fraction[..., 0]
    length = length1 + (length2 - length1) * f
    angle = angle1 + (angle2 - angle1) * f
    return np.stack([length * np.cos(angle), length * np.sin(angle)], axis=-1)


def _interpolate_section_knots(values: np.ndarray,
                               yarn: NumpyExactYarnGeometry,
                               segment: int,
                               u: np.ndarray) -> np.ndarray:
    if yarn.position_interpolated_section:
        total_length = float(yarn.section_lengths.sum())
        yarn_position = (
            float(yarn.section_lengths[:segment].sum())
            + yarn.section_lengths[segment] * u
        ) / total_length
        positions = yarn.section_knot_positions
        left_index = np.searchsorted(
            positions, yarn_position, side="right"
        ) - 1
        right_index = np.searchsorted(
            positions, yarn_position, side="left"
        )
        wrapped_left = left_index < 0
        wrapped_right = right_index >= positions.size
        left_index = np.where(wrapped_left, positions.size - 1, left_index)
        right_index = np.where(wrapped_right, 0, right_index)
        left_position = positions[left_index] - wrapped_left.astype(np.float64)
        right_position = positions[right_index] + wrapped_right.astype(np.float64)
        same = left_position == right_position
        fraction = np.divide(
            yarn_position - left_position,
            right_position - left_position,
            out=np.zeros_like(yarn_position),
            where=~same,
        )
        return _interpolate_arrays(
            values[left_index],
            values[right_index],
            fraction,
            yarn.ramped,
            yarn.polar,
        )

    start = int(yarn.section_knot_offsets[segment])
    stop = int(yarn.section_knot_offsets[segment + 1])
    knots = yarn.section_knot_positions[start:stop]
    segment_values = values[start:stop]
    # side="left" is important: TexGen assigns a point exactly on a mid-node
    # section to the interval ending at that section.
    interval = np.searchsorted(knots[1:-1], u, side="left")
    left = knots[interval]
    right = knots[interval + 1]
    local_u = (u - left) / (right - left)
    return _interpolate_arrays(
        segment_values[interval],
        segment_values[interval + 1],
        local_u,
        yarn.ramped,
        yarn.polar,
    )


def _section_points(yarn: NumpyExactYarnGeometry,
                    segment: int,
                    u: np.ndarray) -> np.ndarray:
    return _interpolate_section_knots(
        yarn.section_points, yarn, segment, u
    )


def _section_mesh_nodes(yarn: NumpyExactYarnGeometry,
                        segment: int,
                        u: np.ndarray) -> np.ndarray:
    return _interpolate_section_knots(
        yarn.mesh_nodes, yarn, segment, u
    )


def _points_inside_polygons(points: np.ndarray,
                            polygons: np.ndarray) -> np.ndarray:
    """Vectorized transcription of TexGen's Paul Bourke ray test."""
    count = np.zeros(points.shape[0], dtype=np.int32)
    p1 = polygons[:, 0, :]
    for index in range(1, polygons.shape[1] + 1):
        p2 = polygons[:, index % polygons.shape[1], :]
        condition = (
            (points[:, 1] > np.minimum(p1[:, 1], p2[:, 1]))
            & (points[:, 1] <= np.maximum(p1[:, 1], p2[:, 1]))
            & (points[:, 0] <= np.maximum(p1[:, 0], p2[:, 0]))
            & (p1[:, 1] != p2[:, 1])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            intercept = (
                (points[:, 1] - p1[:, 1])
                * (p2[:, 0] - p1[:, 0])
                / (p2[:, 1] - p1[:, 1])
                + p1[:, 0]
            )
        condition &= (p1[:, 0] == p2[:, 0]) | (points[:, 0] <= intercept)
        count += condition
        p1 = p2
    return (count & 1) == 1


def _surface_distance(points: np.ndarray,
                      polygons: np.ndarray,
                      tolerance: float) -> np.ndarray:
    closest = np.ones(points.shape[0], dtype=np.float64)
    started = np.zeros(points.shape[0], dtype=bool)
    p1 = polygons[:, 0, :]
    for index in range(1, polygons.shape[1] + 1):
        p2 = polygons[:, index % polygons.shape[1], :]
        edge = p2 - p1
        normal = _normalise(np.stack([edge[:, 1], -edge[:, 0]], axis=1))
        distance = np.sum(normal * (points - p1), axis=1)
        eligible = distance < tolerance
        first = eligible & ~started
        closest[first] = distance[first]
        later = eligible & started
        closest[later] = np.maximum(closest[later], distance[later])
        started |= eligible
        p1 = p2
    return closest


def _find_plane_parameters(points: np.ndarray,
                           yarn: NumpyExactYarnGeometry,
                           segment: int) -> Tuple[np.ndarray, np.ndarray]:
    count = points.shape[0]
    zero = np.zeros(1, dtype=np.float64)
    one = np.ones(1, dtype=np.float64)
    start_position, _, _, _, start_normal, _ = _evaluate_frame(
        yarn, segment, zero
    )
    end_position, _, _, _, end_normal, _ = _evaluate_frame(yarn, segment, one)
    normal1 = start_normal[0]
    normal2 = -end_normal[0]
    d1 = (points - start_position[0]) @ normal1
    d2 = (points - end_position[0]) @ normal2
    valid = (d1 >= -_CONVERGENCE_TOLERANCE) & (
        d2 >= -_CONVERGENCE_TOLERANCE
    )
    u = np.zeros(count, dtype=np.float64)

    at_start = valid & (d1 == 0.0)
    at_end = valid & ~at_start & (d2 == 0.0)
    solving = valid & ~at_start & ~at_end
    u[at_end] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        u[solving] = d1[solving] / (d1[solving] + d2[solving])
    du = u.copy()
    previous = np.minimum(d1, d2)

    active = solving & (np.abs(du) > _CONVERGENCE_TOLERANCE)
    for _ in range(101):
        if not np.any(active):
            break
        active_indices = np.flatnonzero(active)
        active_u = u[active_indices]
        position, _, _, _, normal, _ = _evaluate_frame(
            yarn, segment, active_u
        )
        distance = np.sum(normal * (points[active_indices] - position), axis=1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            next_du = distance * (
                du[active_indices] / (previous[active_indices] - distance)
            )
        upper = 1.0 - active_u + _CONVERGENCE_TOLERANCE
        lower = -active_u - _CONVERGENCE_TOLERANCE
        next_du = np.minimum(np.maximum(next_du, lower), upper)
        next_u = active_u + next_du
        finite = np.isfinite(next_u) & np.isfinite(next_du)
        u[active_indices] = next_u
        du[active_indices] = next_du
        previous[active_indices] = distance
        bad_indices = active_indices[~finite]
        valid[bad_indices] = False
        active = solving & valid & (np.abs(du) > _CONVERGENCE_TOLERANCE)
    else:
        # C++ returns false after exceeding 100 iterations.
        valid[active] = False
    return u, valid


def _inside_one_segment(points: np.ndarray,
                        yarn: NumpyExactYarnGeometry,
                        segment: int,
                        tolerance: float) -> Tuple[
                            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray
                        ]:
    u, valid = _find_plane_parameters(points, yarn, segment)
    inside = np.zeros(points.shape[0], dtype=bool)
    distance = np.ones(points.shape[0], dtype=np.float64)
    location = np.zeros((points.shape[0], 2), dtype=np.float64)
    tangent = np.zeros((points.shape[0], 3), dtype=np.float64)
    up = np.zeros((points.shape[0], 3), dtype=np.float64)
    if not np.any(valid):
        return inside, distance, u, location, tangent, up

    indices = np.flatnonzero(valid)
    position, local_tangent, local_up, side, _, _ = _evaluate_frame(
        yarn, segment, u[indices]
    )
    relative = points[indices] - position
    local = np.stack(
        [
            np.sum(relative * side, axis=1),
            np.sum(relative * local_up, axis=1),
        ],
        axis=1,
    )
    polygons = _section_points(yarn, segment, u[indices])
    minimum = polygons.min(axis=1)
    maximum = polygons.max(axis=1)
    in_bounds = np.all((local >= minimum) & (local <= maximum), axis=1)
    if np.any(in_bounds):
        bounded = np.flatnonzero(in_bounds)
        polygon_inside = _points_inside_polygons(
            local[bounded], polygons[bounded]
        )
        accepted_local = bounded[polygon_inside]
        accepted = indices[accepted_local]
        inside[accepted] = True
        distance[accepted] = _surface_distance(
            local[accepted_local], polygons[accepted_local], tolerance
        )
    location[indices] = local
    tangent[indices] = local_tangent
    up[indices] = local_up
    return inside, distance, u, location, tangent, up


def _inside_one_yarn(points: np.ndarray,
                     yarn: NumpyExactYarnGeometry,
                     tolerance: float) -> Tuple[
                         np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray
                     ]:
    count = points.shape[0]
    found = np.zeros(count, dtype=bool)
    distance = np.ones(count, dtype=np.float64)
    segment_result = np.full(count, -1, dtype=np.int32)
    u_result = np.zeros(count, dtype=np.float64)
    location_result = np.zeros((count, 2), dtype=np.float64)
    up_result = np.zeros((count, 3), dtype=np.float64)

    # TexGen returns the first containing periodic image, then the first
    # containing master-node segment within that image.
    for translation in yarn.translations:
        remaining = ~found
        if not np.any(remaining):
            break
        local_points = points - translation
        yarn_candidate = remaining & np.all(
            (local_points >= yarn.yarn_aabb[0] - tolerance)
            & (local_points <= yarn.yarn_aabb[1] + tolerance),
            axis=1,
        )
        for segment in range(yarn.num_segments):
            candidate = yarn_candidate & ~found & np.all(
                (local_points >= yarn.segment_aabbs[segment, 0] - tolerance)
                & (local_points <= yarn.segment_aabbs[segment, 1] + tolerance),
                axis=1,
            )
            if not np.any(candidate):
                continue
            candidate_indices = np.flatnonzero(candidate)
            (
                local_inside,
                local_distance,
                local_u,
                local_location,
                _,
                local_up,
            ) = _inside_one_segment(
                local_points[candidate_indices], yarn, segment, tolerance
            )
            accepted = candidate_indices[local_inside]
            if accepted.size == 0:
                continue
            accepted_local = np.flatnonzero(local_inside)
            found[accepted] = True
            distance[accepted] = local_distance[accepted_local]
            segment_result[accepted] = segment
            u_result[accepted] = local_u[accepted_local]
            location_result[accepted] = local_location[accepted_local]
            up_result[accepted] = local_up[accepted_local]
    return found, distance, segment_result, u_result, location_result, up_result


def _select_mesh_elements(points: np.ndarray,
                          mesh_nodes: np.ndarray,
                          yarn: NumpyExactYarnGeometry) -> Tuple[np.ndarray, np.ndarray]:
    element_type = np.zeros(points.shape[0], dtype=np.int8)
    element_index = np.full(points.shape[0], -1, dtype=np.int32)
    for type_code, elements in (
        (3, yarn.triangle_indices),
        (4, yarn.quad_indices),
    ):
        unresolved = element_index < 0
        for index, connectivity in enumerate(elements):
            candidate = np.flatnonzero(unresolved)
            if candidate.size == 0:
                break
            inside = _points_inside_polygons(
                points[candidate], mesh_nodes[candidate][:, connectivity, :]
            )
            accepted = candidate[inside]
            element_type[accepted] = type_code
            element_index[accepted] = index
            unresolved[accepted] = False
    return element_type, element_index


def _material_orientations(yarn: NumpyExactYarnGeometry,
                           segments: np.ndarray,
                           u: np.ndarray,
                           locations: np.ndarray,
                           ups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    orientation = np.zeros((u.size, 3), dtype=np.float64)
    for segment in np.unique(segments):
        group = np.flatnonzero(segments == segment)
        group_u = u[group]
        _, tangent, _, _, _, angle = _evaluate_frame(yarn, int(segment), group_u)
        simple = yarn.constant_section & (angle == 0.0)
        orientation[group[simple]] = tangent[simple]
        complex_local = np.flatnonzero(~simple)
        if complex_local.size == 0:
            continue

        selected = group[complex_local]
        selected_u = group_u[complex_local]
        mesh_at_point = _section_mesh_nodes(yarn, int(segment), selected_u)
        element_type, element_index = _select_mesh_elements(
            locations[selected], mesh_at_point, yarn
        )
        for node_count, elements in (
            (3, yarn.triangle_indices),
            (4, yarn.quad_indices),
        ):
            local_members = np.flatnonzero(element_type == node_count)
            if local_members.size == 0:
                continue
            members = selected[local_members]
            member_u = u[members]
            # Preserve TexGen's historical upper-end condition exactly. It is
            # intentionally not written as min(u + 0.1, 1.0).
            u1 = np.where(member_u > 0.1, member_u - 0.1, 0.0)
            u2 = np.where(member_u < 0.99, member_u + 0.1, 1.0)
            pos1, _, up1, side1, _, _ = _evaluate_frame(
                yarn, int(segment), u1
            )
            pos2, _, up2, side2, _, _ = _evaluate_frame(
                yarn, int(segment), u2
            )
            mesh1 = _section_mesh_nodes(yarn, int(segment), u1)
            mesh2 = _section_mesh_nodes(yarn, int(segment), u2)
            connectivity = elements[element_index[local_members]]
            rows = np.arange(local_members.size)[:, None]
            local_nodes1 = mesh1[rows, connectivity]
            local_nodes2 = mesh2[rows, connectivity]
            global_nodes1 = (
                pos1[:, None, :]
                + side1[:, None, :] * local_nodes1[..., 0, None]
                + up1[:, None, :] * local_nodes1[..., 1, None]
            )
            global_nodes2 = (
                pos2[:, None, :]
                + side2[:, None, :] * local_nodes2[..., 0, None]
                + up2[:, None, :] * local_nodes2[..., 1, None]
            )
            orientation[members] = _normalise(
                np.mean(global_nodes2 - global_nodes1, axis=1)
            )

    secondary = _normalise(np.cross(orientation, ups))
    return orientation, secondary


def classify_numpy_exact(points: np.ndarray,
                         geometry: NumpyExactGeometry,
                         *,
                         tolerance: float = 1.0e-9,
                         include_orientations: bool = False) -> Any:
    """Classify arbitrary points using an extracted NumPy exact geometry.

    Parameters
    ----------
    points:
        World coordinates with shape ``(N, 3)``. Calculations always use
        ``float64`` to match TexGen's native precision.
    geometry:
        Snapshot returned by :func:`extract_numpy_exact_geometry`.
    tolerance:
        TexGen surface/AABB tolerance.
    include_orientations:
        Return the two material axes in addition to yarn ids.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    count = points.shape[0]
    yarn_ids = np.full(count, -1, dtype=np.int32)
    best_distance = np.ones(count, dtype=np.float64)
    chosen_segment = np.full(count, -1, dtype=np.int32)
    chosen_u = np.zeros(count, dtype=np.float64)
    chosen_location = np.zeros((count, 2), dtype=np.float64)
    chosen_up = np.zeros((count, 3), dtype=np.float64)

    for yarn_index, yarn in enumerate(geometry.yarns):
        (
            inside,
            distance,
            segments,
            u,
            locations,
            ups,
        ) = _inside_one_yarn(points, yarn, tolerance)
        update = inside & ((yarn_ids < 0) | (distance < best_distance))
        yarn_ids[update] = yarn_index
        best_distance[update] = distance[update]
        chosen_segment[update] = segments[update]
        chosen_u[update] = u[update]
        chosen_location[update] = locations[update]
        chosen_up[update] = ups[update]

    if not include_orientations:
        return yarn_ids

    orientation1 = np.zeros((count, 3), dtype=np.float64)
    orientation2 = np.zeros((count, 3), dtype=np.float64)
    for yarn_index, yarn in enumerate(geometry.yarns):
        selected = np.flatnonzero(yarn_ids == yarn_index)
        if selected.size == 0:
            continue
        first, second = _material_orientations(
            yarn,
            chosen_segment[selected],
            chosen_u[selected],
            chosen_location[selected],
            chosen_up[selected],
        )
        orientation1[selected] = first
        orientation2[selected] = second
    return yarn_ids, orientation1, orientation2


def _progress_iter(iterable: Iterable[Any],
                   progress: Any,
                   total: int) -> Iterable[Any]:
    if not progress:
        return iterable
    if callable(progress):
        return progress(
            iterable, total=total, desc="classify numpy_exact voxels", unit="chunk"
        )
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("progress=True requires tqdm") from exc
    return tqdm(
        iterable, total=total, desc="classify numpy_exact voxels", unit="chunk"
    )


def classify_numpy_exact_chunked(points: np.ndarray,
                                 geometry: NumpyExactGeometry,
                                 *,
                                 chunk_voxels: int = 8192,
                                 workers: Optional[int] = None,
                                 tolerance: float = 1.0e-9,
                                 include_orientations: bool = False,
                                 progress: Any = False) -> Any:
    """Chunked/threaded production wrapper around :func:`classify_numpy_exact`."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if int(chunk_voxels) < 1:
        raise ValueError("chunk_voxels must be >= 1")
    ranges = [
        (start, min(start + int(chunk_voxels), points.shape[0]))
        for start in range(0, points.shape[0], int(chunk_voxels))
    ]
    requested_workers = min(os.cpu_count() or 1, 4) if workers is None else int(workers)
    if requested_workers < 1:
        raise ValueError("workers must be >= 1 or None")
    actual_workers = min(requested_workers, max(1, len(ranges)))

    def classify_range(bounds: Tuple[int, int]) -> Tuple[Any, ...]:
        start, stop = bounds
        result = classify_numpy_exact(
            points[start:stop],
            geometry,
            tolerance=tolerance,
            include_orientations=include_orientations,
        )
        if include_orientations:
            ids, first, second = result
            return start, stop, ids, first, second
        return start, stop, result

    yarn_ids = np.full(points.shape[0], -1, dtype=np.int32)
    orientation1 = (
        np.zeros((points.shape[0], 3), dtype=np.float64)
        if include_orientations else None
    )
    orientation2 = (
        np.zeros((points.shape[0], 3), dtype=np.float64)
        if include_orientations else None
    )
    if actual_workers == 1:
        results = map(classify_range, ranges)
        iterator = _progress_iter(results, progress, len(ranges))
        for result in iterator:
            if include_orientations:
                start, stop, ids, first, second = result
                orientation1[start:stop] = first
                orientation2[start:stop] = second
            else:
                start, stop, ids = result
            yarn_ids[start:stop] = ids
    else:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            results = executor.map(classify_range, ranges)
            iterator = _progress_iter(results, progress, len(ranges))
            for result in iterator:
                if include_orientations:
                    start, stop, ids, first, second = result
                    orientation1[start:stop] = first
                    orientation2[start:stop] = second
                else:
                    start, stop, ids = result
                yarn_ids[start:stop] = ids
    if include_orientations:
        return yarn_ids, orientation1, orientation2, actual_workers
    return yarn_ids, actual_workers


__all__ = [
    "NumpyExactGeometry",
    "NumpyExactYarnGeometry",
    "classify_numpy_exact",
    "classify_numpy_exact_chunked",
    "extract_numpy_exact_geometry",
]
