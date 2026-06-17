"""Modern woven textile model builders."""

from __future__ import annotations

import numpy as np

from .geometry import ModernTextileModel, Section, YarnPath


class PlainWeave2D:
    """Small Python equivalent for the first CTextileWeave2D migration slice."""

    def __init__(
        self,
        width: int,
        height: int,
        spacing: float,
        thickness: float,
        yarn_width: float = 0.8,
        yarn_height: float | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.spacing = float(spacing)
        self.thickness = float(thickness)
        self.yarn_width = float(yarn_width)
        self.yarn_height = float(yarn_height if yarn_height is not None else thickness / 2.0)
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be >= 1")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if self.yarn_width <= 0 or self.yarn_height <= 0:
            raise ValueError("yarn width and height must be positive")
        self._pattern = [
            [["y", "x"] for _ in range(self.height)]
            for _ in range(self.width)
        ]

    def cell(self, x: int, y: int) -> tuple[str, ...]:
        """Return the yarn order at cell ``(x, y)``.

        The storage follows TexGen's ``CTextileWeave::GetCell(x, y)`` order:
        x indexes y-direction yarn columns and y indexes x-direction yarn rows.
        """
        self._validate_cell_indices(x, y)
        return tuple(self._pattern[int(x)][int(y)])

    def swap_position(self, x: int, y: int) -> None:
        """Swap the over/under order at one weave cell."""
        self._validate_cell_indices(x, y)
        self._pattern[int(x)][int(y)] = list(reversed(self._pattern[int(x)][int(y)]))

    def to_model(self) -> ModernTextileModel:
        """Build a snapshot-compatible model with TexGen-like yarn ordering."""
        section = Section.ellipse(self.yarn_width, self.yarn_height)
        yarns: list[YarnPath] = []
        x_nodes = np.linspace(0.0, self.width * self.spacing, self.width + 1)
        y_nodes = np.linspace(0.0, self.height * self.spacing, self.height + 1)

        for row in range(self.height):
            y = row * self.spacing
            z_nodes = np.array(
                [self._cell_z(column % self.width, row, "x") for column in range(self.width + 1)],
                dtype=np.float64,
            )
            positions = np.column_stack(
                [x_nodes, np.full_like(x_nodes, y), z_nodes]
            )
            yarns.append(_yarn(positions, section, direction="x"))

        for column in range(self.width):
            x = column * self.spacing
            z_nodes = np.array(
                [self._cell_z(column, row % self.height, "y") for row in range(self.height + 1)],
                dtype=np.float64,
            )
            positions = np.column_stack(
                [np.full_like(y_nodes, x), y_nodes, z_nodes]
            )
            yarns.append(_yarn(positions, section, direction="y"))

        z_margin = 0.1 * self.yarn_height
        aabb = np.array(
            [
                [-0.5 * self.spacing, -0.5 * self.spacing, -z_margin],
                [
                    (self.width - 0.5) * self.spacing,
                    (self.height - 0.5) * self.spacing,
                    self.thickness + z_margin,
                ],
            ],
            dtype=np.float64,
        )
        return ModernTextileModel(name="PlainWeave2D", yarns=tuple(yarns), aabb=aabb)

    def _cell_z(self, x: int, y: int, yarn: str) -> float:
        order = self.cell(x, y)
        layer = order.index(yarn)
        return self.thickness * (layer + 0.5) / len(order)

    def _validate_cell_indices(self, x: int, y: int) -> None:
        if int(x) < 0 or int(x) >= self.width or int(y) < 0 or int(y) >= self.height:
            raise IndexError(f"cell index out of range: {x}, {y}")


def _yarn(positions: np.ndarray, section: Section, direction: str) -> YarnPath:
    if direction == "x":
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    elif direction == "y":
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError('direction must be "x" or "y"')
    return YarnPath(
        positions=positions.astype(np.float64, copy=False),
        section=section,
        up=up,
        side=side,
        translations=np.zeros((1, 3), dtype=np.float64),
    )


def auto_binder_positions(
    weave_type: str,
    num_x_yarns: int,
    num_y_yarns: int,
    z_layers: int,
    binder_depth: int | None = None,
    straight_binder_depth: int | None = None,
) -> list[tuple[int, int, int]]:
    """Generate binder offsets using the current shallow-cross script rules."""
    kind = _normalise_weave_type(weave_type)
    num_x_yarns = int(num_x_yarns)
    num_y_yarns = int(num_y_yarns)
    z_layers = int(z_layers)
    if z_layers < 1:
        raise ValueError("z_layers must be >= 1")
    if num_x_yarns < 1 or num_y_yarns < 1:
        raise ValueError("yarn counts must be >= 1")
    if binder_depth is None:
        binder_depth = min(3, z_layers)
    binder_depth = int(binder_depth)
    if binder_depth < 1 or binder_depth > z_layers:
        raise ValueError("binder_depth must be between 1 and z_layers")

    if kind == "bent":
        max_offset = binder_depth - 1
        return [
            (x_index, y_index, 0 if (x_index + y_index) % 2 == 0 else max_offset)
            for y_index in range(num_x_yarns)
            for x_index in range(num_y_yarns)
        ]

    if straight_binder_depth is None:
        straight_binder_depth = binder_depth
    peak = max(0, int(straight_binder_depth) - 1)
    if peak >= z_layers:
        raise ValueError("straight_binder_depth must be <= z_layers")

    positions = []
    period = max(1, 2 * peak)
    for y_index in range(num_x_yarns):
        for x_index in range(num_y_yarns):
            if peak == 0:
                offset = 0
            else:
                phase = x_index % period
                offset = phase if phase <= peak else period - phase
                if y_index % 2:
                    offset = peak - offset
            positions.append((x_index, y_index, offset))
    return positions


class ShallowCrossLayerToLayer:
    """Modern subset for the SiC/SiC shallow-cross layer-to-layer model."""

    def __init__(
        self,
        *,
        num_x_yarns: int,
        num_y_yarns: int,
        x_spacing: float,
        y_spacing: float,
        z_layers: int,
        binder_depth: int | None = None,
        weave_type: str = "straight",
        x_height: float = 0.3,
        y_height: float = 0.3,
        warp_yarn_width: float = 1.2,
        weft_yarn_width: float = 1.5,
        binder_yarn_width: float = 0.6,
        binder_yarn_height: float | None = None,
        binder_positions: list[tuple[int, int, int]] | None = None,
    ):
        self.num_x_yarns = int(num_x_yarns)
        self.num_y_yarns = int(num_y_yarns)
        self.x_spacing = float(x_spacing)
        self.y_spacing = float(y_spacing)
        self.z_layers = int(z_layers)
        self.binder_depth = None if binder_depth is None else int(binder_depth)
        self.weave_type = _normalise_weave_type(weave_type)
        self.x_height = float(x_height)
        self.y_height = float(y_height)
        self.warp_yarn_width = float(warp_yarn_width)
        self.weft_yarn_width = float(weft_yarn_width)
        self.binder_yarn_width = float(binder_yarn_width)
        self.binder_yarn_height = float(
            binder_yarn_height if binder_yarn_height is not None else x_height
        )
        self.binder_positions = binder_positions
        self._validate()

    def to_model(self) -> ModernTextileModel:
        """Build a snapshot-compatible shallow-cross approximation."""
        x_length = self.num_y_yarns * self.y_spacing
        y_length = self.num_x_yarns * self.x_spacing
        layer_pitch = max(self.x_height, self.y_height, self.binder_yarn_height)
        z_extent = self.z_layers * layer_pitch

        warp_section = Section.ellipse(self.warp_yarn_width, self.x_height)
        weft_section = Section.ellipse(self.weft_yarn_width, self.y_height)
        binder_section = Section.ellipse(self.binder_yarn_width, self.binder_yarn_height)
        yarns: list[YarnPath] = []

        x_nodes = np.linspace(0.0, x_length, self.num_y_yarns + 1)
        y_nodes = np.linspace(0.0, y_length, self.num_x_yarns + 1)

        for layer in range(self.z_layers):
            z = (layer + 0.5) * layer_pitch
            for y_index in range(self.num_x_yarns):
                y = y_index * self.x_spacing
                positions = np.column_stack(
                    [x_nodes, np.full_like(x_nodes, y), np.full_like(x_nodes, z)]
                )
                yarns.append(_yarn(positions, warp_section, direction="x"))

        for layer in range(self.z_layers + 1):
            z = min(layer * layer_pitch, z_extent)
            for x_index in range(self.num_y_yarns):
                x = x_index * self.y_spacing
                positions = np.column_stack(
                    [np.full_like(y_nodes, x), y_nodes, np.full_like(y_nodes, z)]
                )
                yarns.append(_yarn(positions, weft_section, direction="y"))

        positions = self.binder_positions
        if positions is None:
            positions = auto_binder_positions(
                self.weave_type,
                self.num_x_yarns,
                self.num_y_yarns,
                self.z_layers,
                self.binder_depth,
            )
        by_row: dict[int, list[tuple[int, int]]] = {}
        for x_index, y_index, offset in positions:
            by_row.setdefault(int(y_index), []).append((int(x_index), int(offset)))
        for y_index, row_positions in sorted(by_row.items()):
            row_positions.sort()
            xs = np.asarray([x * self.y_spacing for x, _offset in row_positions], dtype=np.float64)
            y = y_index * self.x_spacing
            zs = np.asarray(
                [
                    min((offset + 0.5) * layer_pitch, z_extent)
                    for _x, offset in row_positions
                ],
                dtype=np.float64,
            )
            binder_path = np.column_stack([xs, np.full_like(xs, y), zs])
            yarns.append(_yarn(binder_path, binder_section, direction="x"))

        aabb = np.array([[0.0, 0.0, 0.0], [x_length, y_length, z_extent]], dtype=np.float64)
        return ModernTextileModel(
            name="ShallowCrossLayerToLayer",
            yarns=tuple(yarns),
            aabb=aabb,
        )

    def _validate(self) -> None:
        if self.num_x_yarns < 1 or self.num_y_yarns < 1:
            raise ValueError("num_x_yarns and num_y_yarns must be >= 1")
        if self.x_spacing <= 0 or self.y_spacing <= 0:
            raise ValueError("x_spacing and y_spacing must be positive")
        if self.z_layers < 1:
            raise ValueError("z_layers must be >= 1")
        for name, value in (
            ("x_height", self.x_height),
            ("y_height", self.y_height),
            ("warp_yarn_width", self.warp_yarn_width),
            ("weft_yarn_width", self.weft_yarn_width),
            ("binder_yarn_width", self.binder_yarn_width),
            ("binder_yarn_height", self.binder_yarn_height),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


def _normalise_weave_type(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "bent": "bent",
        "bend": "bent",
        "curved": "bent",
        "shallow_cross_binder": "bent",
        "straight": "straight",
        "direct": "straight",
        "shallow_cross_straight": "straight",
        "shallow_cross_straight_binder": "straight",
    }
    if key not in aliases:
        raise ValueError("weave_type must be 'bent' or 'straight'")
    return aliases[key]
