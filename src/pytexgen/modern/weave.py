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

        aabb = np.array(
            [
                [0.0, 0.0, 0.0],
                [self.width * self.spacing, self.height * self.spacing, self.thickness],
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
