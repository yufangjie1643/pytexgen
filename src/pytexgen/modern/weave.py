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

    def to_model(self) -> ModernTextileModel:
        """Build a snapshot-compatible model with TexGen-like yarn ordering."""
        section = Section.ellipse(self.yarn_width, self.yarn_height)
        yarns: list[YarnPath] = []
        x_nodes = np.linspace(0.0, self.width * self.spacing, self.width + 1)
        y_nodes = np.linspace(0.0, self.height * self.spacing, self.height + 1)
        lower_z = self.thickness * 0.25
        upper_z = self.thickness * 0.75

        for row in range(self.height):
            y = row * self.spacing
            positions = np.column_stack(
                [x_nodes, np.full_like(x_nodes, y), np.full_like(x_nodes, upper_z)]
            )
            yarns.append(_yarn(positions, section, direction="x"))

        for column in range(self.width):
            x = column * self.spacing
            positions = np.column_stack(
                [np.full_like(y_nodes, x), y_nodes, np.full_like(y_nodes, lower_z)]
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
