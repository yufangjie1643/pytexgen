"""Lightweight geometry containers for modern pytexgen backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Section:
    """Planar yarn cross-section polygon in local side/up coordinates."""

    points: np.ndarray

    @classmethod
    def ellipse(cls, width: float, height: float, samples: int = 32) -> "Section":
        """Create a closed polygon approximating an ellipse."""
        if width <= 0 or height <= 0:
            raise ValueError("section width and height must be positive")
        if samples < 4:
            raise ValueError("ellipse samples must be >= 4")
        angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
        points = np.column_stack(
            [
                0.5 * float(width) * np.cos(angles),
                0.5 * float(height) * np.sin(angles),
            ]
        )
        points = np.vstack([points, points[:1]])
        return cls(points.astype(np.float64, copy=False))

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("section points must have shape (N, 2)")
        if self.points.shape[0] < 4:
            raise ValueError("section polygon needs at least four points")


@dataclass(frozen=True)
class YarnPath:
    """One yarn centerline and its constant section frame."""

    positions: np.ndarray
    section: Section
    up: np.ndarray
    side: np.ndarray
    translations: np.ndarray

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.positions.shape[0] < 2:
            raise ValueError("a yarn path needs at least two positions")
        if self.up.shape != (3,):
            raise ValueError("up must have shape (3,)")
        if self.side.shape != (3,):
            raise ValueError("side must have shape (3,)")
        if self.translations.ndim != 2 or self.translations.shape[1] != 3:
            raise ValueError("translations must have shape (N, 3)")


@dataclass(frozen=True)
class ModernTextileModel:
    """A bundle of yarn paths plus the structured voxelization domain."""

    name: str
    yarns: tuple[YarnPath, ...]
    aabb: np.ndarray

    def __post_init__(self) -> None:
        if len(self.yarns) == 0:
            raise ValueError("model must contain at least one yarn")
        if self.aabb.shape != (2, 3):
            raise ValueError("aabb must have shape (2, 3)")
        if np.any(self.aabb[1] <= self.aabb[0]):
            raise ValueError("aabb max corner must be greater than min corner")
