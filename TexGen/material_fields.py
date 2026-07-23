"""Sparse orientation and constitutive-field utilities.

The public stiffness convention is engineering Voigt in the component order
``(xx, yy, zz, yz, xz, xy)``.  Compact C21 arrays store the row-major upper
triangle of the symmetric ``6 x 6`` matrix.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None


VOIGT_COMPONENTS: Tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
C21_INDICES: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in range(6) for j in range(i, 6)
)


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _all_finite(value: Any) -> bool:
    if _is_torch_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(np.asarray(value)).all())


def _allclose(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if _is_torch_tensor(left):
        return bool(torch.allclose(left, right, rtol=rtol, atol=atol))
    return bool(np.allclose(left, right, rtol=rtol, atol=atol))


def _stack(values: Sequence[Any], axis: int):
    if values and _is_torch_tensor(values[0]):
        return torch.stack(tuple(values), dim=axis)
    return np.stack(tuple(values), axis=axis)


def _zeros(shape: Tuple[int, ...], *, like: Any):
    if _is_torch_tensor(like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    array = np.asarray(like)
    return np.zeros(shape, dtype=array.dtype)


def pack_voigt_c21(
    matrix: Any,
    *,
    symmetry_rtol: float = 1e-10,
    symmetry_atol: float = 1e-12,
):
    """Pack symmetric engineering-Voigt matrices into 21 coefficients."""
    if not hasattr(matrix, "shape") or tuple(matrix.shape[-2:]) != (6, 6):
        shape = getattr(matrix, "shape", None)
        raise ValueError(f"matrix must end with shape (6, 6), got {shape}")
    if not _all_finite(matrix):
        raise ValueError("matrix must contain only finite values")
    transpose = (
        matrix.transpose(-1, -2)
        if _is_torch_tensor(matrix)
        else np.swapaxes(np.asarray(matrix), -1, -2)
    )
    if not _allclose(
        matrix, transpose, rtol=symmetry_rtol, atol=symmetry_atol
    ):
        raise ValueError("matrix must be symmetric")
    return _stack([matrix[..., i, j] for i, j in C21_INDICES], axis=-1)


def unpack_c21(c21: Any):
    """Expand compact C21 coefficients into symmetric Voigt matrices."""
    if not hasattr(c21, "shape") or len(c21.shape) == 0 or c21.shape[-1] != 21:
        shape = getattr(c21, "shape", None)
        raise ValueError(f"c21 must end with length 21, got {shape}")
    if not _all_finite(c21):
        raise ValueError("c21 must contain only finite values")
    result = _zeros(tuple(c21.shape[:-1]) + (6, 6), like=c21)
    for index, (i, j) in enumerate(C21_INDICES):
        result[..., i, j] = c21[..., index]
        result[..., j, i] = c21[..., index]
    return result


def _finite_scalars(values: Sequence[float], names: Sequence[str]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{', '.join(names)} must contain only finite values")
    return result


def isotropic_stiffness_c21(E: float, nu: float) -> np.ndarray:
    """Return isotropic stiffness as engineering-Voigt C21 coefficients."""
    E_value, nu_value = _finite_scalars((E, nu), ("E", "nu"))
    if E_value <= 0.0:
        raise ValueError("Young's modulus E must be positive")
    if not -1.0 < nu_value < 0.5:
        raise ValueError("Poisson ratio nu must satisfy -1 < nu < 0.5")

    lam = E_value * nu_value / (
        (1.0 + nu_value) * (1.0 - 2.0 * nu_value)
    )
    mu = E_value / (2.0 * (1.0 + nu_value))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[0, 0] += 2.0 * mu
    matrix[1, 1] += 2.0 * mu
    matrix[2, 2] += 2.0 * mu
    matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = mu
    return pack_voigt_c21(matrix)


def orthotropic_stiffness_c21(
    E1: float,
    E2: float,
    E3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    G12: float,
    G13: float,
    G23: float,
) -> np.ndarray:
    """Return orthotropic stiffness from nine engineering constants."""
    values = _finite_scalars(
        (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23),
        ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23"),
    )
    E1_value, E2_value, E3_value = values[:3]
    nu12_value, nu13_value, nu23_value = values[3:6]
    G12_value, G13_value, G23_value = values[6:]
    if np.any(
        np.asarray(
            (
                E1_value,
                E2_value,
                E3_value,
                G12_value,
                G13_value,
                G23_value,
            )
        )
        <= 0.0
    ):
        raise ValueError("Young's and shear moduli must be positive")

    compliance = np.zeros((6, 6), dtype=np.float64)
    compliance[0, 0] = 1.0 / E1_value
    compliance[1, 1] = 1.0 / E2_value
    compliance[2, 2] = 1.0 / E3_value
    compliance[0, 1] = compliance[1, 0] = -nu12_value / E1_value
    compliance[0, 2] = compliance[2, 0] = -nu13_value / E1_value
    compliance[1, 2] = compliance[2, 1] = -nu23_value / E2_value
    compliance[3, 3] = 1.0 / G23_value
    compliance[4, 4] = 1.0 / G13_value
    compliance[5, 5] = 1.0 / G12_value
    stiffness = np.linalg.inv(compliance)
    return pack_voigt_c21(stiffness)


__all__ = [
    "VOIGT_COMPONENTS",
    "C21_INDICES",
    "pack_voigt_c21",
    "unpack_c21",
    "isotropic_stiffness_c21",
    "orthotropic_stiffness_c21",
]
