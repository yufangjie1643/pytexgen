"""Sparse orientation and constitutive-field utilities.

The public stiffness convention is engineering Voigt in the component order
``(xx, yy, zz, yz, xz, xy)``.  Compact C21 arrays store the row-major upper
triangle of the symmetric ``6 x 6`` matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

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


def _copy_array(value: Any):
    if _is_torch_tensor(value):
        return value.clone()
    return np.array(value, copy=True)


def _numpy_float_dtype(dtype: Any):
    if dtype is None:
        return None
    if torch is not None and isinstance(dtype, torch.dtype):
        mapping = {
            torch.float32: np.dtype(np.float32),
            torch.float64: np.dtype(np.float64),
        }
        if dtype not in mapping:
            raise ValueError("dtype must be a floating-point dtype")
        return mapping[dtype]
    result = np.dtype(dtype)
    if not np.issubdtype(result, np.floating):
        raise ValueError("dtype must be a floating-point dtype")
    return result


def _torch_float_dtype(dtype: Any, fallback):
    if torch is None:
        raise ImportError(
            'Torch storage requested; install with `pip install "pytexgen[gpu]"`.'
        )
    if dtype is None:
        return fallback
    if isinstance(dtype, torch.dtype):
        result = dtype
    else:
        result = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
        }.get(np.dtype(dtype))
    if result not in {torch.float32, torch.float64}:
        raise ValueError("dtype must be a floating-point dtype")
    return result


def _convert_float_array(
    value: Any,
    storage: str,
    *,
    device: Optional[str],
    dtype: Any,
    copy: bool,
):
    if storage == "numpy":
        target_dtype = _numpy_float_dtype(dtype)
        if _is_torch_tensor(value):
            result = value.detach().cpu().numpy()
            if copy:
                result = result.copy()
        else:
            result = np.array(value, copy=copy)
        if target_dtype is not None:
            result = result.astype(
                target_dtype, copy=copy or result.dtype != target_dtype
            )
        return result

    fallback = value.dtype if _is_torch_tensor(value) else torch.float64
    target_dtype = _torch_float_dtype(dtype, fallback)
    if _is_torch_tensor(value):
        result = value.to(device=device, dtype=target_dtype)
        return result.clone() if copy else result
    result = torch.as_tensor(value, dtype=target_dtype, device=device)
    return result.clone() if copy else result


def _convert_integer_array(
    value: Any,
    storage: str,
    *,
    device: Optional[str],
    copy: bool,
):
    if storage == "numpy":
        if _is_torch_tensor(value):
            result = value.detach().cpu().numpy()
            return result.copy() if copy else result
        return np.array(value, copy=copy)
    if torch is None:
        raise ImportError(
            'Torch storage requested; install with `pip install "pytexgen[gpu]"`.'
        )
    if _is_torch_tensor(value):
        result = value.to(device=device)
        return result.clone() if copy else result
    result = torch.as_tensor(value, device=device)
    return result.clone() if copy else result


def _array_length(value: Any) -> int:
    return int(value.shape[0])


def _is_integer_array(value: Any) -> bool:
    if _is_torch_tensor(value):
        return value.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    return np.issubdtype(np.asarray(value).dtype, np.integer)


def _same_array_backend(values: Sequence[Any]) -> bool:
    return len({_is_torch_tensor(value) for value in values}) <= 1


def _same_torch_device(values: Sequence[Any]) -> bool:
    devices = {
        value.device for value in values if _is_torch_tensor(value)
    }
    return len(devices) <= 1


def _any_true(value: Any) -> bool:
    if _is_torch_tensor(value):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


def _validate_grid_shape(grid_shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must contain (Nz, Ny, Nx), got {grid_shape}")
    result = tuple(int(value) for value in grid_shape)
    if any(value <= 0 for value in result):
        raise ValueError("grid_shape dimensions must be positive")
    return result


def _validate_sparse_common(
    voxel_indices: Any,
    yarn_ids: Any,
    float_arrays: Sequence[Any],
    grid_shape: Sequence[int],
    *,
    other_ids: Sequence[Any] = (),
) -> Tuple[int, int, int]:
    arrays = (voxel_indices, yarn_ids, *other_ids, *float_arrays)
    if not all(hasattr(value, "shape") for value in arrays):
        raise ValueError("sparse field values must be arrays or tensors")
    if not _same_array_backend(arrays):
        raise ValueError("all sparse field arrays must use the same storage backend")
    if not _same_torch_device(arrays):
        raise ValueError("all sparse field tensors must use the same device")
    if voxel_indices.ndim != 1 or yarn_ids.ndim != 1:
        raise ValueError("voxel_indices and yarn_ids must be one-dimensional")
    if not _is_integer_array(voxel_indices) or not _is_integer_array(yarn_ids):
        raise ValueError("voxel_indices and yarn_ids must use integer dtypes")

    count = _array_length(voxel_indices)
    lengths = [_array_length(yarn_ids)]
    lengths.extend(_array_length(value) for value in other_ids)
    lengths.extend(_array_length(value) for value in float_arrays)
    if any(length != count for length in lengths):
        raise ValueError("all sparse field arrays must have the same leading length")
    for value in other_ids:
        if value.ndim != 1 or not _is_integer_array(value):
            raise ValueError("material id arrays must be one-dimensional integers")
    if any(value.shape != (count, 3) for value in float_arrays):
        raise ValueError("orientation arrays must have shape (N, 3)")
    if any(not _all_finite(value) for value in float_arrays):
        raise ValueError("orientation arrays must contain only finite values")

    shape = _validate_grid_shape(grid_shape)
    total = math.prod(shape)
    if count:
        if _any_true(voxel_indices[1:] <= voxel_indices[:-1]):
            raise ValueError("voxel_indices must be strictly increasing")
        if _any_true((voxel_indices < 0) | (voxel_indices >= total)):
            raise ValueError("voxel_indices contain values out of range")
        if _any_true(yarn_ids < 0):
            raise ValueError("yarn_ids must be non-negative")
    return shape


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


@dataclass(frozen=True)
class SparseOrientationField:
    """Orientation frames stored only for yarn voxels."""

    voxel_indices: Any
    yarn_ids: Any
    orientation1: Any
    orientation2: Any
    grid_shape: Tuple[int, int, int]
    order: str = "ix + iy*nx + iz*nx*ny"

    def __post_init__(self):
        shape = _validate_sparse_common(
            self.voxel_indices,
            self.yarn_ids,
            (self.orientation1, self.orientation2),
            self.grid_shape,
        )
        object.__setattr__(self, "grid_shape", shape)

    @property
    def num_yarn_voxels(self) -> int:
        return _array_length(self.voxel_indices)

    @property
    def storage(self) -> str:
        return "torch" if _is_torch_tensor(self.voxel_indices) else "numpy"

    @property
    def device(self) -> str:
        if _is_torch_tensor(self.voxel_indices):
            return str(self.voxel_indices.device)
        return "cpu"

    def to(
        self,
        storage: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        copy: bool = False,
    ) -> "SparseOrientationField":
        """Convert all arrays while preserving integer identifier dtypes."""
        target = self.storage if storage is None else str(storage).lower()
        if target not in {"numpy", "torch"}:
            raise ValueError('storage must be "numpy", "torch", or None')
        return SparseOrientationField(
            voxel_indices=_convert_integer_array(
                self.voxel_indices, target, device=device, copy=copy
            ),
            yarn_ids=_convert_integer_array(
                self.yarn_ids, target, device=device, copy=copy
            ),
            orientation1=_convert_float_array(
                self.orientation1,
                target,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            orientation2=_convert_float_array(
                self.orientation2,
                target,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            grid_shape=self.grid_shape,
            order=self.order,
        )


@dataclass(frozen=True)
class SparseStiffnessField:
    """One matrix C21 value plus rotated C21 values for yarn voxels."""

    matrix_c21: Any
    voxel_indices: Any
    yarn_ids: Any
    material_ids: Any
    yarn_c21: Any
    grid_shape: Tuple[int, int, int]
    unit: Optional[str] = None
    order: str = "ix + iy*nx + iz*nx*ny"

    def __post_init__(self):
        arrays = (
            self.matrix_c21,
            self.voxel_indices,
            self.yarn_ids,
            self.material_ids,
            self.yarn_c21,
        )
        if not _same_array_backend(arrays):
            raise ValueError("all sparse field arrays must use the same storage backend")
        if not _same_torch_device(arrays):
            raise ValueError("all sparse field tensors must use the same device")
        if self.matrix_c21.shape != (21,):
            raise ValueError("matrix_c21 must have shape (21,)")
        if self.yarn_c21.ndim != 2 or self.yarn_c21.shape[1] != 21:
            raise ValueError("yarn_c21 must have shape (N, 21)")
        if self.material_ids.ndim != 1 or not _is_integer_array(self.material_ids):
            raise ValueError("material_ids must be one-dimensional integers")
        shape = _validate_sparse_common(
            self.voxel_indices,
            self.yarn_ids,
            (),
            self.grid_shape,
            other_ids=(self.material_ids,),
        )
        count = _array_length(self.voxel_indices)
        if _array_length(self.yarn_c21) != count:
            raise ValueError("all sparse field arrays must have the same leading length")
        if not _all_finite(self.matrix_c21) or not _all_finite(self.yarn_c21):
            raise ValueError("stiffness coefficients must contain only finite values")
        if count and _any_true(self.material_ids <= 0):
            raise ValueError("yarn material_ids must be positive")
        object.__setattr__(self, "grid_shape", shape)

    @property
    def num_yarn_voxels(self) -> int:
        return _array_length(self.voxel_indices)

    @property
    def storage(self) -> str:
        return "torch" if _is_torch_tensor(self.voxel_indices) else "numpy"

    @property
    def device(self) -> str:
        if _is_torch_tensor(self.voxel_indices):
            return str(self.voxel_indices.device)
        return "cpu"

    def to(
        self,
        storage: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        copy: bool = False,
    ) -> "SparseStiffnessField":
        """Convert all arrays while preserving integer identifier dtypes."""
        target = self.storage if storage is None else str(storage).lower()
        if target not in {"numpy", "torch"}:
            raise ValueError('storage must be "numpy", "torch", or None')
        return SparseStiffnessField(
            matrix_c21=_convert_float_array(
                self.matrix_c21,
                target,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            voxel_indices=_convert_integer_array(
                self.voxel_indices, target, device=device, copy=copy
            ),
            yarn_ids=_convert_integer_array(
                self.yarn_ids, target, device=device, copy=copy
            ),
            material_ids=_convert_integer_array(
                self.material_ids, target, device=device, copy=copy
            ),
            yarn_c21=_convert_float_array(
                self.yarn_c21,
                target,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            grid_shape=self.grid_shape,
            unit=self.unit,
            order=self.order,
        )

    def to_dense_c21(self):
        """Materialize ``(Nz, Ny, Nx, 21)`` on the current backend/device."""
        total = math.prod(self.grid_shape)
        if _is_torch_tensor(self.matrix_c21):
            flat = self.matrix_c21.expand(total, 21).clone()
            flat[self.voxel_indices.to(dtype=torch.long)] = self.yarn_c21
        else:
            flat = np.broadcast_to(
                np.asarray(self.matrix_c21), (total, 21)
            ).copy()
            flat[np.asarray(self.voxel_indices, dtype=np.int64)] = self.yarn_c21
        return flat.reshape(self.grid_shape + (21,))

    def to_dense_voigt(self):
        """Materialize ``(6, 6, Nz, Ny, Nx)`` engineering-Voigt stiffness."""
        field = unpack_c21(self.to_dense_c21())
        if _is_torch_tensor(field):
            return field.permute(3, 4, 0, 1, 2).contiguous()
        return np.moveaxis(field, (-2, -1), (0, 1))

    def to_acdm(self, *, batch: bool = True):
        """Return Voxel-ACDM layout, optionally with a leading batch axis."""
        field = self.to_dense_voigt()
        if not batch:
            return field
        if _is_torch_tensor(field):
            return field.unsqueeze(0)
        return np.expand_dims(field, axis=0)


__all__ = [
    "VOIGT_COMPONENTS",
    "C21_INDICES",
    "SparseOrientationField",
    "SparseStiffnessField",
    "pack_voigt_c21",
    "unpack_c21",
    "isotropic_stiffness_c21",
    "orthotropic_stiffness_c21",
]
