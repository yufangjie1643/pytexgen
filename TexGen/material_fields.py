"""Sparse orientation and constitutive-field utilities.

The public stiffness convention is engineering Voigt in the component order
``(xx, yy, zz, yz, xz, xy)``.  Compact C21 arrays store the row-major upper
triangle of the symmetric ``6 x 6`` matrix.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
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

    if torch is None:
        _torch_float_dtype(dtype, None)
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


def _norm(value: Any, *, axis: int, keepdims: bool):
    if _is_torch_tensor(value):
        return torch.linalg.vector_norm(value, dim=axis, keepdim=keepdims)
    return np.linalg.norm(value, axis=axis, keepdims=keepdims)


def _sum(value: Any, *, axis: int, keepdims: bool):
    if _is_torch_tensor(value):
        return value.sum(dim=axis, keepdim=keepdims)
    return np.sum(value, axis=axis, keepdims=keepdims)


def _cross(left: Any, right: Any):
    if _is_torch_tensor(left):
        return torch.cross(left, right, dim=-1)
    return np.cross(left, right)


def _einsum(expression: str, *values: Any):
    if values and _is_torch_tensor(values[0]):
        return torch.einsum(expression, *values)
    return np.einsum(expression, *values)


def _concatenate(values: Sequence[Any], *, axis: int):
    if values and _is_torch_tensor(values[0]):
        return torch.cat(tuple(values), dim=axis)
    return np.concatenate(tuple(values), axis=axis)


def _mandel_basis(*, like: Any):
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    basis = _zeros((6, 3, 3), like=like)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0
    basis[3, 1, 2] = basis[3, 2, 1] = inverse_sqrt_two
    basis[4, 0, 2] = basis[4, 2, 0] = inverse_sqrt_two
    basis[5, 0, 1] = basis[5, 1, 0] = inverse_sqrt_two
    return basis


def _mandel_weights(*, like: Any):
    values = (1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0))
    if _is_torch_tensor(like):
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)
    return np.asarray(values, dtype=np.asarray(like).dtype)


def _orthonormal_frames(
    orientation1: Any,
    orientation2: Any,
    *,
    eps: float,
):
    norm1 = _norm(orientation1, axis=-1, keepdims=True)
    if _any_true(norm1 <= eps):
        raise ValueError("orientation vectors are zero or collinear")
    e1 = orientation1 / norm1
    projected = orientation2 - (
        _sum(orientation2 * e1, axis=-1, keepdims=True) * e1
    )
    norm2 = _norm(projected, axis=-1, keepdims=True)
    if _any_true(norm2 <= eps):
        raise ValueError("orientation vectors are zero or collinear")
    e2 = projected / norm2
    e3 = _cross(e1, e2)
    return _stack((e1, e2, e3), axis=-1)


def _rotate_stiffness_chunk(
    local_c21: Any,
    orientation1: Any,
    orientation2: Any,
    *,
    eps: float,
):
    rotation = _orthonormal_frames(orientation1, orientation2, eps=eps)
    basis = _mandel_basis(like=local_c21)
    rotated_basis = _einsum(
        "niI,aIJ,nkJ->naik", rotation, basis, rotation
    )
    mandel_rotation = _einsum("bik,naik->nba", basis, rotated_basis)

    weights = _mandel_weights(like=local_c21)
    inverse_weights = 1.0 / weights
    local_voigt = unpack_c21(local_c21)
    local_mandel = (
        local_voigt
        * weights[None, :, None]
        * weights[None, None, :]
    )
    global_mandel = _einsum(
        "nij,njk,nlk->nil",
        mandel_rotation,
        local_mandel,
        mandel_rotation,
    )
    global_voigt = (
        global_mandel
        * inverse_weights[None, :, None]
        * inverse_weights[None, None, :]
    )
    return pack_voigt_c21(global_voigt)


def rotate_stiffness_c21(
    local_c21: Any,
    orientation1: Any,
    orientation2: Any,
    *,
    chunk_voxels: int = 65536,
    eps: float = 1e-12,
):
    """Rotate per-voxel local C21 stiffness into global coordinates."""
    arrays = (local_c21, orientation1, orientation2)
    if not all(hasattr(value, "shape") for value in arrays):
        raise ValueError("stiffness and orientation inputs must be arrays")
    if not _same_array_backend(arrays) or not _same_torch_device(arrays):
        raise ValueError("stiffness and orientations must share backend and device")
    if local_c21.ndim != 2 or local_c21.shape[1] != 21:
        raise ValueError("local_c21 must have shape (N, 21)")
    count = int(local_c21.shape[0])
    if orientation1.shape != (count, 3) or orientation2.shape != (count, 3):
        raise ValueError("orientation arrays must have shape (N, 3)")
    if not all(_all_finite(value) for value in arrays):
        raise ValueError("stiffness and orientations must contain only finite values")
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be >= 1")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if count == 0:
        return _copy_array(local_c21)

    chunks = []
    for start in range(0, count, chunk_voxels):
        stop = min(start + chunk_voxels, count)
        chunks.append(
            _rotate_stiffness_chunk(
                local_c21[start:stop],
                orientation1[start:stop],
                orientation2[start:stop],
                eps=eps,
            )
        )
    return _concatenate(chunks, axis=0)


def _coerce_c21_for_field(value: Any, orientation: SparseOrientationField):
    shape = getattr(value, "shape", None)
    if shape is None:
        value = np.asarray(value)
        shape = value.shape
    if tuple(shape) == (6, 6):
        value = pack_voigt_c21(value)
    elif tuple(shape) != (21,):
        raise ValueError(
            f"stiffness must have shape (21,) or (6, 6), got {shape}"
        )
    return _convert_float_array(
        value,
        orientation.storage,
        device=orientation.device,
        dtype=orientation.orientation1.dtype,
        copy=False,
    )


def _validate_positive_definite(c21: Any, name: str) -> None:
    matrix = unpack_c21(c21)
    if _is_torch_tensor(matrix):
        minimum = torch.linalg.eigvalsh(matrix).amin()
        valid = bool((minimum > 0.0).item())
    else:
        valid = bool(np.linalg.eigvalsh(matrix).min() > 0.0)
    if not valid:
        raise ValueError(f"{name} stiffness must be positive definite")


def build_stiffness_field(
    data: Any,
    *,
    matrix_stiffness: Any,
    default_yarn_stiffness: Any = None,
    yarn_stiffness_by_id: Optional[dict] = None,
    output: str = "sparse",
    chunk_voxels: int = 65536,
    validate_positive_definite: bool = False,
    unit: Optional[str] = None,
):
    """Build rotated sparse C21 stiffness from voxel orientation data."""
    orientation = getattr(data, "sparse_orientation", None)
    if not isinstance(orientation, SparseOrientationField):
        raise ValueError("data must contain a SparseOrientationField")
    overrides = {} if yarn_stiffness_by_id is None else {
        int(key): value for key, value in yarn_stiffness_by_id.items()
    }

    matrix_c21 = _coerce_c21_for_field(matrix_stiffness, orientation)
    default_c21 = (
        None
        if default_yarn_stiffness is None
        else _coerce_c21_for_field(default_yarn_stiffness, orientation)
    )
    override_c21 = {
        yarn_id: _coerce_c21_for_field(value, orientation)
        for yarn_id, value in sorted(overrides.items())
    }
    if validate_positive_definite:
        _validate_positive_definite(matrix_c21, "matrix")
        if default_c21 is not None:
            _validate_positive_definite(default_c21, "default yarn")
        for yarn_id, value in override_c21.items():
            _validate_positive_definite(value, f"yarn {yarn_id}")

    count = orientation.num_yarn_voxels
    if orientation.storage == "torch":
        float_dtype = orientation.orientation1.dtype
        device = orientation.orientation1.device
        local_c21 = torch.empty((count, 21), dtype=float_dtype, device=device)
        material_ids = torch.zeros(
            count, dtype=torch.int32, device=device
        )
        if default_c21 is not None:
            local_c21[:] = default_c21
            material_ids.fill_(1)
    else:
        float_dtype = orientation.orientation1.dtype
        local_c21 = np.empty((count, 21), dtype=float_dtype)
        material_ids = np.zeros(count, dtype=np.int32)
        if default_c21 is not None:
            local_c21[:] = default_c21
            material_ids.fill(1)

    for material_id, (yarn_id, value) in enumerate(
        override_c21.items(), start=2
    ):
        mask = orientation.yarn_ids == yarn_id
        local_c21[mask] = value
        material_ids[mask] = material_id

    missing = material_ids == 0
    if _any_true(missing):
        if _is_torch_tensor(orientation.yarn_ids):
            missing_ids = torch.unique(
                orientation.yarn_ids[missing]
            ).detach().cpu().tolist()
        else:
            missing_ids = np.unique(orientation.yarn_ids[missing]).tolist()
        raise ValueError(
            f"missing yarn stiffness for yarn IDs {missing_ids}"
        )

    rotated_c21 = rotate_stiffness_c21(
        local_c21,
        orientation.orientation1,
        orientation.orientation2,
        chunk_voxels=chunk_voxels,
    )
    result = SparseStiffnessField(
        matrix_c21=matrix_c21,
        voxel_indices=_copy_array(orientation.voxel_indices),
        yarn_ids=_copy_array(orientation.yarn_ids),
        material_ids=material_ids,
        yarn_c21=rotated_c21,
        grid_shape=orientation.grid_shape,
        unit=unit,
        order=orientation.order,
    )
    output_normalized = str(output).lower()
    if output_normalized == "sparse":
        return result
    if output_normalized == "dense":
        return result.to_dense_c21()
    raise ValueError('output must be "sparse" or "dense"')


_MATERIAL_FIELD_FORMAT = "pytexgen.sparse_material_fields"
_MATERIAL_FIELD_VERSION = 1
_MATERIAL_FIELD_FILES = {
    "voxel_indices": "voxel_indices.npy",
    "yarn_ids": "yarn_ids.npy",
    "material_ids": "material_ids.npy",
    "orientation1": "orientation1.npy",
    "orientation2": "orientation2.npy",
    "matrix_c21": "matrix_c21.npy",
    "yarn_c21": "yarn_c21.npy",
}


def _material_field_numpy_arrays(
    orientation: SparseOrientationField,
    stiffness: SparseStiffnessField,
):
    if not isinstance(orientation, SparseOrientationField):
        raise TypeError("orientation must be a SparseOrientationField")
    if not isinstance(stiffness, SparseStiffnessField):
        raise TypeError("stiffness must be a SparseStiffnessField")
    orientation_numpy = orientation.to("numpy", copy=False)
    stiffness_numpy = stiffness.to("numpy", copy=False)
    if orientation_numpy.grid_shape != stiffness_numpy.grid_shape:
        raise ValueError("orientation and stiffness grid_shape must match")
    if orientation_numpy.order != stiffness_numpy.order:
        raise ValueError("orientation and stiffness order must match")
    if not np.array_equal(
        orientation_numpy.voxel_indices, stiffness_numpy.voxel_indices
    ):
        raise ValueError("orientation and stiffness voxel_indices must match")
    if not np.array_equal(
        orientation_numpy.yarn_ids, stiffness_numpy.yarn_ids
    ):
        raise ValueError("orientation and stiffness yarn_ids must match")
    arrays = {
        "voxel_indices": orientation_numpy.voxel_indices,
        "yarn_ids": orientation_numpy.yarn_ids,
        "material_ids": stiffness_numpy.material_ids,
        "orientation1": orientation_numpy.orientation1,
        "orientation2": orientation_numpy.orientation2,
        "matrix_c21": stiffness_numpy.matrix_c21,
        "yarn_c21": stiffness_numpy.yarn_c21,
    }
    float_dtypes = {
        np.dtype(arrays[name].dtype)
        for name in (
            "orientation1", "orientation2", "matrix_c21", "yarn_c21"
        )
    }
    if len(float_dtypes) != 1:
        raise ValueError(
            "orientation and stiffness floating-point dtypes must match"
        )
    return orientation_numpy, stiffness_numpy, arrays


def _material_field_metadata(
    orientation: SparseOrientationField,
    stiffness: SparseStiffnessField,
    arrays,
):
    return {
        "format": _MATERIAL_FIELD_FORMAT,
        "format_version": _MATERIAL_FIELD_VERSION,
        "grid_shape": list(orientation.grid_shape),
        "order": orientation.order,
        "voigt_components": list(VOIGT_COMPONENTS),
        "c21_indices": [list(pair) for pair in C21_INDICES],
        "c21_packing": "row-major upper triangle of symmetric 6x6",
        "dtype": str(arrays["orientation1"].dtype),
        "original_device": orientation.device,
        "original_stiffness_device": stiffness.device,
        "unit": stiffness.unit,
        "arrays": dict(_MATERIAL_FIELD_FILES),
    }


def save_material_field_bundle(
    path: Any,
    orientation: SparseOrientationField,
    stiffness: SparseStiffnessField,
    *,
    compressed: bool = True,
) -> None:
    """Persist compact orientation and C21 stiffness fields.

    Saving is an explicit device-to-CPU boundary for Torch/CUDA fields. A path
    ending in ``.npz`` creates one archive; any other path creates the
    memory-mappable directory schema.
    """
    orientation_numpy, stiffness_numpy, arrays = _material_field_numpy_arrays(
        orientation, stiffness
    )
    metadata = _material_field_metadata(orientation, stiffness, arrays)
    out_path = Path(path)
    if out_path.suffix.lower() == ".npz":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(arrays)
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
        saver = np.savez_compressed if compressed else np.savez
        saver(out_path, **payload)
        return

    out_path.mkdir(parents=True, exist_ok=True)
    for name, filename in _MATERIAL_FIELD_FILES.items():
        np.save(out_path / filename, arrays[name], allow_pickle=False)
    (out_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_material_field_metadata(metadata: Any) -> Tuple[int, int, int]:
    if not isinstance(metadata, dict):
        raise ValueError("material field metadata must be a JSON object")
    if metadata.get("format") != _MATERIAL_FIELD_FORMAT:
        raise ValueError("unsupported material field format")
    if int(metadata.get("format_version", -1)) != _MATERIAL_FIELD_VERSION:
        raise ValueError("unsupported material field format version")
    if tuple(metadata.get("voigt_components", ())) != VOIGT_COMPONENTS:
        raise ValueError("material field Voigt component order is invalid")
    c21_indices = tuple(
        tuple(int(value) for value in pair)
        for pair in metadata.get("c21_indices", ())
    )
    if c21_indices != C21_INDICES:
        raise ValueError("material field C21 packing order is invalid")
    if metadata.get("arrays") != _MATERIAL_FIELD_FILES:
        raise ValueError("material field array manifest is invalid")
    return _validate_grid_shape(metadata.get("grid_shape", ()))


def _construct_material_fields(metadata: Any, arrays):
    grid_shape = _validate_material_field_metadata(metadata)
    order = str(metadata.get("order", ""))
    if not order:
        raise ValueError("material field order must not be empty")
    orientation = SparseOrientationField(
        voxel_indices=arrays["voxel_indices"],
        yarn_ids=arrays["yarn_ids"],
        orientation1=arrays["orientation1"],
        orientation2=arrays["orientation2"],
        grid_shape=grid_shape,
        order=order,
    )
    stiffness = SparseStiffnessField(
        matrix_c21=arrays["matrix_c21"],
        voxel_indices=arrays["voxel_indices"],
        yarn_ids=arrays["yarn_ids"],
        material_ids=arrays["material_ids"],
        yarn_c21=arrays["yarn_c21"],
        grid_shape=grid_shape,
        unit=metadata.get("unit"),
        order=order,
    )
    return orientation, stiffness


def load_material_field_bundle(
    path: Any,
    *,
    output: str = "numpy",
    device: Optional[str] = None,
    mmap_mode: Optional[str] = None,
):
    """Load a compact material-field archive as NumPy or Torch arrays."""
    output_normalized = str(output).lower()
    if output_normalized not in {"numpy", "torch"}:
        raise ValueError('output must be "numpy" or "torch"')
    if output_normalized == "torch" and mmap_mode is not None:
        raise ValueError("mmap_mode is incompatible with Torch output")

    in_path = Path(path)
    if in_path.suffix.lower() == ".npz" or in_path.is_file():
        with np.load(in_path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError("material field archive is missing metadata_json")
            metadata = json.loads(str(archive["metadata_json"].item()))
            missing = sorted(set(_MATERIAL_FIELD_FILES) - set(archive.files))
            if missing:
                raise ValueError(
                    f"material field archive is missing arrays: {missing}"
                )
            arrays = {
                name: archive[name].copy() for name in _MATERIAL_FIELD_FILES
            }
    else:
        metadata_path = in_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_material_field_metadata(metadata)
        arrays = {
            name: np.load(
                in_path / filename,
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
            for name, filename in _MATERIAL_FIELD_FILES.items()
        }

    orientation, stiffness = _construct_material_fields(metadata, arrays)
    if output_normalized == "torch":
        orientation = orientation.to("torch", device=device, copy=False)
        stiffness = stiffness.to("torch", device=device, copy=False)
    return orientation, stiffness


def _synchronize_field_device(field: Any) -> None:
    if torch is None:
        return
    if _is_torch_tensor(field):
        device = field.device
    elif getattr(field, "storage", None) == "torch":
        device = field.yarn_c21.device
    else:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def voxelize_textile_material_fields(
    textile: Any,
    *,
    matrix_stiffness: Any,
    default_yarn_stiffness: Any = None,
    yarn_stiffness_by_id: Optional[dict] = None,
    orientation_storage: str = "sparse",
    stiffness_output: str = "sparse",
    **voxel_kwargs,
):
    """Voxelize a textile and build its rotated engineering-Voigt field."""
    requested_orientations = voxel_kwargs.pop("include_orientations", True)
    if requested_orientations is not True:
        raise ValueError(
            "include_orientations=False is incompatible with material fields"
        )
    if str(orientation_storage).lower() != "sparse":
        raise ValueError(
            'orientation_storage must be "sparse" for material field building'
        )

    from TexGen.gpu_voxelizer import voxelize_textile_data

    voxel_kwargs["include_orientations"] = True
    voxel_kwargs["orientation_storage"] = "sparse"
    data = voxelize_textile_data(textile, **voxel_kwargs)

    build_kwargs = {
        "matrix_stiffness": matrix_stiffness,
        "default_yarn_stiffness": default_yarn_stiffness,
        "yarn_stiffness_by_id": yarn_stiffness_by_id,
        "output": stiffness_output,
    }
    if "chunk_voxels" in voxel_kwargs:
        build_kwargs["chunk_voxels"] = voxel_kwargs["chunk_voxels"]

    start = time.perf_counter()
    field = build_stiffness_field(data, **build_kwargs)
    _synchronize_field_device(field)
    data.timings["stiffness_build"] = time.perf_counter() - start
    return data, field


__all__ = [
    "VOIGT_COMPONENTS",
    "C21_INDICES",
    "SparseOrientationField",
    "SparseStiffnessField",
    "pack_voigt_c21",
    "unpack_c21",
    "isotropic_stiffness_c21",
    "orthotropic_stiffness_c21",
    "rotate_stiffness_c21",
    "build_stiffness_field",
    "save_material_field_bundle",
    "load_material_field_bundle",
    "voxelize_textile_material_fields",
]
