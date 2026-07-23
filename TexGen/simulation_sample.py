"""Validated tensor contracts for simulation and learning consumers.

The public stiffness convention is the engineering-Voigt component order
``(xx, yy, zz, yz, xz, xy)`` with symmetric matrices packed as 21 upper-
triangle coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

try:
    from .material_fields import unpack_c21
except ImportError:  # pragma: no cover - legacy TexGen package name
    from TexGen.material_fields import unpack_c21


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _is_integer_array(value: Any) -> bool:
    if _is_torch_tensor(value):
        return value.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.integer)
    )


def _is_float_array(value: Any) -> bool:
    if _is_torch_tensor(value):
        return value.dtype in {
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        }
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.floating)
    )


def _all_finite(value: Any) -> bool:
    if _is_torch_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _numpy_float_dtype(dtype: Any):
    if dtype is None:
        return None
    if torch is not None and isinstance(dtype, torch.dtype):
        mapping = {
            torch.float16: np.dtype(np.float16),
            torch.float32: np.dtype(np.float32),
            torch.float64: np.dtype(np.float64),
        }
        if dtype not in mapping:
            raise ValueError("dtype must be a supported floating-point dtype")
        return mapping[dtype]
    result = np.dtype(dtype)
    if not np.issubdtype(result, np.floating):
        raise ValueError("dtype must be a floating-point dtype")
    return result


def _torch_float_dtype(dtype: Any, fallback: Any):
    if torch is None:
        raise ImportError(
            'Torch storage requested; install with `pip install "pytexgen[gpu]"`.'
        )
    if dtype is None:
        if isinstance(fallback, torch.dtype):
            return fallback
        return {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
        }.get(np.dtype(fallback), torch.float64)
    if isinstance(dtype, torch.dtype):
        result = dtype
    else:
        result = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
        }.get(np.dtype(dtype))
    if result not in {torch.float16, torch.float32, torch.float64}:
        raise ValueError("dtype must be a supported floating-point dtype")
    return result


def _validate_storage(storage: Optional[str], fallback: str) -> str:
    target = fallback if storage is None else str(storage).lower()
    if target not in {"numpy", "torch"}:
        raise ValueError('storage must be "numpy", "torch", or None')
    return target


def _validate_numpy_device(device: Optional[str]) -> None:
    if device is not None and str(device).lower() != "cpu":
        raise ValueError("NumPy storage is available only on the CPU")


def _convert_float_array(
    value: Any,
    storage: str,
    *,
    device: Optional[str],
    dtype: Any,
    copy: bool,
):
    if storage == "numpy":
        _validate_numpy_device(device)
        target_dtype = _numpy_float_dtype(dtype)
        if _is_torch_tensor(value):
            result = value.detach().cpu().numpy()
            if copy:
                result = result.copy()
        else:
            result = np.array(value, copy=copy)
        if target_dtype is not None:
            result = result.astype(
                target_dtype,
                copy=copy or result.dtype != target_dtype,
            )
        return result

    fallback = value.dtype
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
        _validate_numpy_device(device)
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


def _validate_positive_definite(c21: Any) -> None:
    matrices = unpack_c21(c21)
    if _is_torch_tensor(matrices):
        valid = bool((torch.linalg.eigvalsh(matrices) > 0.0).all().item())
    else:
        valid = bool((np.linalg.eigvalsh(matrices) > 0.0).all())
    if not valid:
        raise ValueError("material stiffness must be positive definite")


@dataclass(frozen=True)
class MaterialTable:
    """Local material stiffness rows addressed by explicit material IDs."""

    c21: Any
    material_ids: Any
    unit: str
    names: Optional[Tuple[str, ...]] = None
    validate_positive_definite: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not _is_float_array(self.c21) or self.c21.ndim != 2:
            raise ValueError("c21 must be a floating-point array with shape (M, 21)")
        if self.c21.shape[1] != 21:
            raise ValueError("c21 must have shape (M, 21)")
        if not _is_integer_array(self.material_ids) or self.material_ids.ndim != 1:
            raise ValueError("material_ids must be a one-dimensional integer array")
        if self.material_ids.shape[0] != self.c21.shape[0]:
            raise ValueError("material_ids and c21 must contain the same number of rows")
        if _is_torch_tensor(self.c21) != _is_torch_tensor(self.material_ids):
            raise ValueError("c21 and material_ids must use the same storage backend")
        if (
            _is_torch_tensor(self.c21)
            and self.c21.device != self.material_ids.device
        ):
            raise ValueError("c21 and material_ids must use the same device")
        if not _all_finite(self.c21):
            raise ValueError("c21 must contain only finite values")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be a non-empty string")

        if _is_torch_tensor(self.material_ids):
            if bool((self.material_ids < 0).any().item()):
                raise ValueError("material_ids must be non-negative")
            unique = torch.unique(self.material_ids)
            unique_count = int(unique.numel())
            matrix_count = int((self.material_ids == 0).sum().item())
        else:
            if bool((self.material_ids < 0).any()):
                raise ValueError("material_ids must be non-negative")
            unique_count = int(np.unique(self.material_ids).size)
            matrix_count = int(np.count_nonzero(self.material_ids == 0))
        if unique_count != int(self.material_ids.shape[0]):
            raise ValueError("material_ids must be unique")
        if matrix_count != 1:
            raise ValueError("material ID 0 must occur exactly once")

        names = None if self.names is None else tuple(self.names)
        if names is not None:
            if len(names) != int(self.c21.shape[0]) or not all(
                isinstance(name, str) for name in names
            ):
                raise ValueError("names must contain one string per material")
            object.__setattr__(self, "names", names)
        object.__setattr__(self, "unit", self.unit.strip())

        if self.validate_positive_definite:
            _validate_positive_definite(self.c21)

    @property
    def storage(self) -> str:
        return "torch" if _is_torch_tensor(self.c21) else "numpy"

    @property
    def device(self) -> str:
        return str(self.c21.device) if _is_torch_tensor(self.c21) else "cpu"

    def row_for_id(self, material_id: int) -> int:
        matches = self.material_ids == int(material_id)
        if _is_torch_tensor(matches):
            indices = matches.nonzero(as_tuple=False).reshape(-1)
            count = int(indices.shape[0])
            row = int(indices[0].item()) if count else -1
        else:
            indices = np.flatnonzero(matches)
            count = int(indices.shape[0])
            row = int(indices[0]) if count else -1
        if count != 1:
            raise KeyError(f"unknown material ID {material_id}")
        return row

    def c21_for_id(self, material_id: int):
        return self.c21[self.row_for_id(material_id)]

    def to(
        self,
        storage: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        copy: bool = False,
    ) -> "MaterialTable":
        """Convert table arrays while preserving integer identifier dtypes."""
        target = _validate_storage(storage, self.storage)
        return MaterialTable(
            c21=_convert_float_array(
                self.c21,
                target,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            material_ids=_convert_integer_array(
                self.material_ids,
                target,
                device=device,
                copy=copy,
            ),
            unit=self.unit,
            names=self.names,
            validate_positive_definite=self.validate_positive_definite,
        )


__all__ = ["MaterialTable"]
