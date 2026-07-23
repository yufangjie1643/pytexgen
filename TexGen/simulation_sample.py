"""Validated tensor contracts for simulation and learning consumers.

The public stiffness convention is the engineering-Voigt component order
``(xx, yy, zz, yz, xz, xy)`` with symmetric matrices packed as 21 upper-
triangle coefficients.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

try:
    from .gpu_voxelizer import VoxelGridData
    from .material_fields import (
        SparseOrientationField,
        SparseStiffnessField,
        unpack_c21,
        voxelize_textile_material_fields,
    )
except ImportError:  # pragma: no cover - legacy TexGen package name
    from TexGen.gpu_voxelizer import VoxelGridData
    from TexGen.material_fields import (
        SparseOrientationField,
        SparseStiffnessField,
        unpack_c21,
        voxelize_textile_material_fields,
    )


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


def _copy_array(value: Any):
    if _is_torch_tensor(value):
        return value.clone()
    return np.array(value, copy=True)


def _array_backend(value: Any) -> str:
    return "torch" if _is_torch_tensor(value) else "numpy"


def _array_device(value: Any) -> str:
    return str(value.device) if _is_torch_tensor(value) else "cpu"


def _array_equal(left: Any, right: Any) -> bool:
    if _is_torch_tensor(left):
        return _is_torch_tensor(right) and bool(torch.equal(left, right))
    return not _is_torch_tensor(right) and bool(
        np.array_equal(np.asarray(left), np.asarray(right))
    )


def _array_allclose(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if _is_torch_tensor(left):
        return _is_torch_tensor(right) and bool(
            torch.allclose(left, right, rtol=rtol, atol=atol)
        )
    return not _is_torch_tensor(right) and bool(
        np.allclose(left, right, rtol=rtol, atol=atol)
    )


def _freeze_json(value: Any):
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_metadata(metadata: Mapping[str, Any]):
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON-compatible mapping")
    try:
        detached = json.loads(
            json.dumps(dict(metadata), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-compatible") from exc
    return _freeze_json(detached)


FIELD_ORDER = (
    "voxel.yarn_id",
    "voxel.material_id",
    "voxel.occupancy",
    "orientation.voxel_indices",
    "orientation.yarn_ids",
    "orientation.primary",
    "orientation.secondary",
    "stiffness.matrix_c21",
    "stiffness.voxel_indices",
    "stiffness.yarn_ids",
    "stiffness.material_ids",
    "stiffness.yarn_c21",
    "material.ids",
    "material.c21",
)


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


@dataclass(frozen=True)
class SimulationSample:
    """Validated composition of voxel, direction, stiffness, and material data."""

    voxels: VoxelGridData
    materials: MaterialTable
    orientation: Optional[SparseOrientationField] = None
    stiffness: Optional[SparseStiffnessField] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.voxels, VoxelGridData):
            raise TypeError("voxels must be a VoxelGridData")
        if not isinstance(self.materials, MaterialTable):
            raise TypeError("materials must be a MaterialTable")
        if self.orientation is not None and not isinstance(
            self.orientation, SparseOrientationField
        ):
            raise TypeError("orientation must be a SparseOrientationField or None")
        if self.stiffness is not None and not isinstance(
            self.stiffness, SparseStiffnessField
        ):
            raise TypeError("stiffness must be a SparseStiffnessField or None")

        embedded_orientation = self.voxels.sparse_orientation
        if self.orientation is None and embedded_orientation is not None:
            object.__setattr__(self, "orientation", embedded_orientation)
        elif (
            embedded_orientation is not None
            and self.orientation is not embedded_orientation
        ):
            raise ValueError(
                "orientation and voxels.sparse_orientation must be the same object"
            )

        self._validate_voxel_arrays()
        self._validate_components()
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def storage(self) -> str:
        return self.voxels.storage

    @property
    def device(self) -> str:
        return self.voxels.device

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(name for name in FIELD_ORDER if self._field_available(name))

    def _validate_voxel_arrays(self) -> None:
        arrays = (
            self.voxels.yarn_id,
            self.voxels.aabb,
            self.voxels.centers,
            self.voxels.orientation1,
            self.voxels.orientation2,
        )
        arrays = tuple(value for value in arrays if value is not None)
        expected_storage = _array_backend(self.voxels.yarn_id)
        expected_device = _array_device(self.voxels.yarn_id)
        if self.voxels.storage != expected_storage:
            raise ValueError(
                "VoxelGridData storage metadata does not match its arrays"
            )
        if str(self.voxels.device) != expected_device:
            raise ValueError(
                "VoxelGridData device metadata does not match its arrays"
            )
        if any(_array_backend(value) != expected_storage for value in arrays):
            raise ValueError(
                "all resident voxel arrays must use the same storage backend"
            )
        if any(_array_device(value) != expected_device for value in arrays):
            raise ValueError(
                "all resident voxel arrays must use the same device"
            )

    def _validate_components(self) -> None:
        expected_shape = tuple(self.voxels.shape)
        expected_order = str(self.voxels.order)
        expected_storage = self.storage
        expected_device = self.device

        if self.materials.storage != expected_storage:
            raise ValueError(
                "all sample arrays must use the same storage backend"
            )
        if self.materials.device != expected_device:
            raise ValueError("all sample arrays must use the same device")

        if self.orientation is not None:
            if tuple(self.orientation.grid_shape) != expected_shape:
                raise ValueError("orientation grid shape does not match voxels")
            if self.orientation.order != expected_order:
                raise ValueError("orientation voxel order does not match voxels")
            if self.orientation.storage != expected_storage:
                raise ValueError(
                    "all sample arrays must use the same storage backend"
                )
            if self.orientation.device != expected_device:
                raise ValueError("all sample arrays must use the same device")

        if self.stiffness is None:
            return
        if tuple(self.stiffness.grid_shape) != expected_shape:
            raise ValueError("stiffness grid shape does not match voxels")
        if self.stiffness.order != expected_order:
            raise ValueError("stiffness voxel order does not match voxels")
        if self.stiffness.storage != expected_storage:
            raise ValueError("all sample arrays must use the same storage backend")
        if self.stiffness.device != expected_device:
            raise ValueError("all sample arrays must use the same device")
        if self.stiffness.unit != self.materials.unit:
            raise ValueError(
                "stiffness unit must match the material table unit"
            )

        if self.orientation is not None:
            if not _array_equal(
                self.orientation.voxel_indices,
                self.stiffness.voxel_indices,
            ):
                raise ValueError(
                    "orientation and stiffness voxel indices must match exactly"
                )
            if not _array_equal(
                self.orientation.yarn_ids,
                self.stiffness.yarn_ids,
            ):
                raise ValueError(
                    "orientation and stiffness yarn IDs must match exactly"
                )

        self._validate_stiffness_material_ids()
        matrix = self.materials.c21_for_id(0)
        itemsize = (
            self.stiffness.matrix_c21.element_size()
            if _is_torch_tensor(self.stiffness.matrix_c21)
            else self.stiffness.matrix_c21.dtype.itemsize
        )
        rtol, atol = (
            (1e-5, 1e-6) if int(itemsize) <= 4 else (1e-10, 1e-12)
        )
        if not _array_allclose(
            self.stiffness.matrix_c21,
            matrix,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "stiffness matrix stiffness must match material ID 0"
            )

    def _validate_stiffness_material_ids(self) -> None:
        if _is_torch_tensor(self.stiffness.material_ids):
            for material_id in torch.unique(self.stiffness.material_ids):
                value = int(material_id.item())
                if not bool((self.materials.material_ids == value).any().item()):
                    raise ValueError(f"unknown material ID {value} in stiffness")
            return
        used_ids = np.unique(self.stiffness.material_ids)
        known_ids = self.materials.material_ids
        missing = used_ids[~np.isin(used_ids, known_ids)]
        if missing.size:
            raise ValueError(
                f"unknown material ID {int(missing[0])} in stiffness"
            )

    def _field_available(self, name: str) -> bool:
        if name in {"voxel.yarn_id", "voxel.occupancy"}:
            return True
        if name == "voxel.material_id":
            return self.stiffness is not None
        if name.startswith("orientation."):
            return self.orientation is not None
        if name.startswith("stiffness."):
            return self.stiffness is not None
        if name.startswith("material."):
            return True
        return False

    def _resident_field(self, name: str):
        fields = {
            "voxel.yarn_id": lambda: self.voxels.grid,
            "orientation.voxel_indices": (
                lambda: self.orientation.voxel_indices
            ),
            "orientation.yarn_ids": lambda: self.orientation.yarn_ids,
            "orientation.primary": lambda: self.orientation.orientation1,
            "orientation.secondary": lambda: self.orientation.orientation2,
            "stiffness.matrix_c21": lambda: self.stiffness.matrix_c21,
            "stiffness.voxel_indices": (
                lambda: self.stiffness.voxel_indices
            ),
            "stiffness.yarn_ids": lambda: self.stiffness.yarn_ids,
            "stiffness.material_ids": lambda: self.stiffness.material_ids,
            "stiffness.yarn_c21": lambda: self.stiffness.yarn_c21,
            "material.ids": lambda: self.materials.material_ids,
            "material.c21": lambda: self.materials.c21,
        }
        getter = fields.get(name)
        return None if getter is None else getter()

    def _materialize_voxel_field(self, name: str):
        if name == "voxel.occupancy":
            return self.voxels.grid >= 0
        if name != "voxel.material_id":
            raise KeyError(f"unknown derived field {name!r}")

        shape = tuple(self.voxels.shape)
        if _is_torch_tensor(self.stiffness.material_ids):
            result = torch.zeros(
                shape,
                dtype=self.stiffness.material_ids.dtype,
                device=self.stiffness.material_ids.device,
            )
            result.reshape(-1)[
                self.stiffness.voxel_indices.to(dtype=torch.long)
            ] = self.stiffness.material_ids
            return result
        result = np.zeros(shape, dtype=self.stiffness.material_ids.dtype)
        result.reshape(-1)[
            np.asarray(self.stiffness.voxel_indices, dtype=np.int64)
        ] = self.stiffness.material_ids
        return result

    def array(
        self,
        name: str,
        *,
        layout: str = "native",
        copy: bool = False,
    ):
        """Return one named resident field without implicit conversion."""
        if name not in self.field_names:
            available = ", ".join(self.field_names)
            raise KeyError(
                f"unknown or unavailable field {name!r}; "
                f"available fields: {available}"
            )
        if layout == "acdm":
            if name != "stiffness.yarn_c21":
                raise ValueError(
                    f"layout {layout!r} is not supported for {name!r}"
                )
            if not copy:
                raise ValueError("ACDM layout allocates; pass copy=True")
            return self.stiffness.to_acdm(batch=True)
        if layout != "native":
            raise ValueError(
                f"layout {layout!r} is not supported for {name!r}"
            )
        resident = self._resident_field(name)
        if resident is None:
            if not copy:
                raise ValueError(f"{name!r} is derived; pass copy=True")
            return self._materialize_voxel_field(name)
        return _copy_array(resident) if copy else resident

    def as_dict(
        self,
        *,
        layout: str = "native",
        copy: bool = False,
    ):
        """Return available fields, excluding allocating fields by default."""
        if copy:
            names = self.field_names
        else:
            names = tuple(
                name for name in self.field_names
                if self._resident_field(name) is not None
            )
        return {
            name: self.array(name, layout=layout, copy=copy)
            for name in names
        }

    def _float_arrays(self):
        arrays = [
            self.voxels.aabb,
            self.voxels.centers,
            self.voxels.orientation1,
            self.voxels.orientation2,
            self.materials.c21,
        ]
        if self.orientation is not None:
            arrays.extend(
                (
                    self.orientation.orientation1,
                    self.orientation.orientation2,
                )
            )
        if self.stiffness is not None:
            arrays.extend(
                (
                    self.stiffness.matrix_c21,
                    self.stiffness.yarn_c21,
                )
            )
        return tuple(value for value in arrays if value is not None)

    def _dtype_is_identity(self, storage: str, dtype: Any) -> bool:
        if dtype is None:
            return True
        arrays = self._float_arrays()
        if storage == "numpy":
            expected = _numpy_float_dtype(dtype)
            return all(np.dtype(value.dtype) == expected for value in arrays)
        expected = _torch_float_dtype(dtype, arrays[0].dtype)
        return all(value.dtype == expected for value in arrays)

    def _device_is_identity(
        self,
        storage: str,
        device: Optional[str],
    ) -> bool:
        if storage == "numpy":
            _validate_numpy_device(device)
            return self.storage == "numpy"
        if self.storage != "torch":
            return False
        if device is None:
            return True
        requested = torch.device(device)
        current = torch.device(self.device)
        return (
            requested.type == current.type
            and (
                requested.index is None
                or requested.index == current.index
            )
        )

    def _convert_stiffness(
        self,
        storage: str,
        *,
        device: Optional[str],
        dtype: Any,
        copy: bool,
        orientation: Optional[SparseOrientationField],
    ):
        if self.stiffness is None:
            return None
        if (
            orientation is not None
            and _array_equal(
                self.orientation.voxel_indices,
                self.stiffness.voxel_indices,
            )
            and _array_equal(
                self.orientation.yarn_ids,
                self.stiffness.yarn_ids,
            )
        ):
            voxel_indices = orientation.voxel_indices
            yarn_ids = orientation.yarn_ids
        else:
            voxel_indices = _convert_integer_array(
                self.stiffness.voxel_indices,
                storage,
                device=device,
                copy=copy,
            )
            yarn_ids = _convert_integer_array(
                self.stiffness.yarn_ids,
                storage,
                device=device,
                copy=copy,
            )
        return SparseStiffnessField(
            matrix_c21=_convert_float_array(
                self.stiffness.matrix_c21,
                storage,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            voxel_indices=voxel_indices,
            yarn_ids=yarn_ids,
            material_ids=_convert_integer_array(
                self.stiffness.material_ids,
                storage,
                device=device,
                copy=copy,
            ),
            yarn_c21=_convert_float_array(
                self.stiffness.yarn_c21,
                storage,
                device=device,
                dtype=dtype,
                copy=copy,
            ),
            grid_shape=self.stiffness.grid_shape,
            unit=self.stiffness.unit,
            order=self.stiffness.order,
        )

    def to(
        self,
        storage: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        copy: bool = False,
    ) -> "SimulationSample":
        """Explicitly convert all resident arrays as one validated sample."""
        target = _validate_storage(storage, self.storage)
        if (
            not copy
            and target == self.storage
            and self._device_is_identity(target, device)
            and self._dtype_is_identity(target, dtype)
        ):
            return self

        conversion_device = device if target == "torch" else None
        if target == "numpy":
            _validate_numpy_device(device)
        voxels = self.voxels.to(
            target,
            device=conversion_device,
            dtype=dtype,
            copy=copy,
        )
        if self.orientation is None:
            orientation = None
        elif self.orientation is self.voxels.sparse_orientation:
            orientation = voxels.sparse_orientation
        else:
            orientation = self.orientation.to(
                target,
                device=conversion_device,
                dtype=dtype,
                copy=copy,
            )
        stiffness = self._convert_stiffness(
            target,
            device=conversion_device,
            dtype=dtype,
            copy=copy,
            orientation=orientation,
        )
        materials = self.materials.to(
            target,
            device=conversion_device,
            dtype=dtype,
            copy=copy,
        )
        result = SimulationSample(
            voxels=voxels,
            orientation=orientation,
            stiffness=stiffness,
            materials=materials,
            metadata={},
        )
        object.__setattr__(result, "metadata", self.metadata)
        return result


def voxelize_textile_simulation_sample(
    textile: Any,
    *,
    materials: MaterialTable,
    default_yarn_material_id: Optional[int] = None,
    yarn_material_id_by_id: Optional[Mapping[int, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **voxel_kwargs,
) -> SimulationSample:
    """Voxelize a textile and build a validated sparse simulation sample."""
    if not isinstance(materials, MaterialTable):
        raise TypeError("materials must be a MaterialTable")
    yarn_material_ids = (
        {}
        if yarn_material_id_by_id is None
        else {
            int(yarn_id): int(material_id)
            for yarn_id, material_id in yarn_material_id_by_id.items()
        }
    )
    default_stiffness = (
        None
        if default_yarn_material_id is None
        else materials.c21_for_id(int(default_yarn_material_id))
    )
    yarn_stiffness = {
        yarn_id: materials.c21_for_id(material_id)
        for yarn_id, material_id in yarn_material_ids.items()
    }
    data, stiffness = voxelize_textile_material_fields(
        textile,
        matrix_stiffness=materials.c21_for_id(0),
        default_yarn_stiffness=default_stiffness,
        yarn_stiffness_by_id=yarn_stiffness,
        default_yarn_material_id=default_yarn_material_id,
        yarn_material_id_by_id=yarn_material_ids,
        unit=materials.unit,
        orientation_storage="sparse",
        stiffness_output="sparse",
        **voxel_kwargs,
    )

    float_dtype = stiffness.matrix_c21.dtype
    if (
        materials.storage == stiffness.storage
        and materials.device == stiffness.device
        and materials.c21.dtype == float_dtype
    ):
        resolved_materials = materials
    else:
        resolved_materials = materials.to(
            stiffness.storage,
            device=(
                stiffness.device
                if stiffness.storage == "torch"
                else None
            ),
            dtype=float_dtype,
        )
    return SimulationSample(
        voxels=data,
        orientation=data.sparse_orientation,
        stiffness=stiffness,
        materials=resolved_materials,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "MaterialTable",
    "SimulationSample",
    "voxelize_textile_simulation_sample",
]
