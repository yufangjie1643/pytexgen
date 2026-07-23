"""Framework-neutral contracts for simulation training datasets.

The module deliberately depends only on NumPy.  Optional framework adapters
live in :mod:`TexGen.torch_training`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Optional, Tuple

import numpy as np


VOXEL_ORDER = "ix + iy*nx + iz*nx*ny"
SPLITS = ("train", "validation", "test")
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_FIELD_KEYS = {
    "name",
    "role",
    "layout",
    "dtype",
    "shape",
    "unit",
    "semantic",
    "ragged_group",
}
_SCHEMA_KEYS = {
    "inputs",
    "targets",
    "grid_shape",
    "voxel_order",
    "shard_size",
    "statistics_fields",
    "geometry_digest_field",
}


def _validate_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _FIELD_NAME.fullmatch(value)
        or ".." in value
    ):
        raise ValueError(
            f"{label} must be a safe dotted identifier beginning with a letter"
        )
    return value


def _canonical_shape(value: Any) -> Tuple[int, ...]:
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise ValueError("shape must be a tuple of positive integers") from exc
    if any(
        isinstance(item, bool)
        or not isinstance(item, Integral)
        or int(item) <= 0
        for item in shape
    ):
        raise ValueError("shape dimensions must be positive integers")
    return tuple(int(item) for item in shape)


def _canonical_dtype(value: Any) -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dtype {value!r} is not a valid NumPy dtype") from exc
    if dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(
            "dtype must be a non-object Boolean, integer, or floating dtype"
        )
    return dtype.str


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validated_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a JSON-compatible mapping")
    try:
        detached = json.loads(
            json.dumps(dict(value), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-compatible") from exc
    return _freeze_json(detached)


def _freeze_field_mapping(
    value: Any, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = {}
    for name, array in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[name] = array
    return MappingProxyType(result)


@dataclass(frozen=True)
class TrainingFieldSpec:
    """Immutable declaration of one stored input or target field."""

    name: str
    role: str
    layout: str
    dtype: str
    shape: Tuple[int, ...]
    unit: Optional[str] = None
    semantic: Optional[str] = None
    ragged_group: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, "name"))
        if self.role not in {"input", "target"}:
            raise ValueError('role must be "input" or "target"')
        if self.layout not in {"fixed", "ragged"}:
            raise ValueError('layout must be "fixed" or "ragged"')
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        object.__setattr__(self, "shape", _canonical_shape(self.shape))

        unit = self.unit
        if unit is not None:
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError("unit must be a non-empty string or None")
            unit = unit.strip()
        object.__setattr__(self, "unit", unit)

        semantic = self.semantic
        if semantic is not None:
            if not isinstance(semantic, str) or not semantic.strip():
                raise ValueError(
                    "semantic must be a non-empty string or None"
                )
            semantic = semantic.strip()
        object.__setattr__(self, "semantic", semantic)

        if self.layout == "ragged":
            group = _validate_name(self.ragged_group, "ragged_group")
        elif self.ragged_group is not None:
            raise ValueError(
                "ragged_group is valid only when layout is ragged"
            )
        else:
            group = None
        object.__setattr__(self, "ragged_group", group)

        if semantic == "engineering_voigt_c21":
            if not self.shape or self.shape[-1] != 21:
                raise ValueError(
                    "engineering_voigt_c21 fields must end with shape (21,)"
                )
            if unit is None:
                raise ValueError(
                    "engineering_voigt_c21 fields require an explicit unit"
                )
            if np.dtype(self.dtype).kind != "f":
                raise ValueError(
                    "engineering_voigt_c21 fields require a floating dtype"
                )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "layout": self.layout,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "semantic": self.semantic,
            "ragged_group": self.ragged_group,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingFieldSpec":
        if not isinstance(value, Mapping):
            raise ValueError("field specification must be a mapping")
        unknown = set(value) - _FIELD_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown field keys: {names}")
        required = {"name", "role", "layout", "dtype", "shape"}
        missing = required - set(value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing field keys: {names}")
        return cls(**dict(value))


@dataclass(frozen=True)
class TrainingDatasetSchema:
    """Immutable registry and spatial contract for one dataset."""

    inputs: Tuple[TrainingFieldSpec, ...]
    targets: Tuple[TrainingFieldSpec, ...]
    grid_shape: Tuple[int, int, int]
    voxel_order: str
    shard_size: int
    statistics_fields: Tuple[str, ...] = ()
    geometry_digest_field: str = "voxel.material_id"

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        targets = tuple(self.targets)
        if not inputs:
            raise ValueError("inputs must contain at least one field")
        if any(not isinstance(item, TrainingFieldSpec) for item in inputs):
            raise TypeError("inputs must contain TrainingFieldSpec values")
        if any(not isinstance(item, TrainingFieldSpec) for item in targets):
            raise TypeError("targets must contain TrainingFieldSpec values")
        if any(item.role != "input" for item in inputs):
            raise ValueError("every inputs field must have role input")
        if any(item.role != "target" for item in targets):
            raise ValueError("every targets field must have role target")

        names = [item.name for item in inputs + targets]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique; duplicate found")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)

        try:
            shape = _canonical_shape(self.grid_shape)
        except ValueError as exc:
            raise ValueError(f"invalid grid_shape: {exc}") from exc
        if len(shape) != 3:
            raise ValueError("grid_shape must contain exactly three dimensions")
        object.__setattr__(self, "grid_shape", shape)
        if self.voxel_order != VOXEL_ORDER:
            raise ValueError(
                f"voxel_order must be {VOXEL_ORDER!r} in schema version 1"
            )
        if (
            isinstance(self.shard_size, bool)
            or not isinstance(self.shard_size, Integral)
            or int(self.shard_size) <= 0
        ):
            raise ValueError("shard_size must be a positive integer")
        object.__setattr__(self, "shard_size", int(self.shard_size))

        statistics = tuple(self.statistics_fields)
        if len(set(statistics)) != len(statistics):
            raise ValueError("statistics_fields must be unique")
        fields = {item.name: item for item in inputs + targets}
        unknown_statistics = set(statistics) - set(fields)
        if unknown_statistics:
            names_text = ", ".join(sorted(unknown_statistics))
            raise ValueError(
                f"statistics fields are not declared: {names_text}"
            )
        non_float = [
            name
            for name in statistics
            if np.dtype(fields[name].dtype).kind != "f"
        ]
        if non_float:
            raise ValueError(
                "statistics fields must use floating dtypes: "
                + ", ".join(non_float)
            )
        object.__setattr__(self, "statistics_fields", statistics)

        if self.geometry_digest_field not in fields:
            raise ValueError(
                "geometry_digest_field must name a declared input field"
            )
        if fields[self.geometry_digest_field].role != "input":
            raise ValueError(
                "geometry_digest_field must name a declared input field"
            )

    @property
    def fields(self) -> Tuple[TrainingFieldSpec, ...]:
        return self.inputs + self.targets

    def field(self, name: str) -> TrainingFieldSpec:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(f"unknown field {name!r}")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "inputs": [item.to_dict() for item in self.inputs],
            "targets": [item.to_dict() for item in self.targets],
            "grid_shape": list(self.grid_shape),
            "voxel_order": self.voxel_order,
            "shard_size": self.shard_size,
            "statistics_fields": list(self.statistics_fields),
            "geometry_digest_field": self.geometry_digest_field,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "TrainingDatasetSchema":
        if not isinstance(value, Mapping):
            raise ValueError("dataset schema must be a mapping")
        unknown = set(value) - _SCHEMA_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown schema keys: {names}")
        required = {
            "inputs",
            "targets",
            "grid_shape",
            "voxel_order",
            "shard_size",
        }
        missing = required - set(value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing schema keys: {names}")
        data = dict(value)
        data["inputs"] = tuple(
            TrainingFieldSpec.from_dict(item) for item in data["inputs"]
        )
        data["targets"] = tuple(
            TrainingFieldSpec.from_dict(item) for item in data["targets"]
        )
        return cls(**data)


@dataclass(frozen=True)
class DatasetQualityPolicy:
    """Validation gates applied before a generated label is accepted."""

    validate_target_positive_definite: bool = True
    maximum_solver_residual: Optional[float] = 1e-8
    require_solver_provenance: bool = True
    require_unique_geometry: bool = True

    def __post_init__(self) -> None:
        for name in (
            "validate_target_positive_definite",
            "require_solver_provenance",
            "require_unique_geometry",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be Boolean")
        residual = self.maximum_solver_residual
        if residual is not None and (
            isinstance(residual, bool)
            or not isinstance(residual, Real)
            or not math.isfinite(float(residual))
            or float(residual) <= 0.0
        ):
            raise ValueError(
                "maximum_solver_residual must be finite and positive or None"
            )
        if residual is not None:
            object.__setattr__(
                self, "maximum_solver_residual", float(residual)
            )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "validate_target_positive_definite": (
                self.validate_target_positive_definite
            ),
            "maximum_solver_residual": self.maximum_solver_residual,
            "require_solver_provenance": self.require_solver_provenance,
            "require_unique_geometry": self.require_unique_geometry,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "DatasetQualityPolicy":
        if not isinstance(value, Mapping):
            raise ValueError("quality policy must be a mapping")
        expected = {
            "validate_target_positive_definite",
            "maximum_solver_residual",
            "require_solver_provenance",
            "require_unique_geometry",
        }
        unknown = set(value) - expected
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown quality policy keys: {names}")
        return cls(**dict(value))


@dataclass(frozen=True)
class RaggedArray:
    """Values plus cumulative sample offsets for a ragged field."""

    values: Any
    offsets: Any

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim < 1:
            raise ValueError(
                "ragged values must be a NumPy array with a row dimension"
            )
        if (
            not isinstance(self.offsets, np.ndarray)
            or self.offsets.ndim != 1
            or not np.issubdtype(self.offsets.dtype, np.integer)
        ):
            raise ValueError(
                "ragged offsets must be a one-dimensional integer array"
            )
        if self.offsets.size == 0 or int(self.offsets[0]) != 0:
            raise ValueError("ragged offsets must start at zero")
        if bool(np.any(self.offsets[1:] < self.offsets[:-1])):
            raise ValueError("ragged offsets must be monotonically increasing")
        if int(self.offsets[-1]) != int(self.values.shape[0]):
            raise ValueError(
                "ragged offsets must end at the number of value rows"
            )


@dataclass(frozen=True)
class TrainingExample:
    """One selected sample with stable identity and immutable metadata."""

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    sample_id: str
    group_id: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if self.split not in SPLITS:
            raise ValueError(
                f"split must be one of {', '.join(SPLITS)}"
            )
        object.__setattr__(self, "sample_id", self.sample_id.strip())
        object.__setattr__(self, "group_id", self.group_id.strip())
        object.__setattr__(
            self, "inputs", _freeze_field_mapping(self.inputs, "inputs")
        )
        object.__setattr__(
            self, "targets", _freeze_field_mapping(self.targets, "targets")
        )
        overlap = set(self.inputs) & set(self.targets)
        if overlap:
            raise ValueError(
                "inputs and targets must not contain duplicate field names"
            )
        object.__setattr__(
            self, "metadata", _validated_metadata(self.metadata)
        )


__all__ = [
    "DatasetQualityPolicy",
    "RaggedArray",
    "SPLITS",
    "TrainingDatasetSchema",
    "TrainingExample",
    "TrainingFieldSpec",
    "VOXEL_ORDER",
]
