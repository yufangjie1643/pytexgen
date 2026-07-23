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


def _is_torch_tensor(value: Any) -> bool:
    value_type = type(value)
    return (
        value_type.__module__.split(".", 1)[0] == "torch"
        and value_type.__name__ == "Tensor"
    )


def _array_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if _is_torch_tensor(value):
        return int(value.numel() * value.element_size())
    raise TypeError(
        "batch fields must contain NumPy arrays or Torch tensors"
    )


def _trusted_ragged(values: Any, offsets: Any) -> "RaggedArray":
    result = object.__new__(RaggedArray)
    object.__setattr__(result, "values", values)
    object.__setattr__(result, "offsets", offsets)
    return result


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
        values_supported = isinstance(
            self.values, np.ndarray
        ) or _is_torch_tensor(self.values)
        offsets_supported = isinstance(
            self.offsets, np.ndarray
        ) or _is_torch_tensor(self.offsets)
        if not values_supported or self.values.ndim < 1:
            raise ValueError(
                "ragged values must be an array with a row dimension"
            )
        if isinstance(self.offsets, np.ndarray):
            integer_offsets = np.issubdtype(
                self.offsets.dtype, np.integer
            )
        else:
            integer_offsets = str(
                getattr(self.offsets, "dtype", "")
            ) in {
                "torch.int8",
                "torch.int16",
                "torch.int32",
                "torch.int64",
                "torch.uint8",
            }
        if not offsets_supported or self.offsets.ndim != 1 or not integer_offsets:
            raise ValueError(
                "ragged offsets must be a one-dimensional integer array"
            )
        offsets_count = (
            int(self.offsets.size)
            if isinstance(self.offsets, np.ndarray)
            else int(self.offsets.numel())
        )
        if offsets_count == 0 or int(self.offsets[0].item()) != 0:
            raise ValueError("ragged offsets must start at zero")
        if isinstance(self.offsets, np.ndarray):
            decreasing = bool(
                np.any(self.offsets[1:] < self.offsets[:-1])
            )
            final_offset = int(self.offsets[-1])
        else:
            decreasing = bool(
                (self.offsets[1:] < self.offsets[:-1]).any().item()
            )
            final_offset = int(self.offsets[-1].item())
        if decreasing:
            raise ValueError("ragged offsets must be monotonically increasing")
        if final_offset != int(self.values.shape[0]):
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


def _validate_example_fields(
    example: TrainingExample,
    schema: TrainingDatasetSchema,
) -> None:
    expected_inputs = {item.name for item in schema.inputs}
    expected_targets = {item.name for item in schema.targets}
    for label, observed, expected in (
        ("inputs", set(example.inputs), expected_inputs),
        ("targets", set(example.targets), expected_targets),
    ):
        missing = expected - observed
        extra = observed - expected
        if missing:
            raise ValueError(
                f"{label} missing fields: {', '.join(sorted(missing))}"
            )
        if extra:
            raise ValueError(
                f"{label} extra fields: {', '.join(sorted(extra))}"
            )

    group_lengths = {}
    for spec in schema.fields:
        mapping = example.inputs if spec.role == "input" else example.targets
        value = mapping[spec.name]
        expected_dtype = np.dtype(spec.dtype)
        if spec.layout == "fixed":
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"field {spec.name!r} must be a NumPy array"
                )
            if tuple(value.shape) != spec.shape:
                raise ValueError(
                    f"field {spec.name!r} shape is {tuple(value.shape)}, "
                    f"expected {spec.shape}"
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"field {spec.name!r} dtype is {value.dtype.str}, "
                    f"expected {expected_dtype.str}"
                )
            continue

        if not isinstance(value, RaggedArray):
            raise TypeError(f"field {spec.name!r} must be a RaggedArray")
        if not isinstance(value.values, np.ndarray):
            raise TypeError(
                f"field {spec.name!r} values must be a NumPy array"
            )
        expected_shape = (int(value.values.shape[0]),) + spec.shape
        if tuple(value.values.shape) != expected_shape:
            raise ValueError(
                f"field {spec.name!r} shape is "
                f"{tuple(value.values.shape)}, expected {expected_shape}"
            )
        if value.values.dtype != expected_dtype:
            raise ValueError(
                f"field {spec.name!r} dtype is {value.values.dtype.str}, "
                f"expected {expected_dtype.str}"
            )
        if int(value.offsets.size) != 2:
            raise ValueError(
                f"field {spec.name!r} sample offsets must have length 2"
            )
        length = int(value.values.shape[0])
        previous = group_lengths.setdefault(spec.ragged_group, length)
        if previous != length:
            raise ValueError(
                f"ragged group {spec.ragged_group!r} has inconsistent "
                "sample lengths"
            )


def _collate_fixed_field(
    examples: Tuple[TrainingExample, ...],
    spec: TrainingFieldSpec,
) -> np.ndarray:
    result = np.empty(
        (len(examples),) + spec.shape,
        dtype=np.dtype(spec.dtype),
        order="C",
    )
    for row, example in enumerate(examples):
        source = (
            example.inputs[spec.name]
            if spec.role == "input"
            else example.targets[spec.name]
        )
        np.copyto(result[row], source, casting="no")
    return result


def _collate_ragged_fields(
    examples: Tuple[TrainingExample, ...],
    schema: TrainingDatasetSchema,
) -> Mapping[str, RaggedArray]:
    groups = {}
    for spec in schema.fields:
        if spec.layout == "ragged":
            groups.setdefault(spec.ragged_group, []).append(spec)

    result = {}
    for group_name, specs in groups.items():
        lengths = []
        for example in examples:
            mapping = (
                example.inputs
                if specs[0].role == "input"
                else example.targets
            )
            lengths.append(
                int(mapping[specs[0].name].values.shape[0])
            )
        offsets = np.empty(len(examples) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        total = int(offsets[-1])
        for spec in specs:
            values = np.empty(
                (total,) + spec.shape,
                dtype=np.dtype(spec.dtype),
                order="C",
            )
            for row, example in enumerate(examples):
                mapping = (
                    example.inputs
                    if spec.role == "input"
                    else example.targets
                )
                source = mapping[spec.name].values
                np.copyto(
                    values[offsets[row]:offsets[row + 1]],
                    source,
                    casting="no",
                )
            result[spec.name] = RaggedArray(values, offsets)
    return result


@dataclass
class SimulationBatch:
    """Owned fixed/ragged arrays plus CPU-only sample metadata."""

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    sample_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    metadata: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        self.inputs = MappingProxyType(dict(self.inputs))
        self.targets = MappingProxyType(dict(self.targets))
        self.sample_ids = tuple(self.sample_ids)
        self.group_ids = tuple(self.group_ids)
        self.metadata = tuple(self.metadata)
        count = len(self.sample_ids)
        if len(self.group_ids) != count or len(self.metadata) != count:
            raise ValueError(
                "sample_ids, group_ids, and metadata must have equal length"
            )

    def _map_arrays(self, operation) -> "SimulationBatch":
        cache = {}

        def convert(value):
            if isinstance(value, RaggedArray):
                return _trusted_ragged(
                    convert(value.values), convert(value.offsets)
                )
            key = id(value)
            if key not in cache:
                cache[key] = operation(value)
            return cache[key]

        return SimulationBatch(
            inputs={
                name: convert(value)
                for name, value in self.inputs.items()
            },
            targets={
                name: convert(value)
                for name, value in self.targets.items()
            },
            sample_ids=self.sample_ids,
            group_ids=self.group_ids,
            metadata=self.metadata,
        )

    def pin_memory(self) -> "SimulationBatch":
        def pin(value):
            method = getattr(value, "pin_memory", None)
            if method is None:
                raise TypeError(
                    "pin_memory requires a batch of Torch tensors"
                )
            return method()

        return self._map_arrays(pin)

    def to(
        self, device: Any, *, non_blocking: bool = False
    ) -> "SimulationBatch":
        def move(value):
            method = getattr(value, "to", None)
            if method is None:
                raise TypeError("to() requires a batch of Torch tensors")
            return method(device, non_blocking=non_blocking)

        return self._map_arrays(move)

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "inputs": self.inputs,
                "targets": self.targets,
                "sample_ids": self.sample_ids,
                "group_ids": self.group_ids,
                "metadata": self.metadata,
            }
        )

    @property
    def nbytes(self) -> int:
        seen = set()

        def count(value):
            if isinstance(value, RaggedArray):
                return count(value.values) + count(value.offsets)
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            return _array_nbytes(value)

        return sum(count(value) for value in self.inputs.values()) + sum(
            count(value) for value in self.targets.values()
        )


def collate_training_examples(
    examples: Any,
    schema: TrainingDatasetSchema,
) -> SimulationBatch:
    """Copy selected examples into one owned contiguous NumPy batch."""
    if not isinstance(schema, TrainingDatasetSchema):
        raise TypeError("schema must be a TrainingDatasetSchema")
    examples_tuple = tuple(examples)
    if not examples_tuple:
        raise ValueError("examples must not be empty")
    if any(
        not isinstance(example, TrainingExample)
        for example in examples_tuple
    ):
        raise TypeError("examples must contain TrainingExample values")
    for example in examples_tuple:
        _validate_example_fields(example, schema)

    ragged = _collate_ragged_fields(examples_tuple, schema)
    inputs = {}
    targets = {}
    for spec in schema.fields:
        value = (
            _collate_fixed_field(examples_tuple, spec)
            if spec.layout == "fixed"
            else ragged[spec.name]
        )
        destination = inputs if spec.role == "input" else targets
        destination[spec.name] = value
    return SimulationBatch(
        inputs=inputs,
        targets=targets,
        sample_ids=tuple(item.sample_id for item in examples_tuple),
        group_ids=tuple(item.group_id for item in examples_tuple),
        metadata=tuple(item.metadata for item in examples_tuple),
    )


def as_torch_batch(batch: SimulationBatch) -> SimulationBatch:
    """Create zero-copy Torch CPU tensor views of an owned NumPy batch."""
    if not isinstance(batch, SimulationBatch):
        raise TypeError("batch must be a SimulationBatch")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required; install with `pip install "pytexgen[gpu]"`.'
        ) from exc

    def convert(value):
        if _is_torch_tensor(value):
            return value
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "Torch conversion requires NumPy arrays or Torch tensors"
            )
        if not value.flags.c_contiguous or not value.flags.writeable:
            raise ValueError(
                "Torch conversion requires owned writable contiguous arrays"
            )
        return torch.from_numpy(value)

    return batch._map_arrays(convert)


__all__ = [
    "DatasetQualityPolicy",
    "RaggedArray",
    "SPLITS",
    "SimulationBatch",
    "TrainingDatasetSchema",
    "TrainingExample",
    "TrainingFieldSpec",
    "VOXEL_ORDER",
    "as_torch_batch",
    "collate_training_examples",
]
