"""Versioned native sharding for simulation training datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from .training_data import (
        DatasetQualityPolicy,
        RaggedArray,
        RunningFieldStatistics,
        TrainingDatasetSchema,
        TrainingExample,
        TrainingFieldSpec,
    )
except ImportError:  # pragma: no cover - legacy TexGen package name
    from TexGen.training_data import (
        DatasetQualityPolicy,
        RaggedArray,
        RunningFieldStatistics,
        TrainingDatasetSchema,
        TrainingExample,
        TrainingFieldSpec,
    )


_SCHEMA = "pytexgen.simulation_dataset"
_VERSION = 1
_TOPOLOGY_ALIASES = {
    "stiffness.voxel_indices": "orientation.voxel_indices",
    "stiffness.yarn_ids": "orientation.yarn_ids",
}
_PROVENANCE_FIELDS = {
    "solver_commit",
    "element_formulation",
    "arithmetic_dtype",
    "tolerance",
    "maximum_residual",
    "iteration_count",
    "wall_time_seconds",
    "target_units",
}
_C21_INDICES = tuple(
    (row, column)
    for row in range(6)
    for column in range(row, 6)
)


class DatasetFormatError(ValueError):
    """Dataset metadata or array layout is structurally invalid."""


class DatasetIntegrityError(RuntimeError):
    """Published bytes do not match the declared dataset identity."""


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-compatible") from exc


def _detached_json(value: Any, label: str) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON-compatible mapping")
    try:
        return json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be JSON-compatible") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(
        _thaw_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    _atomic_write_text(path, text + "\n")


def _write_jsonl(path: Path, values: Any) -> None:
    text = "".join(
        _canonical_json(value) + "\n" for value in values
    )
    _atomic_write_text(path, text)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{label} must be a mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return importlib_metadata.version("pytexgen")
    except importlib_metadata.PackageNotFoundError:
        return "source"


def _safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_float(
    value: Any,
    label: str,
    *,
    positive: bool,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"provenance {label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"provenance {label} must be positive")
    if not positive and result < 0.0:
        raise ValueError(f"provenance {label} must be non-negative")
    return result


def _unpack_c21(value: np.ndarray) -> np.ndarray:
    matrix = np.empty(value.shape[:-1] + (6, 6), dtype=value.dtype)
    for index, (row, column) in enumerate(_C21_INDICES):
        matrix[..., row, column] = value[..., index]
        matrix[..., column, row] = value[..., index]
    return matrix


def _validate_array(
    value: Any,
    spec: TrainingFieldSpec,
    *,
    ragged: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"field {spec.name!r} must use CPU NumPy storage"
        )
    expected_shape = (
        (int(value.shape[0]),) + spec.shape
        if ragged and value.ndim >= 1
        else spec.shape
    )
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"field {spec.name!r} shape is {tuple(value.shape)}, "
            f"expected {expected_shape}"
        )
    expected_dtype = np.dtype(spec.dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"field {spec.name!r} dtype is {value.dtype.str}, "
            f"expected {expected_dtype.str}"
        )
    if np.issubdtype(value.dtype, np.floating) and not bool(
        np.isfinite(value).all()
    ):
        raise ValueError(
            f"field {spec.name!r} must contain only finite values"
        )
    return np.array(value, copy=True, order="C")


def _validate_provenance(
    provenance: Any,
    schema: TrainingDatasetSchema,
    quality: DatasetQualityPolicy,
) -> Mapping[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("solver provenance must be a mapping")
    if not provenance:
        if quality.require_solver_provenance:
            raise ValueError(
                "solver provenance must be a non-empty mapping"
            )
        return {}
    data = _detached_json(provenance, "solver provenance")
    if quality.require_solver_provenance:
        missing = _PROVENANCE_FIELDS - set(data)
        if missing:
            raise ValueError(
                "solver provenance is missing: "
                + ", ".join(sorted(missing))
            )
    for name in ("solver_commit", "element_formulation"):
        if name in data:
            data[name] = _validate_identifier(
                data[name], f"provenance {name}"
            )
    if "arithmetic_dtype" in data:
        try:
            arithmetic_dtype = np.dtype(data["arithmetic_dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "provenance arithmetic_dtype is invalid"
            ) from exc
        if arithmetic_dtype.kind != "f":
            raise ValueError(
                "provenance arithmetic_dtype must be floating point"
            )
        data["arithmetic_dtype"] = arithmetic_dtype.name
    if "tolerance" in data:
        data["tolerance"] = _validate_float(
            data["tolerance"], "tolerance", positive=True
        )
    if "maximum_residual" in data:
        data["maximum_residual"] = _validate_float(
            data["maximum_residual"],
            "maximum_residual",
            positive=False,
        )
        threshold = quality.maximum_solver_residual
        if (
            threshold is not None
            and data["maximum_residual"] > threshold
        ):
            raise ValueError(
                "solver residual "
                f"{data['maximum_residual']} exceeds {threshold}"
            )
    if "iteration_count" in data:
        count = data["iteration_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                "provenance iteration_count must be a non-negative integer"
            )
    if "wall_time_seconds" in data:
        data["wall_time_seconds"] = _validate_float(
            data["wall_time_seconds"],
            "wall_time_seconds",
            positive=False,
        )
    units = data.get("target_units")
    if not isinstance(units, Mapping):
        raise ValueError("provenance target_units must be a mapping")
    expected_names = {spec.name for spec in schema.targets}
    if set(units) != expected_names:
        raise ValueError(
            "provenance target_units must declare every target"
        )
    for spec in schema.targets:
        if units[spec.name] != spec.unit:
            raise ValueError(
                f"target {spec.name!r} unit {units[spec.name]!r} "
                f"does not match {spec.unit!r}"
            )
    return data


def _validate_target_quality(
    value: np.ndarray,
    spec: TrainingFieldSpec,
    quality: DatasetQualityPolicy,
) -> None:
    if (
        spec.semantic == "engineering_voigt_c21"
        and quality.validate_target_positive_definite
    ):
        eigenvalues = np.linalg.eigvalsh(_unpack_c21(value))
        if not bool((eigenvalues > 0.0).all()):
            raise ValueError(
                f"target {spec.name!r} must be positive definite"
            )


class SimulationDatasetWriter:
    """Stream SimulationSample-compatible values into native shards."""

    @classmethod
    def create(
        cls,
        path: Any,
        *,
        schema: TrainingDatasetSchema,
        quality: Optional[DatasetQualityPolicy] = None,
        generation: Optional[Mapping[str, Any]] = None,
        resume: bool = False,
    ) -> "SimulationDatasetWriter":
        return cls(
            path,
            schema=schema,
            quality=quality,
            generation=generation,
            resume=resume,
        )

    def __init__(
        self,
        path: Any,
        *,
        schema: TrainingDatasetSchema,
        quality: Optional[DatasetQualityPolicy],
        generation: Optional[Mapping[str, Any]],
        resume: bool,
    ) -> None:
        if not isinstance(schema, TrainingDatasetSchema):
            raise TypeError("schema must be a TrainingDatasetSchema")
        if quality is None:
            quality = DatasetQualityPolicy()
        if not isinstance(quality, DatasetQualityPolicy):
            raise TypeError("quality must be a DatasetQualityPolicy")
        self.target = Path(path)
        if self.target.exists():
            raise FileExistsError(
                f"published target already exists: {self.target}"
            )
        self.staging = self.target.with_name(
            self.target.name + ".incomplete"
        )
        self.schema = schema
        self.quality = quality
        self.generation = _detached_json(
            {} if generation is None else generation,
            "generation",
        )
        self.generation_digest = hashlib.sha256(
            _canonical_json(self.generation).encode("utf-8")
        ).hexdigest()
        self.configuration = {
            "dataset_schema": schema.to_dict(),
            "quality": quality.to_dict(),
            "generation": self.generation,
        }
        self.configuration_digest = hashlib.sha256(
            _canonical_json(self.configuration).encode("utf-8")
        ).hexdigest()
        self._buffer = []
        self._samples = []
        self._rejections = []
        self._shards = []
        self._sample_ids = set()
        self._geometry_digests = set()
        self._group_splits = {}
        self._statistics = {
            name: RunningFieldStatistics(schema.field(name).shape)
            for name in schema.statistics_fields
        }
        self._finalized = False
        if self.staging.exists():
            if not resume:
                raise FileExistsError(
                    f"staging target already exists: {self.staging}"
                )
            self._resume()
            return
        if resume:
            raise FileNotFoundError(
                f"no resumable staging dataset: {self.staging}"
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir()
        (self.staging / "shards").mkdir()
        _atomic_write_json(
            self.staging / "staging.json",
            {
                "schema": _SCHEMA,
                "version": _VERSION,
                **self.configuration,
                "generation_digest": self.generation_digest,
                "configuration_digest": self.configuration_digest,
            },
        )

    def _dataset_path(self, relative: Any) -> Path:
        if not isinstance(relative, str):
            raise DatasetFormatError(
                "dataset paths must be relative strings"
            )
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise DatasetFormatError(
                f"unsafe dataset path {relative!r}"
            )
        root = self.staging.resolve()
        resolved = (self.staging / candidate).resolve()
        if root != resolved and root not in resolved.parents:
            raise DatasetFormatError(
                f"unsafe dataset path {relative!r}"
            )
        return resolved

    def _append_journal(self, value: Mapping[str, Any]) -> None:
        path = self.staging / "journal.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.staging)

    def _verify_resumed_shard(
        self, shard: Mapping[str, Any], expected_number: int
    ) -> None:
        if shard.get("number") != expected_number:
            raise DatasetFormatError(
                "journaled shard numbers must be contiguous"
            )
        expected_path = (
            Path("shards") / f"shard_{expected_number:05d}"
        ).as_posix()
        if shard.get("path") != expected_path:
            raise DatasetFormatError(
                f"journaled shard {expected_number} has an invalid path"
            )
        shard_path = self._dataset_path(expected_path)
        if not shard_path.is_dir():
            raise DatasetFormatError(
                f"missing journaled shard directory {expected_path}"
            )
        files = shard.get("files")
        fields = shard.get("fields")
        if not isinstance(files, Mapping) or not isinstance(
            fields, Mapping
        ):
            raise DatasetFormatError(
                f"journaled shard {expected_number} metadata is invalid"
            )
        for relative, entry in files.items():
            if not isinstance(entry, Mapping):
                raise DatasetFormatError(
                    f"invalid file entry in shard {expected_number}"
                )
            path = self._dataset_path(relative)
            if not path.is_file():
                raise DatasetFormatError(
                    f"missing shard file {relative!r}"
                )
            observed_size = path.stat().st_size
            if observed_size != entry.get("byte_count"):
                raise DatasetIntegrityError(
                    f"byte count mismatch for {relative!r}: "
                    f"expected {entry.get('byte_count')}, "
                    f"observed {observed_size}"
                )
            observed_digest = _sha256_file(path)
            if observed_digest != entry.get("sha256"):
                raise DatasetIntegrityError(
                    f"checksum mismatch for {relative!r}: "
                    f"expected {entry.get('sha256')}, "
                    f"observed {observed_digest}"
                )
        for name, entry in fields.items():
            if not isinstance(entry, Mapping):
                raise DatasetFormatError(
                    f"invalid field entry {name!r}"
                )
            for key in ("values", "offsets"):
                if key in entry and entry[key] not in files:
                    raise DatasetFormatError(
                        f"field {name!r} references undeclared "
                        f"{key} file"
                    )
        stored_shard = _read_json(
            shard_path / "shard.json",
            f"shard {expected_number} metadata",
        )
        if _canonical_json(stored_shard) != _canonical_json(shard):
            raise DatasetIntegrityError(
                f"journal and shard metadata disagree for "
                f"shard {expected_number}"
            )

    def _restore_sample_record(
        self,
        value: Mapping[str, Any],
        shard_number: int,
        expected_row: int,
    ) -> None:
        if not isinstance(value, Mapping):
            raise DatasetFormatError(
                "journal sample records must be mappings"
            )
        sample_id = _validate_identifier(
            value.get("sample_id"), "sample_id"
        )
        group_id = _validate_identifier(
            value.get("group_id"), "group_id"
        )
        split = value.get("split")
        if split not in {"train", "validation", "test"}:
            raise DatasetFormatError(
                f"invalid split in resumed sample {sample_id!r}"
            )
        if value.get("shard") != shard_number or value.get(
            "row"
        ) != expected_row:
            raise DatasetFormatError(
                f"invalid shard row for resumed sample {sample_id!r}"
            )
        geometry = value.get("geometry_digest")
        if not isinstance(geometry, str) or len(geometry) != 64:
            raise DatasetFormatError(
                f"invalid geometry digest for sample {sample_id!r}"
            )
        if sample_id in self._sample_ids:
            raise DatasetIntegrityError(
                f"duplicate sample_id {sample_id!r} in journal"
            )
        if (
            self.quality.require_unique_geometry
            and geometry in self._geometry_digests
        ):
            raise DatasetIntegrityError(
                f"duplicate geometry digest in journal for "
                f"sample {sample_id!r}"
            )
        existing_split = self._group_splits.get(group_id)
        if existing_split is not None and existing_split != split:
            raise DatasetIntegrityError(
                f"group {group_id!r} leaks across resumed splits"
            )
        restored = json.loads(_canonical_json(value))
        self._sample_ids.add(sample_id)
        self._geometry_digests.add(geometry)
        self._group_splits[group_id] = split
        self._samples.append(restored)

    def _restore_statistics(self) -> None:
        arrays = {}

        def load(relative):
            if relative not in arrays:
                arrays[relative] = np.load(
                    self._dataset_path(relative),
                    mmap_mode="r",
                    allow_pickle=False,
                )
            return arrays[relative]

        for sample in self._samples:
            if sample["split"] != "train":
                continue
            shard = self._shards[sample["shard"]]
            row = sample["row"]
            for name, accumulator in self._statistics.items():
                entry = shard["fields"].get(name)
                if not isinstance(entry, Mapping):
                    raise DatasetFormatError(
                        f"statistics field {name!r} is missing from "
                        f"shard {sample['shard']}"
                    )
                values = load(entry["values"])
                if entry.get("layout") == "fixed":
                    value = values[row]
                elif entry.get("layout") == "ragged":
                    offsets = load(entry["offsets"])
                    value = values[
                        int(offsets[row]):int(offsets[row + 1])
                    ]
                else:
                    raise DatasetFormatError(
                        f"statistics field {name!r} has invalid layout"
                    )
                accumulator.update(value)

    def _resume(self) -> None:
        header = _read_json(
            self.staging / "staging.json", "staging metadata"
        )
        if (
            header.get("schema") != _SCHEMA
            or header.get("version") != _VERSION
        ):
            raise DatasetFormatError(
                "unsupported resumable staging schema"
            )
        if header.get(
            "configuration_digest"
        ) != self.configuration_digest or any(
            _canonical_json(header.get(key))
            != _canonical_json(self.configuration[key])
            for key in ("dataset_schema", "quality", "generation")
        ):
            raise ValueError(
                "resume configuration does not match staging dataset"
            )

        journal_path = self.staging / "journal.jsonl"
        journal_records = []
        if journal_path.exists():
            try:
                lines = journal_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except OSError as exc:
                raise DatasetFormatError(
                    "cannot read staging journal"
                ) from exc
            for line_number, line in enumerate(lines, start=1):
                if not line:
                    raise DatasetFormatError(
                        f"empty journal record at line {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"invalid journal record at line {line_number}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise DatasetFormatError(
                        f"journal line {line_number} must be a mapping"
                    )
                journal_records.append(record)

        for record in journal_records:
            record_type = record.get("type")
            if record_type == "shard":
                shard = record.get("shard")
                samples = record.get("samples")
                if not isinstance(shard, Mapping) or not isinstance(
                    samples, list
                ):
                    raise DatasetFormatError(
                        "invalid shard journal record"
                    )
                shard_number = len(self._shards)
                self._verify_resumed_shard(shard, shard_number)
                if len(samples) != shard.get("row_count"):
                    raise DatasetFormatError(
                        f"sample count mismatch for shard {shard_number}"
                    )
                self._shards.append(
                    json.loads(_canonical_json(shard))
                )
                for row, sample in enumerate(samples):
                    self._restore_sample_record(
                        sample, shard_number, row
                    )
            elif record_type == "rejection":
                rejection = record.get("rejection")
                if not isinstance(rejection, Mapping):
                    raise DatasetFormatError(
                        "invalid rejection journal record"
                    )
                self._rejections.append(
                    json.loads(_canonical_json(rejection))
                )
            else:
                raise DatasetFormatError(
                    f"unknown journal record type {record_type!r}"
                )

        expected_directories = {
            f"shard_{index:05d}" for index in range(len(self._shards))
        }
        shards_path = self.staging / "shards"
        if not shards_path.is_dir():
            raise DatasetFormatError("missing staging shards directory")
        trailing = []
        for entry in shards_path.iterdir():
            if entry.name in expected_directories:
                continue
            if (
                not entry.is_dir()
                or not entry.name.startswith("shard_")
                or not entry.name[6:].isdigit()
                or int(entry.name[6:]) < len(self._shards)
            ):
                raise DatasetFormatError(
                    f"unexpected staging shard entry {entry.name!r}"
                )
            trailing.append(entry)

        self._restore_statistics()
        for entry in trailing:
            shutil.rmtree(entry)
        for transient in (
            "dataset.json",
            "samples.jsonl",
            "rejections.jsonl",
        ):
            path = self.staging / transient
            if path.exists():
                path.unlink()

    def __enter__(self) -> "SimulationDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.finalize()
        return False

    def _extract_fields(
        self,
        sample: Any,
        targets: Mapping[str, Any],
    ) -> Mapping[str, np.ndarray]:
        if tuple(sample.voxels.shape) != self.schema.grid_shape:
            raise ValueError(
                f"sample grid {tuple(sample.voxels.shape)} does not match "
                f"schema grid {self.schema.grid_shape}"
            )
        if sample.voxels.order != self.schema.voxel_order:
            raise ValueError("sample voxel order does not match schema")
        expected_targets = {
            spec.name for spec in self.schema.targets
        }
        if not isinstance(targets, Mapping):
            raise ValueError("targets must be a mapping")
        missing = expected_targets - set(targets)
        extra = set(targets) - expected_targets
        if missing or extra:
            details = []
            if missing:
                details.append(
                    "missing target fields: " + ", ".join(sorted(missing))
                )
            if extra:
                details.append(
                    "extra target fields: " + ", ".join(sorted(extra))
                )
            raise ValueError("; ".join(details))

        fields = {}
        group_lengths = {}
        for spec in self.schema.fields:
            if spec.role == "input":
                try:
                    source = sample.array(spec.name, copy=True)
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"sample cannot provide field {spec.name!r}: {exc}"
                    ) from exc
                if (
                    spec.semantic is not None
                    and spec.semantic.endswith(
                        "engineering_voigt_c21"
                    )
                    and spec.unit != sample.materials.unit
                ):
                    raise ValueError(
                        f"field {spec.name!r} unit does not match sample"
                    )
            else:
                source = targets[spec.name]
            value = _validate_array(
                source, spec, ragged=spec.layout == "ragged"
            )
            if spec.layout == "ragged":
                length = int(value.shape[0])
                previous = group_lengths.setdefault(
                    spec.ragged_group, length
                )
                if previous != length:
                    raise ValueError(
                        f"ragged group {spec.ragged_group!r} has "
                        "inconsistent field lengths"
                    )
            if spec.role == "target":
                _validate_target_quality(value, spec, self.quality)
            fields[spec.name] = value

        for alias, canonical in _TOPOLOGY_ALIASES.items():
            if alias in fields and canonical in fields and not np.array_equal(
                fields[alias], fields[canonical]
            ):
                raise ValueError(
                    f"topology aliases {canonical!r} and {alias!r} "
                    "must match exactly"
                )
        return fields

    def append(
        self,
        sample: Any,
        *,
        targets: Mapping[str, Any],
        sample_id: str,
        group_id: str,
        split: str,
        provenance: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after finalize")
        required = ("array", "materials", "storage", "device")
        missing = tuple(
            name for name in required if not hasattr(sample, name)
        )
        if missing or not callable(getattr(sample, "array", None)):
            details = ", ".join(missing) if missing else "callable array"
            raise TypeError(
                "sample must implement the SimulationSample protocol; "
                f"missing {details}"
            )
        if not hasattr(sample.materials, "unit"):
            raise TypeError(
                "sample materials must expose an explicit unit"
            )
        if sample.storage != "numpy" or sample.device != "cpu":
            raise ValueError(
                "dataset writing requires explicit CPU NumPy samples"
            )
        sample_id = _validate_identifier(sample_id, "sample_id")
        group_id = _validate_identifier(group_id, "group_id")
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be train, validation, or test"
            )
        if sample_id in self._sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r}")

        fields = self._extract_fields(sample, targets)
        validated_provenance = _validate_provenance(
            provenance, self.schema, self.quality
        )
        sample_metadata = _detached_json(
            {} if metadata is None else metadata, "sample metadata"
        )
        geometry = _array_digest(
            fields[self.schema.geometry_digest_field]
        )
        if (
            self.quality.require_unique_geometry
            and geometry in self._geometry_digests
        ):
            raise ValueError(
                f"duplicate geometry digest for sample {sample_id!r}"
            )
        existing_split = self._group_splits.get(group_id)
        if existing_split is not None and existing_split != split:
            raise ValueError(
                f"group {group_id!r} already belongs to split "
                f"{existing_split!r}, not {split!r}"
            )

        record = {
            "fields": fields,
            "sample": {
                "sample_id": sample_id,
                "group_id": group_id,
                "split": split,
                "geometry_digest": geometry,
                "metadata": sample_metadata,
                "provenance": validated_provenance,
            },
        }
        self._sample_ids.add(sample_id)
        self._geometry_digests.add(geometry)
        self._group_splits[group_id] = split
        if split == "train":
            for name, accumulator in self._statistics.items():
                accumulator.update(fields[name])
        self._buffer.append(record)
        if len(self._buffer) >= self.schema.shard_size:
            self._flush_shard()

    def reject(
        self,
        *,
        sample_id: str,
        stage: str,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot reject after finalize")
        rejection = {
            "sample_id": _validate_identifier(
                sample_id, "sample_id"
            ),
            "stage": _validate_identifier(stage, "stage"),
            "reason": _validate_identifier(reason, "reason"),
            "metadata": _detached_json(
                {} if metadata is None else metadata,
                "rejection metadata",
            ),
        }
        self._append_journal(
            {"type": "rejection", "rejection": rejection}
        )
        self._rejections.append(rejection)

    def _save_array(
        self,
        shard_path: Path,
        relative: Path,
        value: np.ndarray,
        files: Dict[str, Any],
    ) -> str:
        path = shard_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
        dataset_relative = (
            Path("shards") / shard_path.name / relative
        ).as_posix()
        files[dataset_relative] = {
            "sha256": _sha256_file(path),
            "byte_count": path.stat().st_size,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        return dataset_relative

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        shard_number = len(self._shards)
        shard_path = (
            self.staging / "shards" / f"shard_{shard_number:05d}"
        )
        shard_path.mkdir()
        files = {}
        field_entries = {}
        stored_paths = {}
        group_offsets = {}

        for spec in self.schema.fields:
            storage_name = _TOPOLOGY_ALIASES.get(spec.name, spec.name)
            if storage_name not in {
                item.name for item in self.schema.fields
            }:
                storage_name = spec.name
            role_dir = "fields" if spec.role == "input" else "targets"
            if spec.layout == "fixed":
                if storage_name not in stored_paths:
                    array = np.empty(
                        (len(self._buffer),) + spec.shape,
                        dtype=np.dtype(spec.dtype),
                    )
                    for row, record in enumerate(self._buffer):
                        np.copyto(
                            array[row],
                            record["fields"][spec.name],
                            casting="no",
                        )
                    relative = Path(role_dir) / (
                        _safe_filename(storage_name) + ".npy"
                    )
                    stored_paths[storage_name] = self._save_array(
                        shard_path, relative, array, files
                    )
                field_entries[spec.name] = {
                    "values": stored_paths[storage_name],
                    "layout": "fixed",
                }
                continue

            if spec.ragged_group not in group_offsets:
                group_specs = [
                    candidate
                    for candidate in self.schema.fields
                    if candidate.layout == "ragged"
                    and candidate.ragged_group == spec.ragged_group
                ]
                lengths = [
                    int(
                        record["fields"][
                            group_specs[0].name
                        ].shape[0]
                    )
                    for record in self._buffer
                ]
                offsets = np.empty(
                    len(self._buffer) + 1, dtype=np.int64
                )
                offsets[0] = 0
                np.cumsum(
                    np.asarray(lengths, dtype=np.int64),
                    out=offsets[1:],
                )
                offsets_relative = Path("fields") / (
                    _safe_filename(spec.ragged_group)
                    + ".offsets.npy"
                )
                group_offsets[spec.ragged_group] = (
                    offsets,
                    self._save_array(
                        shard_path,
                        offsets_relative,
                        offsets,
                        files,
                    ),
                )
            offsets, offsets_path = group_offsets[spec.ragged_group]
            if storage_name not in stored_paths:
                values = np.empty(
                    (int(offsets[-1]),) + spec.shape,
                    dtype=np.dtype(spec.dtype),
                )
                for row, record in enumerate(self._buffer):
                    np.copyto(
                        values[offsets[row]:offsets[row + 1]],
                        record["fields"][spec.name],
                        casting="no",
                    )
                relative = Path(role_dir) / (
                    _safe_filename(storage_name) + ".values.npy"
                )
                stored_paths[storage_name] = self._save_array(
                    shard_path, relative, values, files
                )
            field_entries[spec.name] = {
                "values": stored_paths[storage_name],
                "offsets": offsets_path,
                "layout": "ragged",
            }

        sample_records = []
        for row, record in enumerate(self._buffer):
            sample_record = dict(record["sample"])
            sample_record["shard"] = shard_number
            sample_record["row"] = row
            sample_records.append(sample_record)
            self._samples.append(sample_record)
        shard = {
            "number": shard_number,
            "path": (
                Path("shards") / shard_path.name
            ).as_posix(),
            "row_count": len(self._buffer),
            "byte_count": sum(
                entry["byte_count"] for entry in files.values()
            ),
            "files": files,
            "fields": field_entries,
        }
        _atomic_write_json(shard_path / "shard.json", shard)
        self._shards.append(shard)
        self._append_journal(
            {
                "type": "shard",
                "shard": shard,
                "samples": sample_records,
            }
        )
        self._buffer.clear()

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.target.exists():
            raise FileExistsError(
                f"published target already exists: {self.target}"
            )
        self._flush_shard()
        if not self._samples:
            raise ValueError("cannot publish an empty dataset")
        statistics = {
            name: accumulator.finalize(
                unit=self.schema.field(name).unit
            )
            for name, accumulator in self._statistics.items()
        }
        split_counts = Counter(
            record["split"] for record in self._samples
        )
        group_counts = Counter(
            self._group_splits.values()
        )
        _write_jsonl(
            self.staging / "samples.jsonl", self._samples
        )
        _write_jsonl(
            self.staging / "rejections.jsonl", self._rejections
        )
        manifest = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "dataset_schema": self.schema.to_dict(),
            "quality": self.quality.to_dict(),
            "generation": self.generation,
            "generation_digest": self.generation_digest,
            "sample_count": len(self._samples),
            "rejection_count": len(self._rejections),
            "shard_count": len(self._shards),
            "split_counts": {
                split: int(split_counts.get(split, 0))
                for split in ("train", "validation", "test")
            },
            "group_counts": {
                split: int(group_counts.get(split, 0))
                for split in ("train", "validation", "test")
            },
            "statistics": statistics,
            "shards": self._shards,
            "c21": {
                "component_order": [
                    "xx",
                    "yy",
                    "zz",
                    "yz",
                    "xz",
                    "xy",
                ],
                "packing": "row-major-upper-triangle",
            },
            "provenance": {
                "pytexgen_version": _package_version(),
                "git_commit": os.environ.get("PYTEXGEN_GIT_COMMIT"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        _atomic_write_json(self.staging / "dataset.json", manifest)
        os.replace(self.staging, self.target)
        _fsync_directory(self.target.parent)
        self._finalized = True


class SimulationDataset:
    """Picklable map-style reader with selective lazy mmap access."""

    def __init__(
        self,
        path: Any,
        *,
        split: Optional[str] = None,
        inputs: Optional[Tuple[str, ...]] = None,
        targets: Optional[Tuple[str, ...]] = None,
        verify: str = "shard",
        transform: Any = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise DatasetFormatError(
                f"dataset directory does not exist: {self.path}"
            )
        self._root = self.path.resolve()
        if verify not in {"manifest", "shard", "sample"}:
            raise ValueError(
                'verify must be "manifest", "shard", or "sample"'
            )
        if split is not None and split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "split must be train, validation, test, or None"
            )
        self.verify = verify
        self.split = split
        self.transform = transform
        self.epoch = 0
        self._manifest = _read_json(
            self.path / "dataset.json", "dataset manifest"
        )
        if self._manifest.get("schema") != _SCHEMA:
            raise DatasetFormatError(
                f"unsupported dataset schema "
                f"{self._manifest.get('schema')!r}"
            )
        if self._manifest.get("version") != _VERSION:
            raise DatasetFormatError(
                f"unsupported dataset version "
                f"{self._manifest.get('version')!r}"
            )
        try:
            self.schema = TrainingDatasetSchema.from_dict(
                self._manifest.get("dataset_schema")
            )
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"invalid dataset schema: {exc}"
            ) from exc

        self._shards = self._validate_manifest_shards()
        self._file_entries = {}
        for shard in self._shards:
            for relative, entry in shard["files"].items():
                if relative in self._file_entries:
                    raise DatasetFormatError(
                        f"duplicate file path {relative!r}"
                    )
                self._file_entries[relative] = entry
        self._samples = self._load_samples()
        self._rejection_count = self._load_rejection_count()
        self.input_names = self._selected_names(
            inputs, self.schema.inputs, "input"
        )
        self.target_names = self._selected_names(
            targets, self.schema.targets, "target"
        )
        self._indices = tuple(
            index
            for index, sample in enumerate(self._samples)
            if split is None or sample["split"] == split
        )
        self._cache_pid = None
        self._arrays = {}
        self._verified_files = set()
        self._validated_offsets = set()
        self._checked_files = 0
        self._checked_samples = 0
        if verify == "sample":
            self._verify_complete_dataset()

    def _resolve_path(self, relative: Any) -> Path:
        if not isinstance(relative, str):
            raise DatasetFormatError(
                "manifest file paths must be strings"
            )
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise DatasetFormatError(
                f"unsafe manifest path {relative!r}"
            )
        resolved = (self.path / path).resolve()
        if (
            self._root != resolved
            and self._root not in resolved.parents
        ):
            raise DatasetFormatError(
                f"unsafe manifest path {relative!r}"
            )
        return resolved

    def _validate_file_entry(
        self,
        relative: Any,
        entry: Any,
        shard_number: int,
    ) -> Mapping[str, Any]:
        path = self._resolve_path(relative)
        if not path.is_file():
            raise DatasetFormatError(
                f"missing file {relative!r} in shard {shard_number}"
            )
        if not isinstance(entry, Mapping):
            raise DatasetFormatError(
                f"invalid file metadata for {relative!r}"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DatasetFormatError(
                f"invalid checksum metadata for {relative!r}"
            )
        byte_count = entry.get("byte_count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise DatasetFormatError(
                f"invalid byte count metadata for {relative!r}"
            )
        try:
            dtype = np.dtype(entry.get("dtype"))
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"invalid dtype metadata for {relative!r}"
            ) from exc
        if dtype.kind not in {"b", "i", "u", "f"}:
            raise DatasetFormatError(
                f"unsafe dtype metadata for {relative!r}"
            )
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in shape
            )
        ):
            raise DatasetFormatError(
                f"invalid shape metadata for {relative!r}"
            )
        return {
            "sha256": digest,
            "byte_count": byte_count,
            "dtype": dtype.str,
            "shape": shape,
        }

    def _validate_manifest_shards(self) -> Tuple[Mapping[str, Any], ...]:
        shards = self._manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetFormatError("manifest shards must be a list")
        if self._manifest.get("shard_count") != len(shards):
            raise DatasetFormatError("manifest shard_count mismatch")
        validated = []
        all_paths = set()
        fields_by_name = {
            spec.name: spec for spec in self.schema.fields
        }
        for number, shard in enumerate(shards):
            if not isinstance(shard, Mapping):
                raise DatasetFormatError(
                    f"shard {number} metadata must be a mapping"
                )
            expected_path = (
                Path("shards") / f"shard_{number:05d}"
            ).as_posix()
            if (
                shard.get("number") != number
                or shard.get("path") != expected_path
            ):
                raise DatasetFormatError(
                    f"shard {number} number/path mismatch"
                )
            row_count = shard.get("row_count")
            if (
                isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count <= 0
            ):
                raise DatasetFormatError(
                    f"shard {number} row_count is invalid"
                )
            self._resolve_path(shard["path"])
            files = shard.get("files")
            field_entries = shard.get("fields")
            if not isinstance(files, Mapping) or not isinstance(
                field_entries, Mapping
            ):
                raise DatasetFormatError(
                    f"shard {number} files/fields are invalid"
                )
            validated_files = {}
            for relative, entry in files.items():
                if relative in all_paths:
                    raise DatasetFormatError(
                        f"duplicate file path {relative!r}"
                    )
                all_paths.add(relative)
                validated_files[relative] = self._validate_file_entry(
                    relative, entry, number
                )
            if set(field_entries) != set(fields_by_name):
                raise DatasetFormatError(
                    f"shard {number} field registry does not match schema"
                )
            validated_fields = {}
            for name, entry in field_entries.items():
                if not isinstance(entry, Mapping):
                    raise DatasetFormatError(
                        f"invalid field entry {name!r}"
                    )
                spec = fields_by_name[name]
                if entry.get("layout") != spec.layout:
                    raise DatasetFormatError(
                        f"layout mismatch for field {name!r}"
                    )
                values_relative = entry.get("values")
                self._resolve_path(values_relative)
                if values_relative not in validated_files:
                    raise DatasetFormatError(
                        f"field {name!r} references undeclared values"
                    )
                values_meta = validated_files[values_relative]
                if values_meta["dtype"] != np.dtype(spec.dtype).str:
                    raise DatasetFormatError(
                        f"dtype mismatch for field {name!r}"
                    )
                if spec.layout == "fixed":
                    expected_shape = [row_count, *spec.shape]
                    if values_meta["shape"] != expected_shape:
                        raise DatasetFormatError(
                            f"shape mismatch for field {name!r}"
                        )
                    validated_fields[name] = {
                        "layout": "fixed",
                        "values": values_relative,
                    }
                    continue
                if values_meta["shape"][1:] != list(spec.shape):
                    raise DatasetFormatError(
                        f"shape mismatch for ragged field {name!r}"
                    )
                offsets_relative = entry.get("offsets")
                self._resolve_path(offsets_relative)
                if offsets_relative not in validated_files:
                    raise DatasetFormatError(
                        f"field {name!r} references undeclared offsets"
                    )
                offsets_meta = validated_files[offsets_relative]
                if (
                    np.dtype(offsets_meta["dtype"]) != np.dtype(np.int64)
                    or offsets_meta["shape"] != [row_count + 1]
                ):
                    raise DatasetFormatError(
                        f"offset metadata mismatch for field {name!r}"
                    )
                validated_fields[name] = {
                    "layout": "ragged",
                    "values": values_relative,
                    "offsets": offsets_relative,
                }
            for alias, canonical in _TOPOLOGY_ALIASES.items():
                if (
                    alias in validated_fields
                    and canonical in validated_fields
                    and validated_fields[alias]["values"]
                    != validated_fields[canonical]["values"]
                ):
                    raise DatasetFormatError(
                        f"topology alias fields {canonical!r} and "
                        f"{alias!r} must reference one values file"
                    )
            ragged_offsets = {}
            for spec in self.schema.fields:
                if spec.layout != "ragged":
                    continue
                relative = validated_fields[spec.name]["offsets"]
                previous = ragged_offsets.setdefault(
                    spec.ragged_group, relative
                )
                if previous != relative:
                    raise DatasetFormatError(
                        f"ragged group {spec.ragged_group!r} must "
                        "reference one shared offsets file"
                    )
            validated.append(
                {
                    "number": number,
                    "path": expected_path,
                    "row_count": row_count,
                    "byte_count": shard.get("byte_count"),
                    "files": validated_files,
                    "fields": validated_fields,
                }
            )
        return tuple(validated)

    def _load_jsonl(self, name: str) -> list:
        path = self.path / name
        if not path.is_file():
            raise DatasetFormatError(f"missing dataset index {name}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DatasetFormatError(
                f"cannot read dataset index {name}"
            ) from exc
        result = []
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise DatasetFormatError(
                    f"empty {name} line {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"invalid {name} line {line_number}"
                ) from exc
            if not isinstance(value, Mapping):
                raise DatasetFormatError(
                    f"{name} line {line_number} must be a mapping"
                )
            result.append(value)
        return result

    def _load_samples(self) -> Tuple[Mapping[str, Any], ...]:
        samples = self._load_jsonl("samples.jsonl")
        if self._manifest.get("sample_count") != len(samples):
            raise DatasetFormatError("manifest sample_count mismatch")
        seen_ids = set()
        group_splits = {}
        occupied_rows = set()
        split_counts = Counter()
        for sample in samples:
            sample_id = sample.get("sample_id")
            group_id = sample.get("group_id")
            split = sample.get("split")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in seen_ids
            ):
                raise DatasetFormatError(
                    f"invalid or duplicate sample_id {sample_id!r}"
                )
            if not isinstance(group_id, str) or not group_id:
                raise DatasetFormatError(
                    f"invalid group_id for sample {sample_id!r}"
                )
            if split not in {"train", "validation", "test"}:
                raise DatasetFormatError(
                    f"invalid split for sample {sample_id!r}"
                )
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise DatasetIntegrityError(
                    f"group {group_id!r} leaks across splits "
                    f"{previous!r} and {split!r}"
                )
            shard_number = sample.get("shard")
            row = sample.get("row")
            if (
                isinstance(shard_number, bool)
                or not isinstance(shard_number, int)
                or not 0 <= shard_number < len(self._shards)
                or isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < self._shards[
                    shard_number
                ]["row_count"]
            ):
                raise DatasetFormatError(
                    f"invalid shard row for sample {sample_id!r}"
                )
            location = (shard_number, row)
            if location in occupied_rows:
                raise DatasetFormatError(
                    f"duplicate sample row {location}"
                )
            digest = sample.get("geometry_digest")
            if not isinstance(digest, str) or len(digest) != 64:
                raise DatasetFormatError(
                    f"invalid geometry digest for sample {sample_id!r}"
                )
            seen_ids.add(sample_id)
            occupied_rows.add(location)
            split_counts[split] += 1
        expected_rows = {
            (shard["number"], row)
            for shard in self._shards
            for row in range(shard["row_count"])
        }
        if occupied_rows != expected_rows:
            raise DatasetFormatError(
                "sample index does not cover every shard row exactly once"
            )
        declared_splits = self._manifest.get("split_counts")
        observed_splits = {
            split: int(split_counts.get(split, 0))
            for split in ("train", "validation", "test")
        }
        if declared_splits != observed_splits:
            raise DatasetFormatError("manifest split_counts mismatch")
        declared_groups = self._manifest.get("group_counts")
        observed_groups = Counter(group_splits.values())
        expected_groups = {
            split: int(observed_groups.get(split, 0))
            for split in ("train", "validation", "test")
        }
        if declared_groups != expected_groups:
            raise DatasetFormatError("manifest group_counts mismatch")
        return tuple(
            json.loads(_canonical_json(sample)) for sample in samples
        )

    def _load_rejection_count(self) -> int:
        rejections = self._load_jsonl("rejections.jsonl")
        if self._manifest.get("rejection_count") != len(rejections):
            raise DatasetFormatError(
                "manifest rejection_count mismatch"
            )
        return len(rejections)

    @staticmethod
    def _selected_names(
        requested: Any,
        specs: Tuple[TrainingFieldSpec, ...],
        role: str,
    ) -> Tuple[str, ...]:
        available = tuple(spec.name for spec in specs)
        if requested is None:
            return available
        names = tuple(requested)
        if len(set(names)) != len(names):
            raise ValueError(f"selected {role} fields must be unique")
        unknown = set(names) - set(available)
        if unknown:
            raise ValueError(
                f"unknown selected {role} fields: "
                + ", ".join(sorted(unknown))
            )
        return names

    def _ensure_process_cache(self) -> None:
        process_id = os.getpid()
        if self._cache_pid == process_id:
            return
        self._cache_pid = process_id
        self._arrays = {}
        self._verified_files = set()
        self._validated_offsets = set()

    def _ensure_verified(self, relative: str) -> None:
        if self.verify == "manifest" or relative in self._verified_files:
            return
        entry = self._file_entries[relative]
        path = self._resolve_path(relative)
        observed_size = path.stat().st_size
        if observed_size != entry["byte_count"]:
            raise DatasetIntegrityError(
                f"byte count mismatch for {relative!r}: expected "
                f"{entry['byte_count']}, observed {observed_size}"
            )
        observed = _sha256_file(path)
        if observed != entry["sha256"]:
            raise DatasetIntegrityError(
                f"checksum mismatch for {relative!r}: expected "
                f"{entry['sha256']}, observed {observed}"
            )
        self._verified_files.add(relative)
        self._checked_files += 1

    def _open_array(self, relative: str) -> np.ndarray:
        self._ensure_process_cache()
        self._ensure_verified(relative)
        if relative not in self._arrays:
            try:
                value = np.load(
                    self._resolve_path(relative),
                    mmap_mode="r",
                    allow_pickle=False,
                )
            except (OSError, ValueError) as exc:
                raise DatasetFormatError(
                    f"cannot open array {relative!r}"
                ) from exc
            entry = self._file_entries[relative]
            if value.dtype.str != entry["dtype"]:
                raise DatasetFormatError(
                    f"dtype mismatch while opening {relative!r}"
                )
            if list(value.shape) != entry["shape"]:
                raise DatasetFormatError(
                    f"shape mismatch while opening {relative!r}"
                )
            self._arrays[relative] = value
        return self._arrays[relative]

    def _validated_ragged_offsets(
        self,
        relative: str,
        *,
        values_count: int,
        row_count: int,
    ) -> np.ndarray:
        offsets = self._open_array(relative)
        key = (relative, values_count, row_count)
        if key not in self._validated_offsets:
            if (
                offsets.dtype != np.dtype(np.int64)
                or offsets.shape != (row_count + 1,)
                or int(offsets[0]) != 0
                or bool((offsets[1:] < offsets[:-1]).any())
                or int(offsets[-1]) != values_count
            ):
                raise DatasetFormatError(
                    f"invalid ragged offsets in {relative!r}"
                )
            self._validated_offsets.add(key)
        return offsets

    def _read_field(
        self,
        shard_number: int,
        row: int,
        name: str,
    ) -> Any:
        shard = self._shards[shard_number]
        entry = shard["fields"][name]
        values = self._open_array(entry["values"])
        if entry["layout"] == "fixed":
            return values[row]
        offsets = self._validated_ragged_offsets(
            entry["offsets"],
            values_count=int(values.shape[0]),
            row_count=shard["row_count"],
        )
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        return RaggedArray(
            values[start:stop],
            np.asarray([0, stop - start], dtype=np.int64),
        )

    def _verify_complete_dataset(self) -> None:
        self._ensure_process_cache()
        for relative in self._file_entries:
            self._ensure_verified(relative)
        for shard in self._shards:
            for spec in self.schema.fields:
                entry = shard["fields"][spec.name]
                if entry["layout"] == "ragged":
                    values = self._open_array(entry["values"])
                    self._validated_ragged_offsets(
                        entry["offsets"],
                        values_count=int(values.shape[0]),
                        row_count=shard["row_count"],
                    )
        digest_field = self.schema.geometry_digest_field
        for sample in self._samples:
            value = self._read_field(
                sample["shard"], sample["row"], digest_field
            )
            array = value.values if isinstance(
                value, RaggedArray
            ) else value
            observed = _array_digest(np.asarray(array))
            if observed != sample["geometry_digest"]:
                raise DatasetIntegrityError(
                    f"geometry digest mismatch for sample "
                    f"{sample['sample_id']!r}: expected "
                    f"{sample['geometry_digest']}, observed {observed}"
                )
            self._checked_samples += 1

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> TrainingExample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        sample = self._samples[self._indices[index]]
        inputs = {
            name: self._read_field(
                sample["shard"], sample["row"], name
            )
            for name in self.input_names
        }
        targets = {
            name: self._read_field(
                sample["shard"], sample["row"], name
            )
            for name in self.target_names
        }
        metadata = dict(sample.get("metadata", {}))
        metadata["provenance"] = sample.get("provenance", {})
        metadata["geometry_digest"] = sample["geometry_digest"]
        example = TrainingExample(
            inputs=inputs,
            targets=targets,
            sample_id=sample["sample_id"],
            group_id=sample["group_id"],
            split=sample["split"],
            metadata=metadata,
        )
        if self.transform is not None:
            example = self.transform(example, self.schema)
            if not isinstance(example, TrainingExample):
                raise TypeError(
                    "dataset transform must return a TrainingExample"
                )
        return example

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("epoch must be an integer")
        self.epoch = epoch
        if self.transform is not None and hasattr(
            self.transform, "set_epoch"
        ):
            self.transform.set_epoch(epoch)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache_pid"] = None
        state["_arrays"] = {}
        state["_verified_files"] = set()
        state["_validated_offsets"] = set()
        state["_checked_files"] = 0
        return state

    def audit_report(self) -> Mapping[str, Any]:
        stored_bytes = sum(
            entry["byte_count"]
            for entry in self._file_entries.values()
        )
        logical_bytes = sum(
            int(np.prod(entry["shape"], dtype=np.int64))
            * np.dtype(entry["dtype"]).itemsize
            for entry in self._file_entries.values()
        )
        return {
            "ok": True,
            "schema": _SCHEMA,
            "version": _VERSION,
            "sample_count": len(self._samples),
            "rejection_count": self._rejection_count,
            "shard_count": len(self._shards),
            "split_counts": dict(self._manifest["split_counts"]),
            "group_count": sum(
                self._manifest["group_counts"].values()
            ),
            "stored_bytes": stored_bytes,
            "logical_bytes": logical_bytes,
            "checked_files": self._checked_files,
            "checked_samples": self._checked_samples,
        }


def audit_simulation_dataset(
    path: Any, *, verify: str = "sample"
) -> Mapping[str, Any]:
    """Validate a published dataset and return a JSON-safe audit report."""
    dataset = SimulationDataset(path, verify=verify)
    return dataset.audit_report()


__all__ = [
    "SimulationDataset",
    "DatasetFormatError",
    "DatasetIntegrityError",
    "SimulationDatasetWriter",
    "audit_simulation_dataset",
]
