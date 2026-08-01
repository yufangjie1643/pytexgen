"""Prepared-geometry storage and streaming batch voxelization.

``.tg3`` remains TexGen's editable XML model format.  ``.ptgb`` is a compact,
read-only cache of the flattened yarn geometry required by the voxelizer.  It
is designed for memory mapping and repeated production voxelization, not for
reconstructing an editable :class:`CTextile`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .Core import DeleteTextile, GetTextiles, ReadFromXML
    from .gpu_voxelizer import (
        SnapshotBundle,
        extract_snapshot_bundle,
        voxelize_snapshot_bundle_data,
        voxelize_textile_data,
    )
except ImportError:
    from TexGen.Core import DeleteTextile, GetTextiles, ReadFromXML
    from TexGen.gpu_voxelizer import (
        SnapshotBundle,
        extract_snapshot_bundle,
        voxelize_snapshot_bundle_data,
        voxelize_textile_data,
    )


_PTGB_MAGIC = b"PTGB\r\n\x1a\n"
_PTGB_PREFIX = struct.Struct("<8sHHQ")
_PTGB_VERSION = (1, 0)
_PTGB_ALIGNMENT = 64
_PTGB_FORMAT = "pytexgen.prepared_geometry"
_PTGB_ARRAY_NAMES = (
    "positions",
    "tangents",
    "ups",
    "sides",
    "node_offsets",
    "sections",
    "section_offsets",
    "translations",
    "translation_offsets",
    "aabb",
)
_BATCH_FIELDS = {
    "yarn_id",
    "material_id",
    "orientation",
    "stiffness_c21",
}
_TG3_REGISTRY_LOCK = threading.Lock()


class PTGBFormatError(ValueError):
    """Raised when a prepared-geometry file is malformed or incompatible."""


class BatchVoxelizationError(RuntimeError):
    """Raised for a failed item when ``on_error="raise"`` is selected."""

    def __init__(self, source: str, stage: str, message: str):
        super().__init__(f"{source}: {stage}: {message}")
        self.source = source
        self.stage = stage


@dataclass(frozen=True)
class MaterialSpec:
    """Material data used to build physical IDs and rotated C21 fields."""

    matrix_c21: Any
    default_yarn_c21: Any
    yarn_c21_by_id: Mapping[int, Any] = field(default_factory=dict)
    unit: str = "Pa"

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("MaterialSpec.unit must be a non-empty string")


@dataclass(frozen=True)
class BatchItemResult:
    """Outcome and timings for one input geometry."""

    source: str
    output: Optional[str]
    success: bool
    stage: str
    error: Optional[str]
    timings: Mapping[str, float]
    voxels: int


@dataclass(frozen=True)
class BatchVoxelizationReport:
    """Lightweight report returned by :func:`voxelize_files_batch`."""

    items: Tuple[BatchItemResult, ...]
    total_seconds: float
    resolution_xyz: Tuple[int, int, int]
    fields: Tuple[str, ...]
    device: str
    dtype: str
    classification: str = "tensor"

    @property
    def succeeded(self) -> int:
        return sum(item.success for item in self.items)

    @property
    def failed(self) -> int:
        return len(self.items) - self.succeeded

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible report."""
        result = asdict(self)
        result["succeeded"] = self.succeeded
        result["failed"] = self.failed
        return result


def _align(value: int, alignment: int = _PTGB_ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _canonical_array(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("PTGB does not support object arrays")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and np.little_endian is False
    ):
        array = array.astype(array.dtype.newbyteorder("<"), copy=True)
    elif array.dtype.byteorder == "=" and array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _bundle_arrays(bundle: SnapshotBundle) -> Dict[str, np.ndarray]:
    return {
        name: _canonical_array(getattr(bundle, name))
        for name in _PTGB_ARRAY_NAMES
    }


def _write_padding(stream, count: int) -> None:
    remaining = int(count)
    zeroes = b"\0" * min(_PTGB_ALIGNMENT, max(remaining, 1))
    while remaining:
        chunk = zeroes[:remaining]
        stream.write(chunk)
        remaining -= len(chunk)


def save_prepared_geometry(
    bundle: SnapshotBundle,
    path: Union[str, os.PathLike[str]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Write a :class:`SnapshotBundle` as an mmap-friendly PTGB file."""
    if not isinstance(bundle, SnapshotBundle):
        raise TypeError("bundle must be a SnapshotBundle")
    target = Path(path)
    if target.suffix.lower() != ".ptgb":
        raise ValueError("prepared geometry output must use the .ptgb suffix")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays = _bundle_arrays(bundle)
    descriptors: Dict[str, Dict[str, Any]] = {}
    relative_offset = 0
    for name in _PTGB_ARRAY_NAMES:
        array = arrays[name]
        relative_offset = _align(relative_offset)
        descriptors[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": relative_offset,
            "nbytes": int(array.nbytes),
            "sha256": _sha256_array(array),
        }
        relative_offset += int(array.nbytes)

    header = {
        "format": _PTGB_FORMAT,
        "format_version": _PTGB_VERSION[0],
        "alignment": _PTGB_ALIGNMENT,
        "arrays": descriptors,
        "metadata": dict(metadata or {}),
    }
    header_bytes = json.dumps(
        header,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    prefix = _PTGB_PREFIX.pack(
        _PTGB_MAGIC,
        _PTGB_VERSION[0],
        _PTGB_VERSION[1],
        len(header_bytes),
    )
    data_start = _align(len(prefix) + len(header_bytes))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(prefix)
            stream.write(header_bytes)
            _write_padding(stream, data_start - stream.tell())
            for name in _PTGB_ARRAY_NAMES:
                expected = data_start + descriptors[name]["offset"]
                _write_padding(stream, expected - stream.tell())
                stream.write(memoryview(arrays[name]).cast("B"))
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _read_ptgb_header(path: Path) -> Tuple[Dict[str, Any], int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(_PTGB_PREFIX.size)
            if len(prefix) != _PTGB_PREFIX.size:
                raise PTGBFormatError("truncated PTGB prefix")
            magic, major, minor, header_length = _PTGB_PREFIX.unpack(prefix)
            if magic != _PTGB_MAGIC:
                raise PTGBFormatError("invalid PTGB magic")
            if (major, minor) != _PTGB_VERSION:
                raise PTGBFormatError(
                    f"unsupported PTGB version {major}.{minor}"
                )
            if header_length < 2 or header_length > file_size:
                raise PTGBFormatError("invalid PTGB header length")
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise PTGBFormatError("truncated PTGB header")
    except OSError as error:
        raise PTGBFormatError(str(error)) from error
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PTGBFormatError("invalid PTGB JSON header") from error
    if not isinstance(header, dict) or header.get("format") != _PTGB_FORMAT:
        raise PTGBFormatError("unexpected PTGB format identifier")
    if header.get("format_version") != _PTGB_VERSION[0]:
        raise PTGBFormatError("unsupported PTGB format version")
    if header.get("alignment") != _PTGB_ALIGNMENT:
        raise PTGBFormatError("unsupported PTGB alignment")
    data_start = _align(_PTGB_PREFIX.size + int(header_length))
    return header, data_start


def _validated_array_descriptors(
    path: Path,
    header: Mapping[str, Any],
    data_start: int,
) -> Mapping[str, Mapping[str, Any]]:
    raw = header.get("arrays")
    if not isinstance(raw, dict) or set(raw) != set(_PTGB_ARRAY_NAMES):
        raise PTGBFormatError("PTGB array directory is incomplete")
    file_size = path.stat().st_size
    previous_end = 0
    for name in _PTGB_ARRAY_NAMES:
        descriptor = raw.get(name)
        if not isinstance(descriptor, dict):
            raise PTGBFormatError(f"invalid array descriptor {name!r}")
        try:
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(value) for value in descriptor["shape"])
            offset = int(descriptor["offset"])
            nbytes = int(descriptor["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise PTGBFormatError(
                f"invalid array metadata for {name!r}"
            ) from error
        if dtype.hasobject or any(value < 0 for value in shape):
            raise PTGBFormatError(f"unsafe array metadata for {name!r}")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if expected_nbytes != nbytes:
            raise PTGBFormatError(f"array byte count mismatch for {name!r}")
        if offset < previous_end or offset % _PTGB_ALIGNMENT:
            raise PTGBFormatError(f"invalid array offset for {name!r}")
        if data_start + offset + nbytes > file_size:
            raise PTGBFormatError(f"truncated array payload for {name!r}")
        checksum = descriptor.get("sha256")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise PTGBFormatError(f"invalid checksum for {name!r}")
        previous_end = offset + nbytes
    return raw


def load_prepared_geometry(
    path: Union[str, os.PathLike[str]],
    *,
    mmap_mode: Optional[str] = "r",
    verify: str = "header",
) -> SnapshotBundle:
    """Load a PTGB file, memory-mapping its arrays by default.

    ``verify="header"`` validates bounds and schema without reading all array
    bytes.  ``verify="checksum"`` additionally verifies every payload digest.
    """
    source = Path(path)
    if source.suffix.lower() != ".ptgb":
        raise ValueError("prepared geometry input must use the .ptgb suffix")
    if verify not in {"header", "checksum"}:
        raise ValueError('verify must be "header" or "checksum"')
    if mmap_mode not in {None, "r", "r+", "c"}:
        raise ValueError('mmap_mode must be None, "r", "r+", or "c"')
    header, data_start = _read_ptgb_header(source)
    descriptors = _validated_array_descriptors(source, header, data_start)
    arrays: Dict[str, np.ndarray] = {}
    for name in _PTGB_ARRAY_NAMES:
        descriptor = descriptors[name]
        dtype = np.dtype(descriptor["dtype"])
        shape = tuple(int(value) for value in descriptor["shape"])
        absolute_offset = data_start + int(descriptor["offset"])
        if mmap_mode is None:
            with source.open("rb") as stream:
                stream.seek(absolute_offset)
                payload = stream.read(int(descriptor["nbytes"]))
            array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        else:
            array = np.memmap(
                source,
                dtype=dtype,
                mode=mmap_mode,
                offset=absolute_offset,
                shape=shape,
                order="C",
            )
        if verify == "checksum":
            actual = _sha256_array(np.ascontiguousarray(array))
            if actual != descriptor["sha256"]:
                raise PTGBFormatError(f"checksum mismatch for {name!r}")
        arrays[name] = array
    return SnapshotBundle(**arrays)


def _with_tg3_textile(path: Path, textile_name: Optional[str], operation):
    """Run an operation on one TG3 textile while safely owning its registry entry."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"could not parse TG3 file {path}") from error
    file_names = [
        element.get("name")
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "Textile"
        and element.get("name") is not None
    ]
    if not file_names:
        raise ValueError(f"TG3 contains no named textiles: {path}")
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"TG3 contains duplicate textile names: {path}")
    if textile_name is None:
        if len(file_names) != 1:
            raise ValueError(
                "TG3 must contain exactly one textile when textile_name "
                f"is omitted; found {file_names}"
            )
        selected_name = file_names[0]
    else:
        selected_name = textile_name
        if selected_name not in file_names:
            raise KeyError(
                f"textile {selected_name!r} is not present in {path}"
            )

    # TexGen XML loading uses a process-global registry. Serialize this short
    # operation, reject name collisions, and remove only entries from this TG3.
    with _TG3_REGISTRY_LOCK:
        existing_names = set(GetTextiles())
        conflicts = sorted(existing_names.intersection(file_names))
        if conflicts:
            raise RuntimeError(
                "TG3 textile names already exist in TexGen's global registry: "
                f"{conflicts}"
            )
        try:
            if not ReadFromXML(str(path)):
                raise ValueError(f"TexGen could not read {path}")
            textiles = dict(GetTextiles().items())
            missing = sorted(set(file_names) - set(textiles))
            if missing:
                raise ValueError(
                    f"TexGen did not load textiles {missing} from {path}"
                )
            return selected_name, operation(textiles[selected_name])
        finally:
            current_names = set(GetTextiles())
            for loaded_name in file_names:
                if loaded_name in current_names:
                    DeleteTextile(loaded_name)


def _load_tg3_bundle(path: Path, textile_name: Optional[str]) -> SnapshotBundle:
    """Load one TG3 textile without deleting unrelated registry entries."""

    def detach(textile):
        bundle = extract_snapshot_bundle(textile)
        return SnapshotBundle(
            **{
                name: np.array(getattr(bundle, name), copy=True)
                for name in _PTGB_ARRAY_NAMES
            }
        )

    selected_name, detached = _with_tg3_textile(path, textile_name, detach)
    setattr(detached, "_textile_name", selected_name)
    return detached


def prepare_geometry(
    source: Union[str, os.PathLike[str]],
    output: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    textile_name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Convert one TG3 file into a PTGB voxelization cache."""
    source_path = Path(source)
    if source_path.suffix.lower() != ".tg3":
        raise ValueError("prepare_geometry source must use the .tg3 suffix")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path = (
        source_path.with_suffix(".ptgb") if output is None else Path(output)
    )
    bundle = _load_tg3_bundle(source_path, textile_name)
    return save_prepared_geometry(
        bundle,
        output_path,
        metadata={
            "source_name": source_path.name,
            "textile_name": getattr(bundle, "_textile_name", textile_name),
        },
        overwrite=overwrite,
    )


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch") and hasattr(value, "device")


def _to_numpy(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _zeros(shape: Tuple[int, ...], reference: Any, *, integer: bool = False):
    if _is_torch_tensor(reference):
        import torch

        dtype = torch.int32 if integer else reference.dtype
        return torch.zeros(shape, dtype=dtype, device=reference.device)
    dtype = np.int32 if integer else np.asarray(reference).dtype
    return np.zeros(shape, dtype=dtype)


def _dense_orientation(bundle, grid_shape: Tuple[int, int, int]):
    total = math.prod(grid_shape)
    reference = bundle.orientation1
    dense = _zeros((total, 3, 3), reference)
    if _is_torch_tensor(reference):
        import torch

        indices = bundle.voxel_indices.to(dtype=torch.long)
        third = torch.linalg.cross(
            bundle.orientation1,
            bundle.orientation2,
            dim=1,
        )
    else:
        indices = np.asarray(bundle.voxel_indices, dtype=np.int64)
        third = np.cross(bundle.orientation1, bundle.orientation2)
    dense[indices, 0, :] = bundle.orientation1
    dense[indices, 1, :] = bundle.orientation2
    dense[indices, 2, :] = third
    return dense.reshape(grid_shape + (3, 3))


def _dense_material_ids(field) -> Any:
    total = math.prod(field.grid_shape)
    dense = _zeros((total,), field.material_ids, integer=True)
    if _is_torch_tensor(field.voxel_indices):
        import torch

        indices = field.voxel_indices.to(dtype=torch.long)
    else:
        indices = np.asarray(field.voxel_indices, dtype=np.int64)
    dense[indices] = field.material_ids
    return dense.reshape(field.grid_shape)


def _field_output_bytes(
    fields: Iterable[str],
    resolution_xyz: Tuple[int, int, int],
    dtype: str,
) -> int:
    components = 0
    if "yarn_id" in fields:
        components += 4
    if "material_id" in fields:
        components += 4
    itemsize = np.dtype(dtype).itemsize
    if "orientation" in fields:
        components += 9 * itemsize
    if "stiffness_c21" in fields:
        components += 21 * itemsize
    return math.prod(resolution_xyz) * components


def _write_voxel_output(
    output_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> Tuple[str, float]:
    start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        )
    )
    try:
        for name, array in arrays.items():
            np.save(
                temporary / f"{name}.npy",
                np.ascontiguousarray(array),
                allow_pickle=False,
            )
        (temporary / "metadata.json").write_text(
            json.dumps(
                metadata,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(output_path)
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(output_path), time.perf_counter() - start


def _load_geometry_source(path: Path, verify_ptgb: str) -> SnapshotBundle:
    suffix = path.suffix.lower()
    if suffix == ".ptgb":
        return load_prepared_geometry(
            path,
            # Copy-on-write keeps the PTGB immutable while presenting writable
            # NumPy views, which avoids torch.from_numpy's read-only warning.
            mmap_mode="c",
            verify=verify_ptgb,
        )
    if suffix == ".tg3":
        return _load_tg3_bundle(path, textile_name=None)
    raise ValueError(f"unsupported geometry input suffix {path.suffix!r}")


def voxelize_files_batch(
    inputs: Sequence[Union[str, os.PathLike[str]]],
    *,
    resolution: Sequence[int],
    output_dir: Union[str, os.PathLike[str]],
    fields: Sequence[str] = (
        "material_id",
        "orientation",
        "stiffness_c21",
    ),
    materials: Optional[MaterialSpec] = None,
    device: str = "cuda",
    dtype: str = "float32",
    batch_size: Union[int, str] = "auto",
    memory_budget_bytes: Optional[int] = None,
    chunk_voxels: int = 65536,
    overwrite: bool = False,
    on_error: str = "raise",
    verify_ptgb: str = "header",
    classification: str = "tensor",
    workers: Optional[int] = None,
) -> BatchVoxelizationReport:
    """Stream a collection of TG3/PTGB files through the voxelizer.

    GPU computation remains one geometry at a time in format version 1, while
    bounded background writers overlap raw ``.npy`` output with the following
    geometry.  ``batch_size`` limits the number of in-flight write jobs and
    therefore bounds host-memory pressure.
    """
    sources = tuple(Path(path) for path in inputs)
    if not sources:
        raise ValueError("inputs must contain at least one geometry file")
    resolution_xyz = tuple(int(value) for value in resolution)
    if len(resolution_xyz) != 3 or min(resolution_xyz) < 1:
        raise ValueError("resolution must contain three positive XYZ counts")
    if dtype not in {"float32", "float64"}:
        raise ValueError('dtype must be "float32" or "float64"')
    requested_fields = tuple(dict.fromkeys(str(value) for value in fields))
    unknown = sorted(set(requested_fields) - _BATCH_FIELDS)
    if unknown or not requested_fields:
        raise ValueError(f"unsupported output fields: {unknown}")
    if "stiffness_c21" in requested_fields and materials is None:
        raise ValueError("materials are required for stiffness_c21 output")
    if on_error not in {"raise", "collect"}:
        raise ValueError('on_error must be "raise" or "collect"')
    if verify_ptgb not in {"header", "checksum"}:
        raise ValueError('verify_ptgb must be "header" or "checksum"')
    classification = str(classification).lower()
    if classification not in {"tensor", "exact", "numpy_exact"}:
        raise ValueError(
            'classification must be "tensor", "exact", or "numpy_exact"'
        )
    if workers is not None and int(workers) < 1:
        raise ValueError("workers must be >= 1 or None")
    if batch_size == "auto":
        max_in_flight = 2
    else:
        max_in_flight = int(batch_size)
        if max_in_flight < 1:
            raise ValueError("batch_size must be positive or 'auto'")
    if memory_budget_bytes is not None:
        if int(memory_budget_bytes) <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        estimated = _field_output_bytes(
            requested_fields,
            resolution_xyz,
            dtype,
        )
        required = estimated * (max_in_flight + 1)
        if required > int(memory_budget_bytes):
            raise MemoryError(
                "dense outputs and their working-space allowance are "
                f"estimated to require {required} bytes, exceeding "
                f"memory_budget_bytes={int(memory_budget_bytes)}"
            )

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in {".tg3", ".ptgb"}:
            raise ValueError(f"unsupported geometry input {source}")
        if classification in {"exact", "numpy_exact"} and source.suffix.lower() != ".tg3":
            raise ValueError(
                f"classification={classification!r} requires TG3 inputs; PTGB "
                "v1 stores only the approximate tensor-classifier geometry"
            )
    names = [source.stem for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("input stems must be unique within one batch")
    root = Path(output_dir)
    targets = [root / name for name in names]
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(existing[0])

    backend = "numpy" if str(device).lower() == "cpu" else "torch"
    nx, ny, nz = resolution_xyz
    grid_shape = (nz, ny, nx)
    need_orientation = bool(
        {"orientation", "stiffness_c21"} & set(requested_fields)
    )
    started = time.perf_counter()
    reports: list[Optional[BatchItemResult]] = [None] * len(sources)
    pending: deque[
        Tuple[int, Future[Tuple[str, float]], BatchItemResult]
    ] = deque()

    def finish_one() -> None:
        index, future, base = pending.popleft()
        try:
            output, write_seconds = future.result()
            timings = dict(base.timings)
            timings["write"] = write_seconds
            reports[index] = replace(
                base,
                output=output,
                success=True,
                stage="complete",
                timings=timings,
            )
        except Exception as error:
            reports[index] = replace(
                base,
                success=False,
                stage="write",
                error=str(error),
            )
            if on_error == "raise":
                raise BatchVoxelizationError(
                    base.source,
                    "write",
                    str(error),
                ) from error

    with ThreadPoolExecutor(
        max_workers=min(max_in_flight, 4),
        thread_name_prefix="pytexgen-output",
    ) as writers:
        for index, (source, target) in enumerate(zip(sources, targets)):
            item_started = time.perf_counter()
            try:
                load_started = time.perf_counter()
                if classification in {"exact", "numpy_exact"}:
                    def classify_live_textile(textile):
                        loaded_seconds = time.perf_counter() - load_started
                        voxel_started = time.perf_counter()
                        live_data = voxelize_textile_data(
                            textile,
                            nx=nx,
                            ny=ny,
                            nz=nz,
                            backend=backend,
                            device=None if backend == "numpy" else device,
                            dtype=dtype,
                            chunk_voxels=chunk_voxels,
                            workers=workers,
                            verbose=False,
                            include_orientations=need_orientation,
                            orientation_storage="sparse",
                            output="backend",
                            classification=classification,
                        )
                        return (
                            live_data,
                            loaded_seconds,
                            time.perf_counter() - voxel_started,
                        )

                    _selected_name, live_result = _with_tg3_textile(
                        source, None, classify_live_textile
                    )
                    data, load_seconds, voxel_seconds = live_result
                else:
                    bundle = _load_geometry_source(source, verify_ptgb)
                    load_seconds = time.perf_counter() - load_started

                    voxel_started = time.perf_counter()
                    data = voxelize_snapshot_bundle_data(
                        bundle,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        backend=backend,
                        device=None if backend == "numpy" else device,
                        dtype=dtype,
                        chunk_voxels=chunk_voxels,
                        workers=workers,
                        verbose=False,
                        include_orientations=need_orientation,
                        orientation_storage="sparse",
                        output="backend",
                    )
                    voxel_seconds = time.perf_counter() - voxel_started
                sparse_orientation = data.sparse_orientation

                material_field = None
                stiffness_seconds = 0.0
                if materials is not None and (
                    "material_id" in requested_fields
                    or "stiffness_c21" in requested_fields
                ):
                    try:
                        from .material_fields import build_stiffness_field
                    except ImportError:
                        from TexGen.material_fields import build_stiffness_field

                    stiffness_started = time.perf_counter()
                    material_field = build_stiffness_field(
                        data,
                        matrix_stiffness=materials.matrix_c21,
                        default_yarn_stiffness=materials.default_yarn_c21,
                        yarn_stiffness_by_id=dict(materials.yarn_c21_by_id),
                        output="sparse",
                        chunk_voxels=chunk_voxels,
                        unit=materials.unit,
                    )
                    if _is_torch_tensor(material_field.yarn_c21):
                        import torch

                        field_device = material_field.yarn_c21.device
                        if field_device.type == "cuda":
                            torch.cuda.synchronize(field_device)
                        elif field_device.type == "mps":
                            torch.mps.synchronize()
                    stiffness_seconds = time.perf_counter() - stiffness_started

                arrays: Dict[str, np.ndarray] = {}
                if "yarn_id" in requested_fields:
                    arrays["yarn_id"] = _to_numpy(
                        data.yarn_id.reshape(grid_shape)
                    )
                if "material_id" in requested_fields:
                    physical_ids = (
                        _dense_material_ids(material_field)
                        if material_field is not None
                        else data.material_id()
                    )
                    arrays["material_id"] = _to_numpy(
                        physical_ids.reshape(grid_shape)
                    )
                dense_started = time.perf_counter()
                if "orientation" in requested_fields:
                    if sparse_orientation is None:
                        raise RuntimeError("orientation data was not generated")
                    arrays["orientation"] = _to_numpy(
                        _dense_orientation(sparse_orientation, grid_shape)
                    )
                if "stiffness_c21" in requested_fields:
                    if material_field is None:
                        raise RuntimeError("stiffness data was not generated")
                    arrays["stiffness_c21"] = _to_numpy(
                        material_field.to_dense_c21()
                    )
                dense_seconds = time.perf_counter() - dense_started
                timings = {
                    "load": load_seconds,
                    "voxelize": voxel_seconds,
                    "stiffness": stiffness_seconds,
                    "densify_and_transfer": dense_seconds,
                    "total_before_write": time.perf_counter() - item_started,
                }
                metadata = {
                    "format": "pytexgen.batch_voxel_output",
                    "format_version": 1,
                    "source": source.name,
                    "source_format": source.suffix.lower().lstrip("."),
                    "resolution_xyz": list(resolution_xyz),
                    "grid_shape_zyx": list(grid_shape),
                    "dtype": dtype,
                    "device": str(device),
                    "classification": classification,
                    "fields": list(requested_fields),
                    "array_files": {
                        name: f"{name}.npy" for name in arrays
                    },
                    "timings_seconds": timings,
                    "order": "ix + iy*nx + iz*nx*ny",
                }
                base = BatchItemResult(
                    source=str(source),
                    output=None,
                    success=False,
                    stage="write",
                    error=None,
                    timings=timings,
                    voxels=nx * ny * nz,
                )
                future = writers.submit(
                    _write_voxel_output,
                    target,
                    arrays,
                    metadata,
                    overwrite,
                )
                pending.append((index, future, base))
                if len(pending) >= max_in_flight:
                    finish_one()
            except BatchVoxelizationError:
                raise
            except Exception as error:
                reports[index] = BatchItemResult(
                    source=str(source),
                    output=None,
                    success=False,
                    stage="compute",
                    error=str(error),
                    timings={
                        "total_before_failure": (
                            time.perf_counter() - item_started
                        )
                    },
                    voxels=nx * ny * nz,
                )
                if on_error == "raise":
                    raise BatchVoxelizationError(
                        str(source),
                        "compute",
                        str(error),
                    ) from error
        while pending:
            finish_one()

    final_items = tuple(
        item
        for item in reports
        if item is not None
    )
    return BatchVoxelizationReport(
        items=final_items,
        total_seconds=time.perf_counter() - started,
        resolution_xyz=resolution_xyz,
        fields=requested_fields,
        device=str(device),
        dtype=dtype,
        classification=classification,
    )


__all__ = [
    "BatchItemResult",
    "BatchVoxelizationError",
    "BatchVoxelizationReport",
    "MaterialSpec",
    "PTGBFormatError",
    "load_prepared_geometry",
    "prepare_geometry",
    "save_prepared_geometry",
    "voxelize_files_batch",
]
