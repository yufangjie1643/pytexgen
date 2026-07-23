"""Versioned persistence for :class:`SimulationSample`.

Directory storage uses memory-mappable ``.npy`` arrays. Archive storage uses
one ``.npz`` member per canonical array. Orientation and stiffness topology
names always alias the same stored index and yarn-ID arrays.
"""

from __future__ import annotations

import json
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

try:
    from .gpu_voxelizer import VoxelGridData
    from .material_fields import SparseOrientationField, SparseStiffnessField
    from .simulation_sample import MaterialTable, SimulationSample
except ImportError:  # pragma: no cover - legacy TexGen package name
    from TexGen.gpu_voxelizer import VoxelGridData
    from TexGen.material_fields import (
        SparseOrientationField,
        SparseStiffnessField,
    )
    from TexGen.simulation_sample import MaterialTable, SimulationSample


_SCHEMA = "pytexgen.simulation_sample"
_VERSION = 1


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _as_numpy(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _thaw_json(value: Any):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _package_version() -> str:
    try:
        return importlib_metadata.version("pytexgen")
    except importlib_metadata.PackageNotFoundError:
        return "source"


def _canonical_arrays(sample: SimulationSample) -> Dict[str, np.ndarray]:
    arrays = {
        "voxel_yarn_id": _as_numpy(sample.voxels.yarn_id),
        "voxel_aabb": _as_numpy(sample.voxels.aabb),
        "material_ids": _as_numpy(sample.materials.material_ids),
        "material_c21": _as_numpy(sample.materials.c21),
    }
    optional_voxel = {
        "voxel_centers": sample.voxels.centers,
        "voxel_orientation1": sample.voxels.orientation1,
        "voxel_orientation2": sample.voxels.orientation2,
    }
    for name, value in optional_voxel.items():
        if value is not None:
            arrays[name] = _as_numpy(value)

    topology = sample.orientation
    if topology is None and sample.stiffness is not None:
        topology = sample.stiffness
    if topology is not None:
        arrays["sparse_voxel_indices"] = _as_numpy(
            topology.voxel_indices
        )
        arrays["sparse_yarn_ids"] = _as_numpy(topology.yarn_ids)

    if sample.orientation is not None:
        arrays["orientation_primary"] = _as_numpy(
            sample.orientation.orientation1
        )
        arrays["orientation_secondary"] = _as_numpy(
            sample.orientation.orientation2
        )
    if sample.stiffness is not None:
        arrays["stiffness_matrix_c21"] = _as_numpy(
            sample.stiffness.matrix_c21
        )
        arrays["stiffness_material_ids"] = _as_numpy(
            sample.stiffness.material_ids
        )
        arrays["stiffness_yarn_c21"] = _as_numpy(
            sample.stiffness.yarn_c21
        )
    return arrays


def _field_aliases(sample: SimulationSample) -> Dict[str, str]:
    aliases = {
        "voxel.yarn_id": "voxel_yarn_id",
        "material.ids": "material_ids",
        "material.c21": "material_c21",
    }
    if sample.orientation is not None:
        aliases.update(
            {
                "orientation.voxel_indices": "sparse_voxel_indices",
                "orientation.yarn_ids": "sparse_yarn_ids",
                "orientation.primary": "orientation_primary",
                "orientation.secondary": "orientation_secondary",
            }
        )
    if sample.stiffness is not None:
        aliases.update(
            {
                "stiffness.matrix_c21": "stiffness_matrix_c21",
                "stiffness.voxel_indices": "sparse_voxel_indices",
                "stiffness.yarn_ids": "sparse_yarn_ids",
                "stiffness.material_ids": "stiffness_material_ids",
                "stiffness.yarn_c21": "stiffness_yarn_c21",
            }
        )
    return aliases


def _build_manifest(
    sample: SimulationSample,
    arrays: Mapping[str, np.ndarray],
    *,
    archive: bool,
) -> Dict[str, Any]:
    array_entries = {}
    for name, value in arrays.items():
        array_entries[name] = {
            "location": (
                name if archive else f"arrays/{name}.npy"
            ),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    metadata = _thaw_json(sample.metadata)
    return {
        "schema": _SCHEMA,
        "version": _VERSION,
        "arrays": array_entries,
        "fields": _field_aliases(sample),
        "grid": {
            "resolution": list(sample.voxels.resolution),
            "shape": list(sample.voxels.shape),
            "order": sample.voxels.order,
        },
        "voxel": {
            "backend": sample.voxels.backend,
            "source_device": sample.voxels.device,
            "source_storage": sample.voxels.storage,
            "workers": int(sample.voxels.workers),
            "dtype": sample.voxels.dtype,
            "timings": _thaw_json(sample.voxels.timings),
            "aabb_pruning": bool(sample.voxels.aabb_pruning),
        },
        "materials": {
            "unit": sample.materials.unit,
            "names": (
                None
                if sample.materials.names is None
                else list(sample.materials.names)
            ),
        },
        "metadata": metadata,
        "generation": metadata.get("generation", {}),
        "provenance": {
            "pytexgen_version": _package_version(),
            "git_commit": os.environ.get("PYTEXGEN_GIT_COMMIT"),
        },
    }


def save_simulation_sample(
    path,
    sample: SimulationSample,
    *,
    compressed: bool = True,
) -> None:
    """Save one sample without duplicating shared sparse topology arrays."""
    if not isinstance(sample, SimulationSample):
        raise TypeError("sample must be a SimulationSample")
    out_path = Path(path)
    if out_path.exists():
        raise FileExistsError(f"target already exists: {out_path}")

    archive = out_path.suffix.lower() == ".npz"
    arrays = _canonical_arrays(sample)
    manifest = _build_manifest(sample, arrays, archive=archive)
    manifest_json = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if archive:
        payload = dict(arrays)
        payload["_manifest_json"] = np.asarray(manifest_json)
        saver = np.savez_compressed if compressed else np.savez
        saver(out_path, **payload)
        return

    out_path.mkdir(parents=True)
    array_dir = out_path / "arrays"
    array_dir.mkdir()
    for name, value in arrays.items():
        np.save(array_dir / f"{name}.npy", value, allow_pickle=False)
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_manifest_and_arrays(
    path: Path,
    *,
    mmap_mode: Optional[str],
):
    archive = path.suffix.lower() == ".npz"
    if archive:
        if mmap_mode is not None:
            raise ValueError("mmap_mode is supported only for directory storage")
        if not path.is_file():
            raise ValueError(f"missing simulation sample archive: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if "_manifest_json" not in payload:
                raise ValueError("missing _manifest_json in sample archive")
            try:
                manifest = json.loads(str(payload["_manifest_json"].item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("invalid simulation sample manifest") from exc
            arrays = _load_archive_arrays(payload, manifest)
        return manifest, arrays

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing simulation sample manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid simulation sample manifest") from exc
    arrays = _load_directory_arrays(path, manifest, mmap_mode=mmap_mode)
    return manifest, arrays


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != _SCHEMA:
        raise ValueError(
            f"unsupported simulation sample schema {manifest.get('schema')!r}"
        )
    if manifest.get("version") != _VERSION:
        raise ValueError(
            f"unsupported simulation sample version {manifest.get('version')!r}"
        )
    if not isinstance(manifest.get("arrays"), dict):
        raise ValueError("manifest arrays must be a mapping")
    if not isinstance(manifest.get("fields"), dict):
        raise ValueError("manifest fields must be a mapping")


def _validate_loaded_array(
    name: str,
    value: np.ndarray,
    entry: Mapping[str, Any],
) -> None:
    if str(value.dtype) != entry.get("dtype"):
        raise ValueError(
            f"dtype mismatch for {name}: expected {entry.get('dtype')}, "
            f"got {value.dtype}"
        )
    if list(value.shape) != entry.get("shape"):
        raise ValueError(
            f"shape mismatch for {name}: expected {entry.get('shape')}, "
            f"got {list(value.shape)}"
        )


def _load_archive_arrays(payload, manifest):
    _validate_manifest_header(manifest)
    arrays = {}
    for name, entry in manifest["arrays"].items():
        member = entry.get("location")
        if member not in payload:
            raise ValueError(f"missing array {member!r} for {name}")
        value = np.array(payload[member], copy=True)
        _validate_loaded_array(name, value, entry)
        arrays[name] = value
    return arrays


def _load_directory_arrays(path: Path, manifest, *, mmap_mode):
    _validate_manifest_header(manifest)
    arrays = {}
    root = path.resolve()
    for name, entry in manifest["arrays"].items():
        location = entry.get("location")
        if not isinstance(location, str):
            raise ValueError(f"missing array location for {name}")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid array location for {name}: {location}")
        array_path = (path / relative).resolve()
        if root not in array_path.parents:
            raise ValueError(f"invalid array location for {name}: {location}")
        if not array_path.is_file():
            raise ValueError(f"missing array {location!r} for {name}")
        value = np.load(
            array_path,
            allow_pickle=False,
            mmap_mode=mmap_mode,
        )
        _validate_loaded_array(name, value, entry)
        arrays[name] = value
    return arrays


def _require_arrays(arrays: Mapping[str, np.ndarray], names) -> None:
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing required canonical array {name}")


def _validate_topology_aliases(manifest: Mapping[str, Any]) -> None:
    fields = manifest["fields"]
    orientation_index = fields.get("orientation.voxel_indices")
    stiffness_index = fields.get("stiffness.voxel_indices")
    if (
        orientation_index is not None
        and stiffness_index is not None
        and orientation_index != stiffness_index
    ):
        raise ValueError("sparse voxel index aliases must reference one array")
    orientation_yarn = fields.get("orientation.yarn_ids")
    stiffness_yarn = fields.get("stiffness.yarn_ids")
    if (
        orientation_yarn is not None
        and stiffness_yarn is not None
        and orientation_yarn != stiffness_yarn
    ):
        raise ValueError("sparse yarn ID aliases must reference one array")


def _construct_sample(manifest, arrays) -> SimulationSample:
    _require_arrays(
        arrays,
        ("voxel_yarn_id", "voxel_aabb", "material_ids", "material_c21"),
    )
    _validate_topology_aliases(manifest)
    grid = manifest.get("grid", {})
    resolution = tuple(int(value) for value in grid.get("resolution", ()))
    grid_shape = tuple(int(value) for value in grid.get("shape", ()))
    if len(resolution) != 3 or len(grid_shape) != 3:
        raise ValueError("manifest grid resolution and shape must have length 3")
    expected_shape = (resolution[2], resolution[1], resolution[0])
    if grid_shape != expected_shape:
        raise ValueError("manifest grid shape does not match resolution")
    order = str(grid.get("order", ""))

    has_orientation = {
        "sparse_voxel_indices",
        "sparse_yarn_ids",
        "orientation_primary",
        "orientation_secondary",
    }.issubset(arrays)
    orientation = None
    if has_orientation:
        orientation = SparseOrientationField(
            voxel_indices=arrays["sparse_voxel_indices"],
            yarn_ids=arrays["sparse_yarn_ids"],
            orientation1=arrays["orientation_primary"],
            orientation2=arrays["orientation_secondary"],
            grid_shape=grid_shape,
            order=order,
        )

    has_stiffness = {
        "sparse_voxel_indices",
        "sparse_yarn_ids",
        "stiffness_matrix_c21",
        "stiffness_material_ids",
        "stiffness_yarn_c21",
    }.issubset(arrays)
    stiffness = None
    if has_stiffness:
        stiffness = SparseStiffnessField(
            matrix_c21=arrays["stiffness_matrix_c21"],
            voxel_indices=arrays["sparse_voxel_indices"],
            yarn_ids=arrays["sparse_yarn_ids"],
            material_ids=arrays["stiffness_material_ids"],
            yarn_c21=arrays["stiffness_yarn_c21"],
            grid_shape=grid_shape,
            unit=manifest["materials"]["unit"],
            order=order,
        )

    voxel_meta = manifest.get("voxel", {})
    voxels = VoxelGridData(
        yarn_id=arrays["voxel_yarn_id"],
        aabb=arrays["voxel_aabb"],
        resolution=resolution,
        backend=str(voxel_meta.get("backend", "numpy")),
        device="cpu",
        workers=int(voxel_meta.get("workers", 1)),
        dtype=str(voxel_meta.get("dtype", arrays["voxel_aabb"].dtype)),
        timings=dict(voxel_meta.get("timings", {})),
        centers=arrays.get("voxel_centers"),
        orientation1=arrays.get("voxel_orientation1"),
        orientation2=arrays.get("voxel_orientation2"),
        sparse_orientation=orientation,
        aabb_pruning=bool(voxel_meta.get("aabb_pruning", True)),
        storage="numpy",
        order=order,
    )
    material_meta = manifest.get("materials", {})
    materials = MaterialTable(
        c21=arrays["material_c21"],
        material_ids=arrays["material_ids"],
        unit=material_meta.get("unit"),
        names=material_meta.get("names"),
    )
    return SimulationSample(
        voxels=voxels,
        orientation=orientation,
        stiffness=stiffness,
        materials=materials,
        metadata=manifest.get("metadata", {}),
    )


def load_simulation_sample(
    path,
    *,
    output: str = "numpy",
    device: Optional[str] = None,
    mmap_mode: Optional[str] = None,
) -> SimulationSample:
    """Load a version-1 sample as NumPy/memmap or explicitly as Torch."""
    output_normalized = str(output).lower()
    if output_normalized not in {"numpy", "torch"}:
        raise ValueError('output must be "numpy" or "torch"')
    if output_normalized == "torch" and mmap_mode is not None:
        raise ValueError("mmap_mode cannot be combined with Torch output")
    manifest, arrays = _read_manifest_and_arrays(
        Path(path),
        mmap_mode=mmap_mode,
    )
    sample = _construct_sample(manifest, arrays)
    if output_normalized == "torch":
        return sample.to("torch", device=device)
    if device not in {None, "cpu"}:
        raise ValueError("NumPy output is available only on the CPU")
    return sample


__all__ = ["save_simulation_sample", "load_simulation_sample"]
