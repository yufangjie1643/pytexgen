"""Correctness-gated TexGen CPU versus GPU material-field benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ACCEPTANCE_CASES = ("plain_2x2", "multi_yarn_8x8")
ACCEPTANCE_RESOLUTIONS = (128, 256)
ACCEPTANCE_MODES = ("compute", "practical")


def summarize_timings(
    values: Sequence[float], *, cpu_median: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """Return stable timing statistics and an optional CPU/GPU speedup."""
    if not values:
        raise ValueError("at least one timing value is required")
    timings = np.asarray(values, dtype=np.float64)
    if not np.isfinite(timings).all() or bool((timings < 0.0).any()):
        raise ValueError("timings must be finite and non-negative")
    median = float(np.median(timings))
    p90 = float(np.percentile(timings, 90))
    speedup = None
    if cpu_median is not None:
        speedup = (
            float(cpu_median) / median if median > 0.0 else float("inf")
        )
    return {
        "values_s": [float(value) for value in timings],
        "median_s": median,
        "p90_s": p90,
        "speedup": speedup,
    }


def _record_correct(record: Dict[str, Any]) -> bool:
    dtype = str(record.get("dtype", "float32"))
    stiffness_limit = 1e-10 if dtype == "float64" else 5e-5
    return bool(
        record.get("correctness", False)
        and float(record.get("occupancy_mismatch_fraction", math.inf))
        <= 0.005
        and float(record.get("yarn_mismatch_fraction", math.inf)) <= 0.005
        and float(record.get("minimum_axis_dot", -math.inf)) >= 0.999
        and float(record.get("stiffness_relative_error", math.inf))
        <= stiffness_limit
    )


def evaluate_acceptance(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate correctness and the 5x gate for all large reference cases."""
    by_key = {
        (
            str(record.get("case")),
            int(record.get("resolution", -1)),
            str(record.get("mode")),
        ): record
        for record in records
    }
    required = {
        (case, resolution, mode)
        for case in ACCEPTANCE_CASES
        for resolution in ACCEPTANCE_RESOLUTIONS
        for mode in ACCEPTANCE_MODES
    }
    large_present = {
        key for key in by_key if key[1] in ACCEPTANCE_RESOLUTIONS
    }
    if not large_present:
        failures = [
            "correctness failed for smoke record "
            f"{record.get('case')}/{record.get('resolution')}/"
            f"{record.get('mode')}"
            for record in records
            if not _record_correct(record)
        ]
        return {
            "applicable": False,
            "passed": not failures,
            "required_speedup": None,
            "failures": failures,
        }

    failures = []
    for key in sorted(required - set(by_key)):
        failures.append(f"missing required record {key}")
    for key in sorted(required & set(by_key)):
        record = by_key[key]
        if not _record_correct(record):
            failures.append(f"correctness failed for {key}")
        speedup = record.get("speedup")
        if speedup is None or float(speedup) < 5.0:
            failures.append(f"speedup below 5x for {key}: {speedup}")
    return {
        "applicable": True,
        "passed": not failures,
        "required_speedup": 5.0,
        "failures": failures,
    }


def _synchronize_torch(torch_mod: Any, device: str) -> None:
    device_type = torch_mod.device(device).type
    if device_type == "cuda":
        torch_mod.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch_mod, "mps"):
        torch_mod.mps.synchronize()


def timed_cuda(
    fn: Any,
    repeat: int,
    warmup: int,
    *,
    torch_mod: Any,
    device: str,
) -> Tuple[Any, List[float], int, int]:
    """Time a callable with explicit accelerator synchronization."""
    for _ in range(warmup):
        fn()
    _synchronize_torch(torch_mod, device)
    values = []
    result = None
    peak_allocated = 0
    peak_reserved = 0
    for _ in range(repeat):
        if torch_mod.device(device).type == "cuda":
            torch_mod.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = fn()
        _synchronize_torch(torch_mod, device)
        values.append(time.perf_counter() - start)
        if torch_mod.device(device).type == "cuda":
            peak_allocated = max(
                peak_allocated,
                int(torch_mod.cuda.max_memory_allocated(device)),
            )
            peak_reserved = max(
                peak_reserved,
                int(torch_mod.cuda.max_memory_reserved(device)),
            )
    return result, values, peak_allocated, peak_reserved


def timed_wall(
    fn: Any, repeat: int, warmup: int = 0
) -> Tuple[Any, List[float]]:
    """Time a synchronous CPU or persistence callable."""
    for _ in range(warmup):
        fn()
    result = None
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, values


@dataclass
class CpuReference:
    yarn_id: np.ndarray
    voxel_indices: np.ndarray
    yarn_ids: np.ndarray
    orientation1: np.ndarray
    orientation2: np.ndarray


def build_textile(case: str):
    """Build one deterministic TexGen benchmark textile."""
    from pytexgen import CTextileWeave2D

    if case == "plain_2x2":
        count = 2
    elif case == "multi_yarn_8x8":
        count = 8
    else:
        raise ValueError(f"unknown benchmark case {case!r}")
    textile = CTextileWeave2D(count, count, 1.0, 0.2, False, True)
    for y_index in range(count):
        for x_index in range(count):
            if (x_index + y_index) % 2 == 0:
                textile.SwapPosition(x_index, y_index)
    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(20)
    textile.AssignDefaultDomain()
    return textile


def _chunk_coordinates(
    start: int,
    stop: int,
    resolution: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Iterable[Tuple[float, float, float]]:
    indices = np.arange(start, stop, dtype=np.int64)
    ix = indices % resolution
    iy = (indices // resolution) % resolution
    iz = indices // (resolution * resolution)
    spacing = (hi - lo) / resolution
    x = lo[0] + (ix + 0.5) * spacing[0]
    y = lo[1] + (iy + 0.5) * spacing[1]
    z = lo[2] + (iz + 0.5) * spacing[2]
    return zip(x.tolist(), y.tolist(), z.tolist())


def texgen_point_information_reference(
    textile: Any,
    aabb: np.ndarray,
    resolution: int,
    *,
    chunk_voxels: int = 65536,
    dtype: Any = np.float32,
) -> CpuReference:
    """Evaluate TexGen's native point-information path in bounded chunks."""
    from pytexgen import PointInfoVector, XYZ

    total = resolution ** 3
    yarn_id = np.full(total, -1, dtype=np.int32)
    occupied_indices = []
    occupied_yarn_ids = []
    directions = []
    ups = []
    lo = np.asarray(aabb[0], dtype=np.float64)
    hi = np.asarray(aabb[1], dtype=np.float64)
    for start in range(0, total, chunk_voxels):
        stop = min(start + chunk_voxels, total)
        points = [
            XYZ(x, y, z)
            for x, y, z in _chunk_coordinates(
                start, stop, resolution, lo, hi
            )
        ]
        information = PointInfoVector()
        textile.GetPointInformation(points, information)
        for offset, info in enumerate(information):
            yarn_index = int(info.iYarnIndex)
            flat_index = start + offset
            yarn_id[flat_index] = yarn_index
            if yarn_index >= 0:
                occupied_indices.append(flat_index)
                occupied_yarn_ids.append(yarn_index)
                direction = info.Orientation
                up = info.Up
                directions.append((direction.x, direction.y, direction.z))
                ups.append((up.x, up.y, up.z))
    return CpuReference(
        yarn_id=yarn_id,
        voxel_indices=np.asarray(occupied_indices, dtype=np.int64),
        yarn_ids=np.asarray(occupied_yarn_ids, dtype=np.int32),
        orientation1=np.asarray(directions, dtype=dtype).reshape(-1, 3),
        orientation2=np.asarray(ups, dtype=dtype).reshape(-1, 3),
    )


def _parse_texgen_export(
    inp_path: Path,
    resolution: int,
    *,
    dtype: Any,
) -> CpuReference:
    total = resolution ** 3
    yarn_id = np.full(total, -1, dtype=np.int32)
    eld_path = inp_path.with_suffix(".eld")
    with eld_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                element = int(parts[0]) - 1
                yarn_index = int(parts[1])
            except ValueError:
                continue
            if 0 <= element < total:
                yarn_id[element] = yarn_index

    voxel_indices = np.flatnonzero(yarn_id >= 0).astype(np.int64, copy=False)
    yarn_ids = yarn_id[voxel_indices]
    positions = np.full(total, -1, dtype=np.int32)
    positions[voxel_indices] = np.arange(
        len(voxel_indices), dtype=np.int32
    )
    orientation1 = np.zeros((len(voxel_indices), 3), dtype=dtype)
    orientation2 = np.zeros((len(voxel_indices), 3), dtype=dtype)
    ori_path = inp_path.with_suffix(".ori")
    with ori_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                element = int(parts[0]) - 1
                direction = np.asarray(parts[1:4], dtype=dtype)
                perpendicular = np.asarray(parts[4:7], dtype=dtype)
            except ValueError:
                continue
            if 0 <= element < total:
                sparse_position = int(positions[element])
                if sparse_position >= 0:
                    orientation1[sparse_position] = direction
                    orientation2[sparse_position] = np.cross(
                        perpendicular, direction
                    )
    return CpuReference(
        yarn_id=yarn_id,
        voxel_indices=voxel_indices,
        yarn_ids=yarn_ids,
        orientation1=orientation1,
        orientation2=orientation2,
    )


def texgen_save_reference(
    textile: Any,
    output_dir: Path,
    resolution: int,
    *,
    dtype: Any,
) -> CpuReference:
    """Run the original CPU voxel export and parse its material directions."""
    from pytexgen import CRectangularVoxelMesh

    output_dir.mkdir(parents=True, exist_ok=True)
    inp_path = output_dir / "texgen_cpu.inp"
    voxel_mesh = CRectangularVoxelMesh("CPeriodicBoundaries")
    voxel_mesh.SaveVoxelMesh(
        textile,
        str(inp_path),
        resolution,
        resolution,
        resolution,
        True,
        True,
        5,
        0,
    )
    return _parse_texgen_export(inp_path, resolution, dtype=dtype)


def _normalised_abs_dots(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.abs(np.einsum("ij,ij->i", left, right))
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(
        right, axis=1
    )
    return numerator / np.maximum(denominator, np.finfo(left.dtype).eps)


def compare_reference(
    cpu: CpuReference,
    gpu_data: Any,
    gpu_orientation: Any,
) -> Dict[str, Any]:
    """Compare occupancy, yarn labels, and both material frame axes."""
    gpu_yarn_id = np.asarray(gpu_data.yarn_id)
    occupancy_mismatch = float(
        np.mean((cpu.yarn_id >= 0) != (gpu_yarn_id >= 0))
    )
    yarn_mismatch = float(np.mean(cpu.yarn_id != gpu_yarn_id))
    _common, cpu_positions, gpu_positions = np.intersect1d(
        cpu.voxel_indices,
        np.asarray(gpu_orientation.voxel_indices),
        assume_unique=True,
        return_indices=True,
    )
    if len(cpu_positions):
        matching_yarn = (
            cpu.yarn_ids[cpu_positions]
            == np.asarray(gpu_orientation.yarn_ids)[gpu_positions]
        )
        cpu_positions = cpu_positions[matching_yarn]
        gpu_positions = gpu_positions[matching_yarn]
    if not len(cpu_positions):
        minimum_axis_dot = 0.0
    else:
        dot1 = _normalised_abs_dots(
            cpu.orientation1[cpu_positions],
            np.asarray(gpu_orientation.orientation1)[gpu_positions],
        )
        dot2 = _normalised_abs_dots(
            cpu.orientation2[cpu_positions],
            np.asarray(gpu_orientation.orientation2)[gpu_positions],
        )
        minimum_axis_dot = float(min(dot1.min(), dot2.min()))
    return {
        "occupancy_mismatch_fraction": occupancy_mismatch,
        "yarn_mismatch_fraction": yarn_mismatch,
        "minimum_axis_dot": minimum_axis_dot,
        "matched_orientation_voxels": int(len(cpu_positions)),
    }


def stiffness_reference_error(
    material_fields: Any,
    gpu_field: Any,
    default_yarn_c21: np.ndarray,
    overrides: Dict[int, np.ndarray],
    *,
    max_samples: int = 65536,
) -> float:
    """Check GPU C21 rotation against NumPy on a deterministic sample."""
    count = gpu_field.num_yarn_voxels
    if count == 0:
        return 0.0
    if count <= max_samples:
        sample = np.arange(count, dtype=np.int64)
    else:
        sample = np.linspace(
            0, count - 1, max_samples, dtype=np.int64
        )
    yarn_ids = np.asarray(gpu_field.yarn_ids)[sample]
    local = np.broadcast_to(default_yarn_c21, (len(sample), 21)).copy()
    for yarn_id, c21 in overrides.items():
        local[yarn_ids == yarn_id] = c21
    expected = material_fields.rotate_stiffness_c21(
        local,
        np.asarray(gpu_field.orientation1)[sample],
        np.asarray(gpu_field.orientation2)[sample],
    )
    actual = np.asarray(gpu_field.yarn_c21)[sample]
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).eps)
    return float(np.linalg.norm(actual - expected) / denominator)


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    )


def _git_commit(root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def environment_metadata(torch_mod: Any, device: str) -> Dict[str, Any]:
    """Collect reproducibility metadata without optional third-party tools."""
    device_obj = torch_mod.device(device)
    gpu_name = None
    driver = None
    if device_obj.type == "cuda":
        gpu_name = torch_mod.cuda.get_device_name(device_obj)
        driver = getattr(torch_mod._C, "_cuda_getDriverVersion", lambda: None)()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_mod.__version__,
        "device": str(device_obj),
        "gpu": gpu_name,
        "cuda_runtime": getattr(torch_mod.version, "cuda", None),
        "cuda_driver": driver,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }


def _make_record(
    *,
    case: str,
    resolution: int,
    mode: str,
    dtype: str,
    gpu_times: Sequence[float],
    cpu_times: Optional[Sequence[float]],
    correctness: Dict[str, Any],
    stiffness_error: float,
    gpu_allocated: int,
    gpu_reserved: int,
    output_bytes: int,
) -> Dict[str, Any]:
    cpu_summary = (
        None if cpu_times is None else summarize_timings(cpu_times)
    )
    gpu_summary = summarize_timings(
        gpu_times,
        cpu_median=(
            None if cpu_summary is None else cpu_summary["median_s"]
        ),
    )
    record = {
        "case": case,
        "resolution": resolution,
        "mode": mode,
        "dtype": dtype,
        "voxels": resolution ** 3,
        "gpu": gpu_summary,
        "cpu": cpu_summary,
        "speedup": gpu_summary["speedup"],
        "stiffness_relative_error": stiffness_error,
        "gpu_peak_allocated_bytes": gpu_allocated,
        "gpu_peak_reserved_bytes": gpu_reserved,
        "output_bytes": output_bytes,
        "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    record.update(correctness)
    record["correctness"] = _record_correct(
        dict(record, correctness=True)
    )
    return record


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute all selected benchmark cases and return a JSON-ready report."""
    import torch
    from pytexgen import __version__ as pytexgen_version
    from pytexgen.gpu_voxelizer import extract_snapshot_bundle
    from pytexgen import material_fields

    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup must be >= 0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    matrix_c21 = material_fields.isotropic_stiffness_c21(
        3.5e9, 0.35
    ).astype(np_dtype)
    default_yarn_c21 = material_fields.orthotropic_stiffness_c21(
        150e9, 10e9, 10e9,
        0.25, 0.25, 0.30,
        5e9, 5e9, 3.8e9,
    ).astype(np_dtype)
    overrides = {3: (1.1 * default_yarn_c21).astype(np_dtype)}

    if args.keep_output:
        output_root = Path("build") / "gpu_material_fields_outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(
            prefix="pytexgen-material-fields-"
        )
        output_root = Path(temporary.name)

    records = []
    try:
        for case in ACCEPTANCE_CASES:
            textile = build_textile(case)
            bundle = extract_snapshot_bundle(textile)
            yarn_count = bundle.num_yarns
            for resolution in args.resolutions:
                print(
                    f"[benchmark] {case} {resolution}^3 "
                    f"({resolution ** 3:,} voxels, {yarn_count} yarns)"
                )
                case_dir = output_root / case / str(resolution)
                case_dir.mkdir(parents=True, exist_ok=True)

                def gpu_compute():
                    return material_fields.voxelize_textile_material_fields(
                        textile,
                        nx=resolution,
                        ny=resolution,
                        nz=resolution,
                        backend="torch",
                        device=str(device),
                        output="backend",
                        dtype=args.dtype,
                        chunk_voxels=args.chunk_voxels,
                        verbose=False,
                        matrix_stiffness=matrix_c21,
                        default_yarn_stiffness=default_yarn_c21,
                        yarn_stiffness_by_id=overrides,
                    )

                (
                    gpu_result,
                    gpu_compute_times,
                    peak_allocated,
                    peak_reserved,
                ) = timed_cuda(
                    gpu_compute,
                    args.repeat,
                    args.warmup,
                    torch_mod=torch,
                    device=str(device),
                )
                gpu_data_torch, gpu_field_torch = gpu_result
                gpu_data = gpu_data_torch.to("numpy")
                gpu_orientation = gpu_data.sparse_orientation
                gpu_field = gpu_field_torch.to("numpy")
                stiffness_error = stiffness_reference_error(
                    material_fields,
                    type(
                        "CombinedField",
                        (),
                        {
                            "num_yarn_voxels": gpu_field.num_yarn_voxels,
                            "yarn_ids": gpu_field.yarn_ids,
                            "orientation1": gpu_orientation.orientation1,
                            "orientation2": gpu_orientation.orientation2,
                            "yarn_c21": gpu_field.yarn_c21,
                        },
                    )(),
                    default_yarn_c21,
                    overrides,
                )

                cpu_compute_reference = None
                cpu_compute_times = None
                if not args.skip_cpu:
                    cpu_compute_reference, cpu_compute_times = timed_wall(
                        lambda: texgen_point_information_reference(
                            textile,
                            bundle.aabb,
                            resolution,
                            chunk_voxels=args.cpu_chunk_voxels,
                            dtype=np_dtype,
                        ),
                        args.repeat,
                    )
                    compute_correctness = compare_reference(
                        cpu_compute_reference, gpu_data, gpu_orientation
                    )
                else:
                    compute_correctness = {
                        "occupancy_mismatch_fraction": 0.0,
                        "yarn_mismatch_fraction": 0.0,
                        "minimum_axis_dot": 1.0,
                        "matched_orientation_voxels": (
                            gpu_orientation.num_yarn_voxels
                        ),
                    }
                compute_record = _make_record(
                    case=case,
                    resolution=resolution,
                    mode="compute",
                    dtype=args.dtype,
                    gpu_times=gpu_compute_times,
                    cpu_times=cpu_compute_times,
                    correctness=compute_correctness,
                    stiffness_error=stiffness_error,
                    gpu_allocated=peak_allocated,
                    gpu_reserved=peak_reserved,
                    output_bytes=0,
                )
                compute_record["yarn_count"] = yarn_count
                compute_record["gpu_phase_timings_s"] = dict(
                    gpu_data_torch.timings
                )
                records.append(compute_record)

                gpu_persist_dir = case_dir / "gpu_fields"

                def gpu_practical():
                    data, field = gpu_compute()
                    material_fields.save_material_field_bundle(
                        gpu_persist_dir, data.sparse_orientation, field
                    )
                    return data, field

                _gpu_saved, gpu_practical_times = timed_wall(
                    gpu_practical, args.repeat
                )
                gpu_output_bytes = _directory_size(gpu_persist_dir)

                cpu_practical_reference = None
                cpu_practical_times = None
                if not args.skip_cpu:
                    cpu_practical_reference, cpu_practical_times = timed_wall(
                        lambda: texgen_save_reference(
                            textile,
                            case_dir / "texgen_export",
                            resolution,
                            dtype=np_dtype,
                        ),
                        args.repeat,
                    )
                    practical_correctness = compare_reference(
                        cpu_practical_reference, gpu_data, gpu_orientation
                    )
                else:
                    practical_correctness = dict(compute_correctness)
                practical_record = _make_record(
                    case=case,
                    resolution=resolution,
                    mode="practical",
                    dtype=args.dtype,
                    gpu_times=gpu_practical_times,
                    cpu_times=cpu_practical_times,
                    correctness=practical_correctness,
                    stiffness_error=stiffness_error,
                    gpu_allocated=peak_allocated,
                    gpu_reserved=peak_reserved,
                    output_bytes=gpu_output_bytes,
                )
                practical_record["yarn_count"] = yarn_count
                practical_record["gpu_phase_timings_s"] = dict(
                    _gpu_saved[0].timings
                )
                records.append(practical_record)
                print(
                    f"  compute={compute_record['gpu']['median_s']:.3f}s "
                    f"({compute_record['speedup']}x), "
                    f"practical={practical_record['gpu']['median_s']:.3f}s "
                    f"({practical_record['speedup']}x), "
                    f"mismatch={compute_correctness['occupancy_mismatch_fraction']:.3%}"
                )
    finally:
        if temporary is not None:
            temporary.cleanup()

    report = {
        "format": "pytexgen.gpu_material_fields_benchmark",
        "format_version": 1,
        "environment": dict(
            environment_metadata(torch, str(device)),
            pytexgen=pytexgen_version,
        ),
        "parameters": {
            "resolutions": list(args.resolutions),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "device": str(device),
            "dtype": args.dtype,
            "chunk_voxels": args.chunk_voxels,
            "cpu_chunk_voxels": args.cpu_chunk_voxels,
            "skip_cpu": args.skip_cpu,
        },
        "records": records,
    }
    report["acceptance"] = evaluate_acceptance(records)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[128, 256]
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--chunk-voxels", type=int, default=65536)
    parser.add_argument("--cpu-chunk-voxels", type=int, default=65536)
    parser.add_argument(
        "--json-out", type=Path, default=Path("build/material_fields.json")
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.resolutions):
        parser.error("resolutions must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[benchmark] report: {args.json_out}")
    acceptance = report["acceptance"]
    print(
        "[benchmark] acceptance: "
        + ("PASS" if acceptance["passed"] else "FAIL")
    )
    for failure in acceptance["failures"]:
        print(f"  - {failure}")
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
