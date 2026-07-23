"""Check PyTexGen simulation-sample accuracy, zero-copy handoff, and speed."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from bench_gpu_material_fields import (
    _git_commit,
    build_textile,
    compare_reference,
    environment_metadata,
    texgen_save_reference,
    timed_cuda,
    timed_wall,
)


SCHEMA = "pytexgen.simulation_interop_benchmark/v1"
REQUIRED_REPORT_FIELDS = (
    "schema",
    "device",
    "resolution",
    "num_voxels",
    "resident_sparse_bytes",
    "acdm_dense_bytes",
    "handoff_host_transfer_bytes",
    "dlpack_pointer_shared",
    "phase_equal_numpy",
    "acdm_dense_max_abs_error",
    "pytexgen_seconds",
    "texgen_cpu_seconds",
    "speedup_vs_texgen_cpu",
    "accepted",
)


def _array_nbytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    return int(np.asarray(value).nbytes)


def _resident_bytes(fields: Mapping[str, Any]) -> int:
    """Count resident field bytes once when field aliases share an address."""
    regions = {}
    for value in fields.values():
        nbytes = _array_nbytes(value)
        if hasattr(value, "data_ptr"):
            key = ("torch", str(value.device))
            start = int(value.data_ptr())
        else:
            array = np.asarray(value)
            key = ("numpy", "cpu")
            start = int(array.__array_interface__["data"][0])
        regions.setdefault(key, []).append((start, start + nbytes))

    total = 0
    for intervals in regions.values():
        stop = None
        for start, end in sorted(intervals):
            if stop is None or start > stop:
                total += end - start
                stop = end
            elif end > stop:
                total += end - stop
                stop = end
    return int(total)


def measure_dlpack_handoff(
    fields: Mapping[str, Any],
    *,
    torch_mod: Any,
) -> dict[str, Any]:
    """Consume Torch fields through DLPack and account for any moved storage."""
    details = {}
    consumers = []
    transferred = 0
    all_shared = True
    for name, source in fields.items():
        nbytes = _array_nbytes(source)
        shared = False
        source_device = str(getattr(source, "device", "cpu"))
        consumer_device = None
        consumer_pointer = None
        try:
            consumer = torch_mod.from_dlpack(source)
            consumers.append(consumer)
            consumer_device = str(consumer.device)
            consumer_pointer = int(consumer.data_ptr())
            shared = bool(
                hasattr(source, "data_ptr")
                and int(source.data_ptr()) == consumer_pointer
                and source_device == consumer_device
            )
        except (BufferError, RuntimeError, TypeError):
            shared = False
        if not shared:
            transferred += nbytes
            all_shared = False
        details[name] = {
            "bytes": nbytes,
            "source_device": source_device,
            "consumer_device": consumer_device,
            "source_pointer": (
                int(source.data_ptr())
                if hasattr(source, "data_ptr")
                else None
            ),
            "consumer_pointer": consumer_pointer,
            "pointer_shared": shared,
        }
    return {
        "pointer_shared": all_shared,
        "host_transfer_bytes": int(transferred),
        "fields": details,
    }


def _acceptance_failures(
    report: Mapping[str, Any],
    *,
    min_speedup: float,
) -> list[str]:
    failures = []
    if not bool(report.get("dlpack_pointer_shared", False)):
        failures.append("DLPack consumer does not share every source pointer")
    if int(report.get("handoff_host_transfer_bytes", -1)) != 0:
        failures.append("simulation handoff performs a full-field transfer")
    if not bool(report.get("phase_equal_numpy", False)):
        failures.append("phase IDs differ from the NumPy reference")
    dense_error = float(
        report.get("acdm_dense_max_abs_error", math.inf)
    )
    dense_tolerance = float(report.get("acdm_dense_tolerance", 0.0))
    if (
        not bool(report.get("acdm_dense_equal", True))
        or dense_error > dense_tolerance
    ):
        failures.append(
            f"ACDM dense stiffness error {dense_error:g} exceeds "
            f"{dense_tolerance:g}"
        )
    if float(
        report.get("texgen_cpu_occupancy_mismatch_fraction", 0.0)
    ) > 0.005:
        failures.append("occupancy differs from the TexGen CPU reference")
    if float(
        report.get("texgen_cpu_yarn_mismatch_fraction", 0.0)
    ) > 0.005:
        failures.append("yarn IDs differ from the TexGen CPU reference")
    if float(report.get("texgen_cpu_minimum_axis_dot", 1.0)) < 0.999:
        failures.append("material directions differ from the TexGen CPU reference")
    speedup = float(report.get("speedup_vs_texgen_cpu", -math.inf))
    if speedup < min_speedup:
        failures.append(
            f"speedup {speedup:.3f}x is below required {min_speedup:.3f}x"
        )
    return failures


def build_report(
    metrics: Mapping[str, Any],
    *,
    dtype: str,
    min_speedup: float,
) -> dict[str, Any]:
    """Build the stable JSON record and apply all acceptance gates."""
    report = dict(metrics)
    report["schema"] = SCHEMA
    report["dtype"] = str(dtype)
    report["minimum_speedup"] = float(min_speedup)
    failures = _acceptance_failures(report, min_speedup=min_speedup)
    report["acceptance_failures"] = failures
    report["accepted"] = not failures
    missing = [
        name for name in REQUIRED_REPORT_FIELDS if name not in report
    ]
    if missing:
        raise ValueError(
            "missing benchmark report fields: " + ", ".join(missing)
        )
    return report


def _dense_accuracy(sample: Any, *, dtype: str) -> dict[str, Any]:
    import torch

    dense = sample.array(
        "stiffness.yarn_c21",
        layout="acdm",
        copy=True,
    )
    sample_numpy = sample.to("numpy")
    reference = sample_numpy.array(
        "stiffness.yarn_c21",
        layout="acdm",
        copy=True,
    )
    actual = dense.detach().cpu().numpy()
    rtol, atol = (
        (1e-10, 1e-12) if dtype == "float64" else (1e-5, 1e-6)
    )
    max_error = (
        0.0
        if reference.size == 0
        else float(np.max(np.abs(actual - reference)))
    )
    scale = (
        0.0
        if reference.size == 0
        else float(np.max(np.abs(reference)))
    )
    return {
        "dense": dense,
        "sample_numpy": sample_numpy,
        "max_error": max_error,
        "tolerance": float(atol + rtol * scale),
        "equal": bool(np.allclose(actual, reference, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run a representative textile through GPU and original CPU paths."""
    import torch
    from pytexgen import __version__ as pytexgen_version
    from pytexgen.acdm_solver import to_acdm_phase_ids
    from pytexgen.material_fields import (
        isotropic_stiffness_c21,
        orthotropic_stiffness_c21,
    )
    from pytexgen.simulation_sample import (
        MaterialTable,
        voxelize_textile_simulation_sample,
    )

    if args.repeat < 3:
        raise ValueError("repeat must be at least 3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    table_numpy = MaterialTable(
        c21=np.stack(
            (
                isotropic_stiffness_c21(3.5, 0.35),
                orthotropic_stiffness_c21(
                    150.0,
                    10.0,
                    10.0,
                    0.25,
                    0.25,
                    0.30,
                    5.0,
                    5.0,
                    3.8,
                ),
            )
        ).astype(np_dtype),
        material_ids=np.asarray((0, 7), dtype=np.int32),
        unit="GPa",
        names=("matrix", "yarn"),
    )
    materials = table_numpy.to(
        "torch",
        device=str(device),
        dtype=args.dtype,
    )
    textile = build_textile("plain_2x2")

    def pytexgen_path():
        return voxelize_textile_simulation_sample(
            textile,
            materials=materials,
            default_yarn_material_id=7,
            nx=args.resolution,
            ny=args.resolution,
            nz=args.resolution,
            backend="torch",
            device=str(device),
            output="backend",
            dtype=args.dtype,
            chunk_voxels=65536,
            verbose=False,
        )

    sample, pytexgen_times, peak_allocated, peak_reserved = timed_cuda(
        pytexgen_path,
        args.repeat,
        1,
        torch_mod=torch,
        device=str(device),
    )

    with tempfile.TemporaryDirectory(
        prefix="pytexgen-simulation-interop-"
    ) as cpu_output:
        cpu_output_path = Path(cpu_output)

        def texgen_cpu_path():
            return texgen_save_reference(
                textile,
                cpu_output_path,
                args.resolution,
                dtype=np_dtype,
            )

        cpu_reference, cpu_times = timed_wall(
            texgen_cpu_path,
            args.repeat,
            warmup=1,
        )
    pytexgen_seconds = float(statistics.median(pytexgen_times))
    texgen_cpu_seconds = float(statistics.median(cpu_times))
    speedup = (
        texgen_cpu_seconds / pytexgen_seconds
        if pytexgen_seconds > 0.0
        else math.inf
    )

    fields = sample.as_dict(copy=False)
    handoff = measure_dlpack_handoff(fields, torch_mod=torch)
    dense_accuracy = _dense_accuracy(sample, dtype=args.dtype)
    sample_numpy = dense_accuracy["sample_numpy"]
    phase = to_acdm_phase_ids(sample.voxels, batch=True)
    phase_numpy = to_acdm_phase_ids(sample_numpy.voxels, batch=True)
    phase_equal = bool(
        np.array_equal(phase.detach().cpu().numpy(), phase_numpy)
    )
    cpu_correctness = compare_reference(
        cpu_reference,
        sample_numpy.voxels,
        sample_numpy.orientation,
    )

    root = Path(__file__).resolve().parent
    voxel_acdm_root = root.parent / "Voxel-ACDM"
    metrics = {
        "device": str(device),
        "resolution": int(args.resolution),
        "num_voxels": int(args.resolution**3),
        "resident_sparse_bytes": _resident_bytes(fields),
        "acdm_dense_bytes": _array_nbytes(dense_accuracy["dense"]),
        "handoff_host_transfer_bytes": handoff["host_transfer_bytes"],
        "dlpack_pointer_shared": handoff["pointer_shared"],
        "phase_equal_numpy": phase_equal,
        "acdm_dense_max_abs_error": dense_accuracy["max_error"],
        "acdm_dense_tolerance": dense_accuracy["tolerance"],
        "acdm_dense_equal": dense_accuracy["equal"],
        "pytexgen_seconds": pytexgen_seconds,
        "texgen_cpu_seconds": texgen_cpu_seconds,
        "speedup_vs_texgen_cpu": float(speedup),
        "pytexgen_timings_seconds": [
            float(value) for value in pytexgen_times
        ],
        "texgen_cpu_timings_seconds": [
            float(value) for value in cpu_times
        ],
        "warmup_runs_per_path": 1,
        "repeat": int(args.repeat),
        "texgen_cpu_reference_mode": (
            "CRectangularVoxelMesh.SaveVoxelMesh plus .eld/.ori parsing"
        ),
        "dlpack_fields": handoff["fields"],
        "tolerances": {
            "rtol": dense_accuracy["rtol"],
            "atol": dense_accuracy["atol"],
        },
        "texgen_cpu_occupancy_mismatch_fraction": cpu_correctness[
            "occupancy_mismatch_fraction"
        ],
        "texgen_cpu_yarn_mismatch_fraction": cpu_correctness[
            "yarn_mismatch_fraction"
        ],
        "texgen_cpu_minimum_axis_dot": cpu_correctness[
            "minimum_axis_dot"
        ],
        "matched_orientation_voxels": cpu_correctness[
            "matched_orientation_voxels"
        ],
        "gpu_peak_allocated_bytes": int(peak_allocated),
        "gpu_peak_reserved_bytes": int(peak_reserved),
        "environment": dict(
            environment_metadata(torch, str(device)),
            pytexgen=pytexgen_version,
            voxel_acdm_commit=(
                _git_commit(voxel_acdm_root)
                if voxel_acdm_root.is_dir()
                else None
            ),
        ),
    }
    return build_report(
        metrics,
        dtype=args.dtype,
        min_speedup=args.min_speedup,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--min-speedup", type=float, default=5.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.resolution < 1:
        parser.error("resolution must be positive")
    if args.repeat < 3:
        parser.error("repeat must be at least 3")
    if args.min_speedup < 0.0:
        parser.error("min-speedup must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[interop] report: {args.json_out}")
    print(
        "[interop] "
        f"PyTexGen={report['pytexgen_seconds']:.6f}s, "
        f"TexGen CPU={report['texgen_cpu_seconds']:.6f}s, "
        f"speedup={report['speedup_vs_texgen_cpu']:.2f}x, "
        f"handoff={report['handoff_host_transfer_bytes']} B"
    )
    print("[interop] acceptance: " + ("PASS" if report["accepted"] else "FAIL"))
    for failure in report["acceptance_failures"]:
        print(f"  - {failure}")
    return 1 if args.check and not report["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
