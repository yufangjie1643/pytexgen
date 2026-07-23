#!/usr/bin/env python3
"""Checked native-shard, compressed-NPZ, and CUDA-prefetch benchmark."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from TexGen.training_data import (
    DatasetQualityPolicy,
    TrainingDatasetSchema,
    TrainingFieldSpec,
    VOXEL_ORDER,
)
from TexGen.training_io import (
    SimulationDataset,
    SimulationDatasetWriter,
    audit_simulation_dataset,
)
from TexGen.torch_training import (
    CudaPrefetcher,
    make_simulation_dataloader,
)

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None


def _positive_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def evaluate_benchmark(
    *,
    native_samples_per_second: float,
    npz_samples_per_second: float,
    synchronous_wait_seconds: float,
    prefetch_wait_seconds: float,
    expected_h2d_bytes: int,
    observed_h2d_bytes: int,
    training_loss: float,
    min_read_speedup: float,
    min_prefetch_speedup: float,
) -> Mapping[str, Any]:
    """Evaluate benchmark thresholds without performing any I/O."""
    native_rate = _positive_finite(
        native_samples_per_second, "native_samples_per_second"
    )
    npz_rate = _positive_finite(
        npz_samples_per_second, "npz_samples_per_second"
    )
    synchronous_wait = _nonnegative_finite(
        synchronous_wait_seconds, "synchronous_wait_seconds"
    )
    prefetch_wait = _nonnegative_finite(
        prefetch_wait_seconds, "prefetch_wait_seconds"
    )
    expected_bytes = _nonnegative_integer(
        expected_h2d_bytes, "expected_h2d_bytes"
    )
    observed_bytes = _nonnegative_integer(
        observed_h2d_bytes, "observed_h2d_bytes"
    )
    read_threshold = _positive_finite(
        min_read_speedup, "min_read_speedup"
    )
    prefetch_threshold = _positive_finite(
        min_prefetch_speedup, "min_prefetch_speedup"
    )
    if isinstance(training_loss, bool) or not isinstance(
        training_loss, (int, float)
    ):
        raise TypeError("training_loss must be numeric")
    loss = float(training_loss)
    loss_finite = math.isfinite(loss)

    read_speedup = native_rate / npz_rate
    if prefetch_wait == 0.0:
        prefetch_speedup = (
            1.0 if synchronous_wait == 0.0 else sys.float_info.max
        )
    else:
        prefetch_speedup = synchronous_wait / prefetch_wait

    failed = []
    if read_speedup < read_threshold:
        failed.append("read_speedup")
    if prefetch_speedup < prefetch_threshold:
        failed.append("prefetch_speedup")
    if observed_bytes != expected_bytes:
        failed.append("h2d_bytes")
    if not loss_finite:
        failed.append("training_loss")
    return {
        "passed": not failed,
        "failed_metrics": failed,
        "read_speedup": read_speedup,
        "prefetch_speedup": prefetch_speedup,
        "native_samples_per_second": native_rate,
        "npz_samples_per_second": npz_rate,
        "synchronous_wait_seconds": synchronous_wait,
        "prefetch_wait_seconds": prefetch_wait,
        "expected_h2d_bytes": expected_bytes,
        "observed_h2d_bytes": observed_bytes,
        "h2d_bytes_match": expected_bytes == observed_bytes,
        "training_loss": loss if loss_finite else None,
        "training_loss_finite": loss_finite,
        "min_read_speedup": read_threshold,
        "min_prefetch_speedup": prefetch_threshold,
    }


class _ArraySample:
    """Minimal public SimulationSample protocol used by the benchmark."""

    storage = "numpy"
    device = "cpu"

    def __init__(self, material_id: np.ndarray):
        self.materials = SimpleNamespace(unit="GPa")
        self.voxels = SimpleNamespace(
            shape=material_id.shape, order=VOXEL_ORDER
        )
        self._material_id = material_id

    def array(self, name: str, *, copy: bool = False, **_: Any):
        if name != "voxel.material_id":
            raise KeyError(name)
        return (
            np.array(self._material_id, copy=True, order="C")
            if copy
            else self._material_id
        )


def _isotropic_c21(youngs_modulus: float, poisson_ratio: float):
    lam = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    np.fill_diagonal(matrix[:3, :3], lam + 2.0 * mu)
    np.fill_diagonal(matrix[3:, 3:], mu)
    return np.asarray(
        [
            matrix[row, column]
            for row in range(6)
            for column in range(row, 6)
        ],
        dtype=np.float64,
    )


def _record(
    index: int, resolution: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    z, y, x = np.ogrid[
        :resolution, :resolution, :resolution
    ]
    phase = (
        ((x + 2 * index) % 17 < 7)
        ^ ((y + index) % 19 < 8)
        ^ ((z + 3 * index) % 23 < 9)
    )
    rng = np.random.default_rng(seed + index)
    perturbation = rng.random(
        (resolution, resolution, resolution)
    ) < 0.01
    material_id = np.where(
        phase ^ perturbation, 7, 0
    ).astype(np.int32)
    # Guarantee a distinct physical geometry digest even at tiny resolutions.
    material_id.reshape(-1)[index % material_id.size] = 8 + index
    effective_c21 = _isotropic_c21(
        10.0 + 0.05 * index,
        0.20 + 0.001 * (index % 20),
    )
    return material_id, effective_c21


def _schema(resolution: int, shard_size: int):
    return TrainingDatasetSchema(
        inputs=(
            TrainingFieldSpec(
                "voxel.material_id",
                "input",
                "fixed",
                "int32",
                (resolution, resolution, resolution),
                semantic="material_id_grid",
            ),
        ),
        targets=(
            TrainingFieldSpec(
                "effective_c21",
                "target",
                "fixed",
                "float64",
                (21,),
                "GPa",
                "engineering_voigt_c21",
            ),
        ),
        grid_shape=(resolution, resolution, resolution),
        voxel_order=VOXEL_ORDER,
        shard_size=shard_size,
        statistics_fields=("effective_c21",),
    )


def _fixture_configuration(
    *, resolution: int, sample_count: int, shard_size: int, seed: int
) -> Mapping[str, Any]:
    return {
        "schema": "pytexgen.training_benchmark_fixture",
        "version": 1,
        "resolution": resolution,
        "sample_count": sample_count,
        "shard_size": shard_size,
        "seed": seed,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_fixture(
    root: Path,
    *,
    resolution: int,
    sample_count: int,
    shard_size: int,
    seed: int,
) -> Tuple[Path, Path, Mapping[str, Any]]:
    configuration = _fixture_configuration(
        resolution=resolution,
        sample_count=sample_count,
        shard_size=shard_size,
        seed=seed,
    )
    native = root / "native"
    compressed = root / "compressed_npz"
    metadata_path = root / "benchmark_fixture.json"
    if metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != configuration:
            raise ValueError(
                "existing benchmark fixture configuration does not match"
            )
        if not (native / "dataset.json").is_file() or not compressed.is_dir():
            raise ValueError("existing benchmark fixture is incomplete")
        return native, compressed, configuration
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"benchmark root is non-empty and unrecognized: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    schema = _schema(resolution, shard_size)
    quality = DatasetQualityPolicy(
        maximum_solver_residual=1e-8,
        require_unique_geometry=True,
    )
    with SimulationDatasetWriter.create(
        native,
        schema=schema,
        quality=quality,
        generation=dict(configuration),
    ) as writer:
        for index in range(sample_count):
            material_id, effective_c21 = _record(
                index, resolution, seed
            )
            writer.append(
                _ArraySample(material_id),
                targets={"effective_c21": effective_c21},
                sample_id=f"sample-{index:06d}",
                group_id=f"geometry-{index:06d}",
                split="train",
                provenance={
                    "solver_commit": "benchmark-fixture",
                    "element_formulation": "periodic-c3d8",
                    "arithmetic_dtype": "float32",
                    "tolerance": 1e-8,
                    "maximum_residual": 1e-10,
                    "iteration_count": 20,
                    "wall_time_seconds": 0.01,
                    "target_units": {"effective_c21": "GPa"},
                },
            )

    compressed.mkdir()
    for start in range(0, sample_count, shard_size):
        stop = min(start + shard_size, sample_count)
        material_values = []
        target_values = []
        sample_ids = []
        for index in range(start, stop):
            material_id, effective_c21 = _record(
                index, resolution, seed
            )
            material_values.append(material_id)
            target_values.append(effective_c21)
            sample_ids.append(f"sample-{index:06d}")
        np.savez_compressed(
            compressed / f"chunk_{start // shard_size:05d}.npz",
            sample_ids=np.asarray(sample_ids),
            voxel_material_id=np.stack(material_values),
            effective_c21=np.stack(target_values),
        )
    _write_json(metadata_path, configuration)
    return native, compressed, configuration


def _npz_paths(compressed: Path) -> Tuple[Path, ...]:
    paths = tuple(sorted(compressed.glob("chunk_*.npz")))
    if not paths:
        raise ValueError(f"no compressed chunks found under {compressed}")
    return paths


def _check_fixture(
    native: Path,
    compressed: Path,
) -> Mapping[str, Any]:
    audit = audit_simulation_dataset(native, verify="sample")
    dataset = SimulationDataset(
        native,
        split="train",
        inputs=("voxel.material_id",),
        targets=("effective_c21",),
        verify="manifest",
    )
    index = 0
    for path in _npz_paths(compressed):
        with np.load(path, allow_pickle=False) as chunk:
            ids = chunk["sample_ids"]
            materials = chunk["voxel_material_id"]
            targets = chunk["effective_c21"]
            for row in range(len(ids)):
                example = dataset[index]
                if str(ids[row]) != example.sample_id:
                    raise AssertionError("sample IDs differ between formats")
                np.testing.assert_array_equal(
                    materials[row],
                    example.inputs["voxel.material_id"],
                )
                np.testing.assert_array_equal(
                    targets[row],
                    example.targets["effective_c21"],
                )
                index += 1
    if index != len(dataset):
        raise AssertionError("sample counts differ between formats")
    return {
        "values_equal": True,
        "native_audit": audit,
    }


def _drop_file_cache(paths: Iterable[Path]) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        return False
    applied = False
    for path in paths:
        if not path.is_file():
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(
                    descriptor, 0, 0, os.POSIX_FADV_DONTNEED
                )
                applied = True
            finally:
                os.close(descriptor)
        except OSError:
            continue
    return applied


def _read_native_once(native: Path) -> Tuple[int, int]:
    dataset = SimulationDataset(
        native,
        split="train",
        inputs=("voxel.material_id",),
        targets=("effective_c21",),
        verify="manifest",
    )
    checksum = 0
    logical_bytes = 0
    for index in range(len(dataset)):
        example = dataset[index]
        material_id = np.asarray(
            example.inputs["voxel.material_id"]
        )
        effective_c21 = np.asarray(
            example.targets["effective_c21"]
        )
        checksum ^= int(
            np.sum(material_id, dtype=np.int64)
        ) + index
        checksum ^= int(
            np.sum(
                effective_c21.view(np.uint64),
                dtype=np.uint64,
            )
        )
        logical_bytes += material_id.nbytes + effective_c21.nbytes
    return checksum, logical_bytes


def _read_npz_once(compressed: Path) -> Tuple[int, int, int]:
    checksum = 0
    logical_bytes = 0
    sample_count = 0
    for path in _npz_paths(compressed):
        with np.load(path, allow_pickle=False) as chunk:
            material_id = chunk["voxel_material_id"]
            effective_c21 = chunk["effective_c21"]
            checksum ^= int(np.sum(material_id, dtype=np.int64))
            checksum ^= int(
                np.sum(
                    effective_c21.view(np.uint64),
                    dtype=np.uint64,
                )
            )
            logical_bytes += material_id.nbytes + effective_c21.nbytes
            sample_count += int(material_id.shape[0])
    return checksum, logical_bytes, sample_count


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _measure_reads(
    native: Path,
    compressed: Path,
    *,
    sample_count: int,
    repeat: int,
) -> Mapping[str, Any]:
    native_files = tuple(native.rglob("*.npy"))
    npz_files = _npz_paths(compressed)
    native_times = []
    npz_times = []
    native_checksum = None
    npz_checksum = None
    logical_bytes = 0
    cache_drop = {"native": False, "npz": False}
    for iteration in range(repeat):
        gc.collect()
        if iteration == 0:
            cache_drop["native"] = _drop_file_cache(native_files)
        started = time.perf_counter()
        checksum, current_bytes = _read_native_once(native)
        native_times.append(time.perf_counter() - started)
        native_checksum = checksum
        logical_bytes = current_bytes

        gc.collect()
        if iteration == 0:
            cache_drop["npz"] = _drop_file_cache(npz_files)
        started = time.perf_counter()
        checksum, current_bytes, observed_count = _read_npz_once(
            compressed
        )
        npz_times.append(time.perf_counter() - started)
        npz_checksum = checksum
        if observed_count != sample_count or current_bytes != logical_bytes:
            raise AssertionError("logical NPZ/native payloads differ")
    if native_checksum is None or npz_checksum is None:
        raise AssertionError("read benchmark did not execute")

    def summarize(times):
        warm = times[1:] if len(times) > 1 else times
        median = statistics.median(warm)
        return {
            "first_pass_seconds": times[0],
            "warm_median_seconds": median,
            "warm_p90_seconds": _percentile(warm, 90),
            "samples_per_second": sample_count / median,
            "logical_mb_per_second": (
                logical_bytes / 1_000_000.0 / median
            ),
            "all_seconds": times,
        }

    return {
        "native": summarize(native_times),
        "compressed_npz": summarize(npz_times),
        "logical_bytes_per_pass": logical_bytes,
        "cache_drop_applied": cache_drop,
        "checksums_computed": True,
    }


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS and some BSDs report bytes.
    return value if sys.platform == "darwin" else value * 1024


def _make_loader(
    native: Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    dataset = SimulationDataset(
        native,
        split="train",
        inputs=("voxel.material_id",),
        targets=("effective_c21",),
        verify="manifest",
    )
    return make_simulation_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        seed=23,
    )


def _measure_device_pipeline(
    native: Path,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the device pipeline benchmark"
        ) from exc
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pin_memory = device.type == "cuda"

    synchronous_loader = _make_loader(
        native,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    expected_h2d_bytes = 0
    synchronous_waits = []
    synchronous_started = time.perf_counter()
    for cpu_batch in synchronous_loader:
        if device.type == "cuda":
            expected_h2d_bytes += cpu_batch.nbytes
            started = time.perf_counter()
            cpu_batch.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            synchronous_waits.append(time.perf_counter() - started)
    synchronous_total = time.perf_counter() - synchronous_started
    synchronous_wait = (
        sum(synchronous_waits) if device.type == "cuda" else 0.0
    )

    loader = _make_loader(
        native,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    prefetcher = CudaPrefetcher(loader, device=device)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 21),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    yield_waits = []
    loss_value = math.nan
    gradients_finite = False
    iterator = iter(prefetcher)
    batch_index = 0
    prefetch_started = time.perf_counter()
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        yield_waits.append(time.perf_counter() - started)
        if batch_index == 0:
            features = (
                batch.inputs["voxel.material_id"] != 0
            ).to(torch.float32).unsqueeze(1)
            targets = batch.targets["effective_c21"].to(
                torch.float32
            )
            prediction = model(features)
            loss = torch.nn.functional.mse_loss(
                prediction, targets
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradients_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            )
            optimizer.step()
            loss_value = float(loss.detach().item())
        else:
            # Consume the tensor on the current stream so record_stream()
            # coverage and prefetch ordering remain part of the benchmark.
            _ = batch.inputs["voxel.material_id"].sum()
        batch_index += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefetch_total = time.perf_counter() - prefetch_started
    if not gradients_finite:
        loss_value = math.nan
    observed_h2d_bytes = (
        prefetcher.transferred_bytes if device.type == "cuda" else 0
    )
    return {
        "device": str(device),
        "batch_count": batch_index,
        "pin_memory": pin_memory,
        "pinned_logical_bytes": expected_h2d_bytes,
        "expected_h2d_bytes": expected_h2d_bytes,
        "observed_h2d_bytes": observed_h2d_bytes,
        "synchronous": {
            "total_seconds": synchronous_total,
            "h2d_wait_seconds": synchronous_wait,
            "batch_wait_median_seconds": (
                statistics.median(synchronous_waits)
                if synchronous_waits
                else 0.0
            ),
            "batch_wait_p90_seconds": (
                _percentile(synchronous_waits, 90)
                if synchronous_waits
                else 0.0
            ),
        },
        "prefetch": {
            "total_seconds": prefetch_total,
            "stream_wait_seconds": prefetcher.wait_seconds,
            "yield_wait_median_seconds": (
                statistics.median(yield_waits)
                if yield_waits
                else 0.0
            ),
            "yield_wait_p90_seconds": (
                _percentile(yield_waits, 90)
                if yield_waits
                else 0.0
            ),
            "recorded_tensors": prefetcher.recorded_tensors,
        },
        "training_loss": (
            loss_value if math.isfinite(loss_value) else None
        ),
        "training_gradients_finite": gradients_finite,
    }


def _directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8"
            ).splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _environment(device_name: str) -> Mapping[str, Any]:
    try:
        package_version = importlib.metadata.version("pytexgen")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source"
    torch_data: Dict[str, Any] = {
        "version": None,
        "cuda_runtime": None,
        "gpu": None,
    }
    try:
        import torch

        torch_data["version"] = torch.__version__
        torch_data["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            torch_data["gpu"] = torch.cuda.get_device_name(
                torch.device(device_name)
                if torch.device(device_name).type == "cuda"
                else 0
            )
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "numpy": np.__version__,
        "torch": torch_data,
        "pytexgen": package_version,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }


def _positive_cli(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checked native mmap shards with compressed NPZ and "
            "measure the selected-field CUDA prefetch path."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "fixture root to create or reuse; a temporary directory is "
            "used when omitted"
        ),
    )
    parser.add_argument("--samples", type=_positive_cli, default=32)
    parser.add_argument("--resolution", type=_positive_cli, default=64)
    parser.add_argument("--batch-size", type=_positive_cli, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repeat", type=_positive_cli, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-read-speedup", type=float, default=1.5)
    parser.add_argument(
        "--min-prefetch-speedup", type=float, default=1.0
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare all values, audit hashes, and enforce thresholds",
    )
    return parser


def _run(args: argparse.Namespace, root: Path) -> Mapping[str, Any]:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.batch_size > args.samples:
        raise ValueError("--batch-size cannot exceed --samples")
    native, compressed, configuration = _prepare_fixture(
        root,
        resolution=args.resolution,
        sample_count=args.samples,
        shard_size=args.batch_size,
        seed=args.seed,
    )
    correctness = (
        _check_fixture(native, compressed)
        if args.check
        else {"values_equal": None, "native_audit": None}
    )
    reads = _measure_reads(
        native,
        compressed,
        sample_count=args.samples,
        repeat=args.repeat,
    )
    device = _measure_device_pipeline(
        native,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    loss = device["training_loss"]
    evaluation = evaluate_benchmark(
        native_samples_per_second=reads["native"][
            "samples_per_second"
        ],
        npz_samples_per_second=reads["compressed_npz"][
            "samples_per_second"
        ],
        synchronous_wait_seconds=device["synchronous"][
            "h2d_wait_seconds"
        ],
        prefetch_wait_seconds=device["prefetch"][
            "stream_wait_seconds"
        ],
        expected_h2d_bytes=device["expected_h2d_bytes"],
        observed_h2d_bytes=device["observed_h2d_bytes"],
        training_loss=math.nan if loss is None else loss,
        min_read_speedup=args.min_read_speedup,
        min_prefetch_speedup=args.min_prefetch_speedup,
    )
    return {
        "schema": "pytexgen.training_data_benchmark",
        "version": 1,
        "passed": evaluation["passed"],
        "fixture": dict(configuration),
        "paths": {
            "root": str(root.resolve()),
            "native": str(native.resolve()),
            "compressed_npz": str(compressed.resolve()),
        },
        "environment": _environment(args.device),
        "correctness": correctness,
        "storage": {
            "native_directory_bytes": _directory_bytes(native),
            "compressed_npz_bytes": _directory_bytes(compressed),
            "selected_logical_bytes": reads[
                "logical_bytes_per_pass"
            ],
        },
        "reads": reads,
        "device_pipeline": device,
        "peak_rss_bytes": _peak_rss_bytes(),
        "evaluation": evaluation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _positive_finite(args.min_read_speedup, "--min-read-speedup")
    _positive_finite(
        args.min_prefetch_speedup, "--min-prefetch-speedup"
    )
    if args.dataset is None:
        with tempfile.TemporaryDirectory(
            prefix="pytexgen-training-benchmark-"
        ) as directory:
            report = _run(args, Path(directory))
    else:
        report = _run(args, args.dataset)
    text = json.dumps(
        report, sort_keys=True, indent=2, allow_nan=False
    )
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.check and not report["passed"]:
        failures = ", ".join(
            report["evaluation"]["failed_metrics"]
        )
        print(
            f"benchmark acceptance failed: {failures}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
