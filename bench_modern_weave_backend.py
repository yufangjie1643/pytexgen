"""Benchmark pytexgen.modern voxelization backends.

The benchmark uses the Python-first ``PlainWeave2D`` model and reports
end-to-end ``voxelize_model_data(...)`` time for numpy workers and optional
Torch/CUDA. It also checks every result against the numpy serial baseline.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Iterable

import numpy as np

from pytexgen.modern import PlainWeave2D, voxelize_model_data


def _parse_resolution(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("*", "x").split("x")
    if len(parts) == 1:
        n = int(parts[0])
        return n, n, n
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("resolution must be N or NxYxZ")
    return tuple(int(part) for part in parts)


def _parse_workers(value: str):
    return value.lower() if value.lower() == "auto" else int(value)


def _best_time(fn, repeat: int, sync=None):
    result = None
    timings = []
    for _ in range(repeat):
        gc.collect()
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        result = fn()
        if sync is not None:
            sync()
        timings.append(time.perf_counter() - t0)
    return min(timings), result


def _mvox_per_second(resolution: tuple[int, int, int], seconds: float) -> float:
    voxels = resolution[0] * resolution[1] * resolution[2]
    return voxels / seconds / 1_000_000


def _format_workers(workers) -> str:
    return str(workers)


def _bench_numpy(
    model: PlainWeave2D,
    resolution: tuple[int, int, int],
    workers,
    repeat: int,
    chunk_voxels: int,
):
    return _best_time(
        lambda: voxelize_model_data(
            model,
            resolution=resolution,
            backend="numpy",
            workers=workers,
            chunk_voxels=chunk_voxels,
        ),
        repeat=repeat,
    )


def _bench_cuda(
    model: PlainWeave2D,
    resolution: tuple[int, int, int],
    repeat: int,
    chunk_voxels: int,
):
    import torch

    return _best_time(
        lambda: voxelize_model_data(
            model,
            resolution=resolution,
            backend="torch",
            device="cuda",
            chunk_voxels=chunk_voxels,
        ),
        repeat=repeat,
        sync=torch.cuda.synchronize,
    )


def _print_row(name: str, seconds: float, resolution: tuple[int, int, int], speedup: float, equal: bool):
    print(
        f"{name:<20} {seconds:>9.6f}s  "
        f"{_mvox_per_second(resolution, seconds):>8.3f} Mvox/s  "
        f"{speedup:>6.2f}x  equal={equal}"
    )


def run_benchmark(
    resolutions: Iterable[tuple[int, int, int]],
    workers_values,
    repeat: int,
    chunk_voxels: int,
    include_cuda: bool,
):
    model = PlainWeave2D(width=4, height=4, spacing=1.0, thickness=0.2)

    torch = None
    if include_cuda:
        try:
            import torch as torch_module

            torch = torch_module
        except Exception:
            torch = None

    for resolution in resolutions:
        voxels = resolution[0] * resolution[1] * resolution[2]
        print(f"\nresolution={resolution} voxels={voxels:,} repeat={repeat}")
        serial_time, serial_data = _bench_numpy(
            model,
            resolution,
            workers=1,
            repeat=repeat,
            chunk_voxels=chunk_voxels,
        )
        _print_row("numpy workers=1", serial_time, resolution, 1.0, True)

        for workers in workers_values:
            if workers == 1:
                continue
            seconds, data = _bench_numpy(
                model,
                resolution,
                workers=workers,
                repeat=repeat,
                chunk_voxels=chunk_voxels,
            )
            equal = np.array_equal(serial_data.yarn_id, data.yarn_id)
            _print_row(
                f"numpy workers={_format_workers(workers)}",
                seconds,
                resolution,
                serial_time / seconds,
                equal,
            )

        if not include_cuda:
            continue
        if torch is None:
            print("torch cuda           unavailable: torch import failed")
            continue
        if not torch.cuda.is_available():
            print(f"torch cuda           unavailable: torch={torch.__version__} cuda_available=False")
            continue
        seconds, data = _bench_cuda(model, resolution, repeat=repeat, chunk_voxels=chunk_voxels)
        equal = np.array_equal(serial_data.yarn_id, data.to_numpy().yarn_id)
        _print_row("torch cuda", seconds, resolution, serial_time / seconds, equal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[(64, 64, 64), (128, 128, 128)],
        help="Resolution list, each as N or NxYxZ.",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=_parse_workers,
        default=["auto", 1, 2, 4, 8, 12],
        help='Numpy worker counts to test, e.g. "auto 1 2 4 8 12".',
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chunk-voxels", type=int, default=8192)
    parser.add_argument("--include-cuda", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        resolutions=args.resolutions,
        workers_values=args.workers,
        repeat=args.repeat,
        chunk_voxels=args.chunk_voxels,
        include_cuda=args.include_cuda,
    )


if __name__ == "__main__":
    main()
